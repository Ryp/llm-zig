const std = @import("std");
const weights = @import("weights.zig");

pub const Config = struct {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
    rope_theta: f32 = 10000.0,
};

pub const Transformer = struct {
    config: Config,
    weights: weights.TransformerWeights,
    state: RunState,
};

pub fn build_transformer(allocator: *std.mem.Allocator, t: *Transformer, checkpoint_path: [:0]const u8) !void {
    t.weights = try weights.open_weights_from_file(checkpoint_path, &t.config);
    errdefer weights.close_weights_from_file(t.weights);

    try create_run_state(allocator, &t.state, t.config);
    errdefer destroy_run_state(allocator, &t.state);
}

pub fn free_transformer(allocator: *std.mem.Allocator, t: *Transformer) void {
    destroy_run_state(allocator, &t.state);

    weights.close_weights_from_file(t.weights);
}

pub fn forward(arg_transformer: *Transformer, token: u16, pos: usize) []f32 {
    const p = arg_transformer.config;
    const w = arg_transformer.weights;
    const s = &arg_transformer.state;

    const x = s.x;

    const dim = p.dim;
    const kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    const kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery
    const head_size = dim / p.n_heads;

    // copy the token embedding into x
    const content_offset = token * dim;
    const content_size = dim;
    const content_row = w.token_embedding_table[content_offset .. content_offset + content_size];

    @memcpy(x, content_row);

    // forward all the layers
    for (0..p.n_layers) |layer_index| {
        // attention rmsnorm
        const rms_att_weight_offset = layer_index * dim;
        const rms_att_weight_layer = w.rms_att_weight[rms_att_weight_offset .. rms_att_weight_offset + dim];
        rmsnorm(s.xb, x, rms_att_weight_layer);

        // key and value point to the kv cache
        const layer_kv_offset = layer_index * p.seq_len * kv_dim; // kv cache layer offset for convenience
        const kv_offset = layer_kv_offset + pos * kv_dim;

        s.k = s.key_cache[kv_offset .. kv_offset + kv_dim]; // FIXME size
        s.v = s.value_cache[kv_offset .. kv_offset + kv_dim]; // FIXME size

        const layer_wq = w.wq[layer_index * p.dim * p.dim.. (layer_index + 1) * (p.dim * p.dim)]; // FIXME size
        const layer_wk = w.wk[layer_index * p.dim * kv_dim..(layer_index + 1) * (p.dim * kv_dim)];
        const layer_wv = w.wv[layer_index * p.dim * kv_dim..(layer_index + 1) * (p.dim * kv_dim)];

        // qkv matmuls for this position
        matmul_1d(s.q, s.xb, layer_wq);
        matmul_1d(s.k, s.xb, layer_wk);
        matmul_1d(s.v, s.xb, layer_wv);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        {
            var i: usize = 0;
            while (i < dim) : (i += @as(c_int, 2)) {
                const head_dim = i % head_size;
                const freq = 1.0 / std.math.pow(f32, p.rope_theta, @as(f32, @floatFromInt(head_dim)) / @as(f32, @floatFromInt(head_size)));
                const val = freq * @as(f32, @floatFromInt(pos));
                const fcr = std.math.cos(val);
                const fci = std.math.sin(val);

                rotate_pair(s.q[i .. i + 2], fcr, fci);

                if (i < kv_dim) {
                    rotate_pair(s.k[i .. i + 2], fcr, fci);
                }
            }
        }

        // multihead attention. iterate over all heads
        for (0..p.n_heads) |h| {
            // get the query vector for this head
            const q = s.q[h * head_size ..(h + 1) * head_size];
            // attention scores for this head
            const att = s.att[h * p.seq_len ..]; // FIXME
            // iterate over all timesteps, including the current one
            for (0..pos + 1) |t| {
                // get the key vector for this head and at this timestep
                const k_offset = layer_kv_offset + t * kv_dim + (h / kv_mul) * head_size;
                const k = s.key_cache[k_offset ..k_offset + head_size];
                // calculate the attention score as the dot product of q and k
                //std.debug.assert(head_size == q.len);
                //std.debug.assert(head_size == k.len);

                var score = dot(q, k);
                score *= 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_size)));

                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att[0..pos + 1]);

            // weighted sum of the values, store back into xb
            const xb = s.xb[h * head_size .. (h + 1) * head_size];

            @memset(xb, 0.0);

            for (0..pos + 1) |t| {
                // get the value vector for this head and at this timestep
                const v = s.value_cache[layer_kv_offset + t * kv_dim + (h / kv_mul) * head_size ..]; // FIXME
                // get the attention weight for this timestep
                const a = att[t];
                // accumulate the weighted value into xb
                for (0..head_size) |i| {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul_1d(s.xb2, s.xb, w.wo[layer_index * dim * dim..(layer_index + 1) * dim * dim]);

        // residual connection back into x
        for (0..dim) |i| {
            x[i] += s.xb2[i];
        }

        // ffn rmsnorm
        const rms_ffn_weight_offset = layer_index * dim;
        const rms_ffn_weight_layer = w.rms_ffn_weight[rms_ffn_weight_offset .. rms_ffn_weight_offset + dim];
        rmsnorm(s.xb, x, rms_ffn_weight_layer);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul_1d(s.hb, s.xb, w.w1[layer_index * dim * p.hidden_dim..(layer_index + 1) * dim * p.hidden_dim]);
        matmul_1d(s.hb2, s.xb, w.w3[layer_index * dim * p.hidden_dim..(layer_index + 1) * dim * p.hidden_dim]);

        // SwiGLU non-linearity
        for (0..p.hidden_dim) |i| {
            var val = s.hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0 / (1.0 + std.math.exp(-val)));
            // elementwise multiply with w3(x)
            val *= s.hb2[i];
            s.hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul_1d(s.xb, s.hb, w.w2[layer_index * dim * p.hidden_dim..(layer_index + 1) * dim * p.hidden_dim]);

        // residual connection
        for (0..dim) |i| {
            x[i] += s.xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w.rms_final_weight);

    // classifier into logits
    // Assuming we're reusing the same embedding table for this step
    matmul_1d(s.logits, x, w.token_embedding_table);

    return s.logits;
}

fn rotate_pair(pair: []f32, r: f32, i: f32) void {
    const rslt: [2]f32 = .{
        pair[0] * r - pair[1] * i,
        pair[0] * i + pair[1] * r,
    };

    @memcpy(pair, &rslt);
}

fn rmsnorm(o: []f32, x: []f32, weight: []const f32) void {
    std.debug.assert(o.len == weight.len);
    std.debug.assert(x.len == weight.len);

    // calculate sum of squares
    var ss: f32 = 0.0;
    for (0..x.len) |j| {
        ss += x[j] * x[j];
    }

    ss /= @as(f32, @floatFromInt(x.len));
    ss += 0.00001;
    ss = 1.0 / std.math.sqrt(ss);

    // normalize and scale
    for (0..x.len) |j| {
        o[j] = weight[j] * (ss * x[j]);
    }
}

pub fn softmax(x: []f32) void {
    // find max value (for numerical stability)
    var max_val = x[0];
    for (0..x.len) |i| {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    // exp and sum
    var sum: f32 = 0.0;
    for (0..x.len) |i| {
        x[i] = std.math.exp(x[i] - max_val);
        sum += x[i];
    }

    const sum_inv = 1.0 / sum;

    // normalize
    for (0..x.len) |i| {
        x[i] *= sum_inv;
    }
}

// W (d,n) @ x (n,) -> xout (d,)
fn matmul_1d(o: []f32, x: []f32, w: []const f32) void {
    std.debug.assert(o.len * x.len == w.len);

    for (0..o.len) |i| {
        var val: f32 = 0.0;

        for (0..x.len) |j| {
            val += w[i * x.len + j] * x[j];
        }

        o[i] = val;
    }
}

fn dot(o: []f32, x: []f32) f32 {
    var acc: f32 = 0.0;

    for (o, x) |l_o, l_x| {
        acc += l_o * l_x;
    }

    return acc;
}

const RunState = struct {
    x: []f32,
    xb: []f32,
    xb2: []f32,
    hb: []f32,
    hb2: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    att: []f32,
    logits: []f32,
    key_cache: []f32,
    value_cache: []f32,
};

fn create_run_state(allocator: *std.mem.Allocator, s: *RunState, p: Config) !void {
    const kv_dim = @divTrunc(p.dim * p.n_kv_heads, p.n_heads);

    s.x = try allocator.alloc(f32, p.dim);
    errdefer allocator.free(s.x);

    s.xb = try allocator.alloc(f32, p.dim);
    errdefer allocator.free(s.xb);

    s.xb2 = try allocator.alloc(f32, p.dim);
    errdefer allocator.free(s.xb2);

    s.hb = try allocator.alloc(f32, p.hidden_dim);
    errdefer allocator.free(s.hb);

    s.hb2 = try allocator.alloc(f32, p.hidden_dim);
    errdefer allocator.free(s.hb2);

    s.q = try allocator.alloc(f32, p.dim);
    errdefer allocator.free(s.q);

    s.key_cache = try allocator.alloc(f32, (p.n_layers * p.seq_len) * kv_dim);
    errdefer allocator.free(s.key_cache);

    s.value_cache = try allocator.alloc(f32, (p.n_layers * p.seq_len) * kv_dim);
    errdefer allocator.free(s.value_cache);

    s.att = try allocator.alloc(f32, p.n_heads * p.seq_len);
    errdefer allocator.free(s.att);

    s.logits = try allocator.alloc(f32, p.vocab_size);
    errdefer allocator.free(s.logits);
}

fn destroy_run_state(allocator: *std.mem.Allocator, s: *RunState) void {
    allocator.free(s.x);
    allocator.free(s.xb);
    allocator.free(s.xb2);
    allocator.free(s.hb);
    allocator.free(s.hb2);
    allocator.free(s.q);
    allocator.free(s.att);
    allocator.free(s.logits);
    allocator.free(s.key_cache);
    allocator.free(s.value_cache);
}

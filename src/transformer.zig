const std = @import("std");
const weights = @import("weights.zig");
const tensor = @import("tensor/tensor.zig");
const layout = @import("tensor/layout.zig");

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

pub fn create_transformer(allocator: *std.mem.Allocator, t: *Transformer, checkpoint_path: [:0]const u8) !void {
    t.weights = try weights.open_weights_from_file(checkpoint_path, &t.config);
    errdefer weights.close_weights_from_file(t.weights);

    try create_run_state(allocator, &t.state, t.config);
    errdefer destroy_run_state(allocator, &t.state);
}

pub fn destroy_transformer(allocator: *std.mem.Allocator, t: *Transformer) void {
    destroy_run_state(allocator, &t.state);

    weights.close_weights_from_file(t.weights);
}

pub fn forward(arg_transformer: *Transformer, token: u16, pos: usize) []f32 {
    const p = arg_transformer.config;
    const w = arg_transformer.weights;
    const s = &arg_transformer.state;

    const dim = p.dim;
    const kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    const kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery
    const head_size = dim / p.n_heads;

    // copy the token embedding into x
    const token_embedding = w.token_embedding_table.sub_tensor(0, token);

    @memcpy(s.x.raw_data, token_embedding.raw_data);

    // forward all the layers
    for (0..p.n_layers) |layer_index| {
        // attention rmsnorm
        const rms_att_weight_layer = w.rms_att_weight.sub_tensor(0, layer_index);
        rmsnorm(s.xb.raw_data, s.x.raw_data, rms_att_weight_layer.raw_data);

        // key and value point to the kv cache
        const k_layer = s.key_cache.sub_tensor(0, layer_index);
        const v_layer = s.value_cache.sub_tensor(0, layer_index);

        const t_k = k_layer.sub_tensor(0, pos);
        const t_v = v_layer.sub_tensor(0, pos);

        const layer_wq = w.wq.sub_tensor(0, layer_index);
        const layer_wk = w.wk.sub_tensor(0, layer_index);
        const layer_wv = w.wv.sub_tensor(0, layer_index);

        // qkv matmuls for this position
        matmul_1d(s.q, s.xb, layer_wq);
        matmul_1d(t_k, s.xb, layer_wk);
        matmul_1d(t_v, s.xb, layer_wv);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        {
            var i: usize = 0;
            while (i < dim) : (i += @as(c_int, 2)) {
                const head_dim = i % head_size;
                const freq = 1.0 / std.math.pow(f32, p.rope_theta, @as(f32, @floatFromInt(head_dim)) / @as(f32, @floatFromInt(head_size)));
                const val = freq * @as(f32, @floatFromInt(pos));
                const fcr = std.math.cos(val);
                const fci = std.math.sin(val);

                rotate_pair(s.q.raw_data[i .. i + 2], fcr, fci);

                if (i < kv_dim) {
                    rotate_pair(t_k.raw_data[i .. i + 2], fcr, fci);
                }
            }
        }

        // multihead attention. iterate over all heads
        for (0..p.n_heads) |head_index| {
            // get the query vector for this head
            const q = s.q.raw_data[head_index * head_size ..(head_index + 1) * head_size];
            // attention scores for this head
            const att = s.att.sub_tensor(0, head_index).raw_data; // FIXME
            // iterate over all timesteps, including the current one
            for (0..pos + 1) |t| {
                // get the key vector for this head and at this timestep
                const k_offset = t * kv_dim + (head_index / kv_mul) * head_size;
                const key = k_layer.raw_data[k_offset ..k_offset + head_size];
                // calculate the attention score as the dot product of q and k
                //std.debug.assert(head_size == q.len);
                //std.debug.assert(head_size == k.len);

                var score = dot(q, key);
                score *= 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_size)));

                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att[0..pos + 1]);

            // weighted sum of the values, store back into xb
            const xb = s.xb.raw_data[head_index * head_size .. (head_index + 1) * head_size];

            @memset(xb, 0.0);

            for (0..pos + 1) |t| {
                // get the value vector for this head and at this timestep
                const v = v_layer.raw_data[t * kv_dim + (head_index / kv_mul) * head_size ..]; // FIXME
                // get the attention weight for this timestep
                const a = att[t];
                // accumulate the weighted value into xb
                for (0..head_size) |i| {
                    xb[i] += a * v[i];
                }
            }
        }

        const wo_layer = w.wo.sub_tensor(0, layer_index);

        // final matmul to get the output of the attention
        matmul_1d(s.xb2, s.xb, wo_layer);

        // residual connection back into x
        for (s.x.raw_data, s.xb2.raw_data) |*x_elt, xb2_elt| {
            x_elt.* += xb2_elt;
        }

        // ffn rmsnorm
        const rms_ffn_weight_layer = w.rms_ffn_weight.sub_tensor(0, layer_index);

        rmsnorm(s.xb.raw_data, s.x.raw_data, rms_ffn_weight_layer.raw_data);

        const w1_layer = w.w1.sub_tensor(0, layer_index);
        const w2_layer = w.w2.sub_tensor(0, layer_index);
        const w3_layer = w.w3.sub_tensor(0, layer_index);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul_1d(s.hb, s.xb, w1_layer);
        matmul_1d(s.hb2, s.xb, w3_layer);

        // SwiGLU non-linearity
        for (s.hb.raw_data, s.hb2.raw_data) |*hb_elt, hb2_elt| {
            var val = hb_elt.*;
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0 / (1.0 + std.math.exp(-val)));
            // elementwise multiply with w3(x)
            val *= hb2_elt;
            hb_elt.* = val;
        }

        // final matmul to get the output of the ffn
        matmul_1d(s.xb, s.hb, w2_layer);

        // residual connection
        for (s.x.raw_data, s.xb.raw_data) |*x_elt, xb_elt| {
            x_elt.* += xb_elt;
        }
    }

    // final rmsnorm
    rmsnorm(s.x.raw_data, s.x.raw_data, w.rms_final_weight.raw_data);

    // classifier into logits
    // Assuming we're reusing the same embedding table for this step
    matmul_1d(s.logits, s.x, w.token_embedding_table);

    return s.logits.raw_data;
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
// FIXME Tensor n2 must be const!
fn matmul_1d(o: tensor.Tensor(f32, 1), x: tensor.Tensor(f32, 1), w: tensor.ConstTensor(f32, 2)) void {
    const o_size = o.layout.shape[0];
    const x_size = x.layout.shape[0];
    std.debug.assert(o_size * x_size == w.layout.shape[0] * w.layout.shape[1]);

    for (0..o_size) |i| {
        var val: f32 = 0.0;

        for (0..x_size) |j| {
            // const w_slice = ;
            val += w.raw_data[i * x_size + j] * x.raw_data[j];
        }

        o.raw_data[i] = val;
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
    x: tensor.Tensor(f32, 1),
    xb: tensor.Tensor(f32, 1),
    xb2: tensor.Tensor(f32, 1),
    hb: tensor.Tensor(f32, 1),
    hb2: tensor.Tensor(f32, 1),
    q: tensor.Tensor(f32, 1),
    att: tensor.Tensor(f32, 2),
    logits: tensor.Tensor(f32, 1),
    key_cache: tensor.Tensor(f32, 3),
    value_cache: tensor.Tensor(f32, 3),
};

fn create_run_state(allocator: *std.mem.Allocator, s: *RunState, p: Config) !void {
    const kv_dim = @divTrunc(p.dim * p.n_kv_heads, p.n_heads);

    s.x = tensor.Tensor(f32, 1).init(layout.right(1, .{p.dim}), try allocator.alloc(f32, p.dim));
    errdefer allocator.free(s.x.raw_data);

    s.xb = tensor.Tensor(f32, 1).init(layout.right(1, .{p.dim}), try allocator.alloc(f32, p.dim));
    errdefer allocator.free(s.xb.raw_data);

    s.xb2 = tensor.Tensor(f32, 1).init(layout.right(1, .{p.dim}), try allocator.alloc(f32, p.dim));
    errdefer allocator.free(s.xb2.raw_data);

    s.hb = tensor.Tensor(f32, 1).init(layout.right(1, .{p.hidden_dim}), try allocator.alloc(f32, p.hidden_dim));
    errdefer allocator.free(s.hb.raw_data);

    s.hb2 = tensor.Tensor(f32, 1).init(layout.right(1, .{p.hidden_dim}), try allocator.alloc(f32, p.hidden_dim));
    errdefer allocator.free(s.hb2.raw_data);

    s.q = tensor.Tensor(f32, 1).init(layout.right(1, .{p.dim}), try allocator.alloc(f32, p.dim));
    errdefer allocator.free(s.q.raw_data);

    s.key_cache = tensor.Tensor(f32, 3).init(layout.right(3, .{p.n_layers, p.seq_len, kv_dim}), try allocator.alloc(f32, p.n_layers * p.seq_len * kv_dim));
    errdefer allocator.free(s.key_cache.raw_data);

    s.value_cache = tensor.Tensor(f32, 3).init(layout.right(3, .{p.n_layers, p.seq_len, kv_dim}), try allocator.alloc(f32, p.n_layers * p.seq_len * kv_dim));
    errdefer allocator.free(s.value_cache.raw_data);

    s.att = tensor.Tensor(f32, 2).init(layout.right(2, .{p.n_heads, p.seq_len}), try allocator.alloc(f32, p.n_heads * p.seq_len));
    errdefer allocator.free(s.att.raw_data);

    s.logits = tensor.Tensor(f32, 1).init(layout.right(1, .{p.vocab_size}), try allocator.alloc(f32, p.vocab_size));
    errdefer allocator.free(s.logits.raw_data);
}

fn destroy_run_state(allocator: *std.mem.Allocator, s: *RunState) void {
    allocator.free(s.x.raw_data);
    allocator.free(s.xb.raw_data);
    allocator.free(s.xb2.raw_data);
    allocator.free(s.hb.raw_data);
    allocator.free(s.hb2.raw_data);
    allocator.free(s.q.raw_data);
    allocator.free(s.att.raw_data);
    allocator.free(s.logits.raw_data);
    allocator.free(s.key_cache.raw_data);
    allocator.free(s.value_cache.raw_data);
}

const std = @import("std");
const llama = @import("llama.zig");

pub const Config = extern struct {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    seq_len: u32,
};

pub const Transformer = extern struct {
    config: Config,
    weights: llama.TransformerWeights,
    state: RunState,
    fd: c_int,
    data: [*]f32,
    file_size: isize,
};

pub const RunState = extern struct {
    x: [*]f32,
    xb: [*]f32,
    xb2: [*]f32,
    hb: [*]f32,
    hb2: [*]f32,
    q: [*]f32,
    k: [*]f32,
    v: [*]f32,
    att: [*]f32,
    logits: [*]f32,
    key_cache: [*]f32,
    value_cache: [*]f32,
};

pub fn build_transformer(arg_t: *Transformer, arg_checkpoint_path: [:0]const u8) void {
    var t = arg_t;
    _ = &t;
    var checkpoint_path = arg_checkpoint_path;
    _ = &checkpoint_path;
    llama.read_checkpoint(checkpoint_path, &t.config, &t.weights, &t.fd, &t.data, &t.file_size);
    malloc_run_state(&t.state, t.config);
}

pub fn free_transformer(arg_t: *Transformer) void {
    var t = arg_t;
    _ = &t;
    //if (t.*.data != @as([*c]f32, @ptrCast(@as(?*anyopaque, @ptrFromInt(@as(usize, 0) -% 1))))) {
        _ = llama.munmap(@as(?*anyopaque, @ptrCast(t.data)), @as(usize, @bitCast(t.file_size)));
    //}
    if (t.fd != -@as(c_int, 1)) {
        _ = llama.close(t.fd);
    }
    free_run_state(&t.state);
}

pub fn rmsnorm(arg_o: [*c]f32, arg_x: [*c]f32, arg_weight: [*c]f32, arg_size: usize) void {
    var o = arg_o;
    _ = &o;
    var x = arg_x;
    _ = &x;
    var weight = arg_weight;
    _ = &weight;
    var size = arg_size;
    _ = &size;
    var ss: f32 = 0.0;
    _ = &ss;
    {
        var j: c_int = 0;
        _ = &j;
        while (j < size) : (j += 1) {
            ss += (blk: {
                const tmp = j;
                if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* * (blk: {
                const tmp = j;
                if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).*;
        }
    }
    ss /= @as(f32, @floatFromInt(size));
    ss += 0.00001;
    ss = 1.0 / std.math.sqrt(ss);
    {
        var j: c_int = 0;
        _ = &j;
        while (j < size) : (j += 1) {
            (blk: {
                const tmp = j;
                if (tmp >= 0) break :blk o + @as(usize, @intCast(tmp)) else break :blk o - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* = (blk: {
                const tmp = j;
                if (tmp >= 0) break :blk weight + @as(usize, @intCast(tmp)) else break :blk weight - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* * (ss * (blk: {
                const tmp = j;
                if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).*);
        }
    }
}

pub fn softmax(arg_x: [*c]f32, size: usize) void {
    var x = arg_x;
    _ = &x;
    var max_val: f32 = x[@as(c_uint, @intCast(@as(c_int, 0)))];
    _ = &max_val;
    {
        var i: c_int = 1;
        _ = &i;
        while (i < size) : (i += 1) {
            if ((blk: {
                const tmp = i;
                if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* > max_val) {
                max_val = (blk: {
                    const tmp = i;
                    if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*;
            }
        }
    }
    var sum: f32 = 0.0;
    _ = &sum;
    {
        var i: c_int = 0;
        _ = &i;
        while (i < size) : (i += 1) {
            (blk: {
                const tmp = i;
                if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* = std.math.exp((blk: {
                const tmp = i;
                if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* - max_val);
            sum += (blk: {
                const tmp = i;
                if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).*;
        }
    }
    {
        var i: c_int = 0;
        _ = &i;
        while (i < size) : (i += 1) {
            (blk: {
                const tmp = i;
                if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* /= sum;
        }
    }
}

pub fn matmul(xout: [*c]f32, x: [*c]f32, w: [*c]f32, n: usize, d: usize) void {
    for (0..d) |i| {
        var val: f32 = 0.0;

        for (0..n) |j| {
            val += (blk: {
                const tmp = (i * n) + j;
                if (tmp >= 0) break :blk w + @as(usize, @intCast(tmp)) else break :blk w - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* * (blk: {
                const tmp = j;
                if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).*;
        }

        (blk: {
            const tmp = i;
            if (tmp >= 0) break :blk xout + @as(usize, @intCast(tmp)) else break :blk xout - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
        }).* = val;
    }
}

pub fn forward(arg_transformer: *Transformer, token: u16, pos: usize) [*]f32 {
    const p = arg_transformer.config;
    const w = arg_transformer.weights;
    const s = &arg_transformer.state;

    const x = s.x;

    const dim = p.dim;
    const kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    const kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery
    const head_size = dim / p.n_heads;

    // copy the token embedding into x
    const content_row: [*]f32 = w.token_embedding_table + token * dim;

    _ = llama.memcpy(x, content_row, dim * @sizeOf(f32));

    // forward all the layers
    for (0..p.n_layers) |layer_index|
    {
        // attention rmsnorm
        rmsnorm(s.xb, x, w.rms_att_weight + layer_index * dim, dim);

        // key and value point to the kv cache
        const layer_kv_offset = layer_index * p.seq_len * kv_dim; // kv cache layer offset for convenience
        s.k = s.key_cache + layer_kv_offset + pos * kv_dim;
        s.v = s.value_cache + layer_kv_offset + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s.q, s.xb, w.wq + layer_index * p.dim * p.dim, dim, dim);
        matmul(s.k, s.xb, w.wk + layer_index * p.dim * kv_dim, dim, kv_dim);
        matmul(s.v, s.xb, w.wv + layer_index * p.dim * kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        {
        var i: usize = 0;
        while (i < dim) : (i += @as(c_int, 2)) {
            const head_dim = i % head_size;
            const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32,@floatFromInt(head_dim)) / @as(f32,@floatFromInt(head_size)));
            const val = freq * @as(f32, @floatFromInt(pos));
            const fcr = std.math.cos(val);
            const fci = std.math.sin(val);
            const rotn: usize = if (i < kv_dim) 2 else 1; // how many vectors? 2 = q & k, 1 = q only
            for(0..rotn) |v| {
                const vec = if (v == 0) s.q else s.k; // the vector to rotate (query or key)
                const v0 = vec[i];
                const v1 = vec[i+1];

                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
        }

        // multihead attention. iterate over all heads
        for (0..p.n_heads) |h| {
            // get the query vector for this head
            const q = s.q + h * head_size;
            // attention scores for this head
            const att = s.att + h * p.seq_len;
            // iterate over all timesteps, including the current one
            for (0..pos + 1) |t| {
                // get the key vector for this head and at this timestep
                const k = s.key_cache + layer_kv_offset + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                var score: f32 = 0.0;
                for (0..head_size) |i| {
                    score += q[i] * k[i];
                }
                score *= 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_size)));
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            const xb = s.xb + h * head_size;

            _ = llama.memset(xb, 0, head_size * @sizeOf(f32));

            for (0..pos + 1) |t| {
                // get the value vector for this head and at this timestep
                const v = s.value_cache + layer_kv_offset + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                const a = att[t];
                // accumulate the weighted value into xb
                for (0..head_size) |i| {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s.xb2, s.xb, w.wo + layer_index * dim * dim, dim, dim);

        // residual connection back into x
        for (0..dim) |i| {
            x[i] += s.xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s.xb, x, w.rms_ffn_weight + layer_index * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s.hb, s.xb, w.w1 + layer_index * dim * p.hidden_dim, dim, p.hidden_dim);
        matmul(s.hb2, s.xb, w.w3 + layer_index * dim * p.hidden_dim, dim, p.hidden_dim);

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
        matmul(s.xb, s.hb, w.w2 + layer_index * dim * p.hidden_dim, p.hidden_dim, dim);

        // residual connection
        for (0..dim) |i| {
            x[i] += s.xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w.rms_final_weight, dim);

    // classifier into logits
    matmul(s.logits, x, w.wcls, p.dim, p.vocab_size);
    return s.logits;
}

pub fn malloc_run_state(s: *RunState, p: Config) void {
    var kv_dim: usize = @divTrunc(p.dim * p.n_kv_heads, p.n_heads);
    _ = &kv_dim;

    s.x = @as([*]f32, @ptrCast(@alignCast(llama.calloc(p.dim, @sizeOf(f32)))));
    s.xb = @as([*]f32, @ptrCast(@alignCast(llama.calloc(p.dim, @sizeOf(f32)))));
    s.xb2 = @as([*]f32, @ptrCast(@alignCast(llama.calloc(p.dim, @sizeOf(f32)))));
    s.hb = @as([*]f32, @ptrCast(@alignCast(llama.calloc(p.hidden_dim, @sizeOf(f32)))));
    s.hb2 = @as([*]f32, @ptrCast(@alignCast(llama.calloc(p.hidden_dim, @sizeOf(f32)))));
    s.q = @as([*]f32, @ptrCast(@alignCast(llama.calloc(p.dim, @sizeOf(f32)))));
    s.key_cache = @as([*]f32, @ptrCast(@alignCast(llama.calloc((p.n_layers * p.seq_len) * kv_dim, @sizeOf(f32)))));
    s.value_cache = @as([*]f32, @ptrCast(@alignCast(llama.calloc((p.n_layers * p.seq_len) * kv_dim, @sizeOf(f32)))));
    s.att = @as([*]f32, @ptrCast(@alignCast(llama.calloc(p.n_heads * p.seq_len, @sizeOf(f32)))));
    s.logits = @as([*]f32, @ptrCast(@alignCast(llama.calloc(p.vocab_size, @sizeOf(f32)))));
    // FIXME error handling
}

pub fn free_run_state(s: *RunState) void {
    llama.free(@as(?*anyopaque, @ptrCast(s.x)));
    llama.free(@as(?*anyopaque, @ptrCast(s.xb)));
    llama.free(@as(?*anyopaque, @ptrCast(s.xb2)));
    llama.free(@as(?*anyopaque, @ptrCast(s.hb)));
    llama.free(@as(?*anyopaque, @ptrCast(s.hb2)));
    llama.free(@as(?*anyopaque, @ptrCast(s.q)));
    llama.free(@as(?*anyopaque, @ptrCast(s.att)));
    llama.free(@as(?*anyopaque, @ptrCast(s.logits)));
    llama.free(@as(?*anyopaque, @ptrCast(s.key_cache)));
    llama.free(@as(?*anyopaque, @ptrCast(s.value_cache)));
}

const std = @import("std");
const weights = @import("weights.zig");
const tensor = @import("tensor/tensor.zig");
const layout = @import("tensor/layout.zig");

pub const Config = struct {
    vocab_size: usize,
    seq_len: usize,
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    hidden_dim: usize,
    n_layers: usize,
    rope_theta: f32 = 10000.0,
    rms_epsilon: f32 = 0.00001,
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

const RunState = struct {
    x: tensor.Tensor(f32, 1),
    xb: tensor.Tensor(f32, 1),
    xb2: tensor.Tensor(f32, 1),
    hb: tensor.Tensor(f32, 1),
    hb2: tensor.Tensor(f32, 1),
    q: tensor.Tensor(f32, 1),
    attn_score: tensor.Tensor(f32, 2),
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

    s.key_cache = tensor.Tensor(f32, 3).init(layout.right(3, .{ p.n_layers, p.seq_len, kv_dim }), try allocator.alloc(f32, p.n_layers * p.seq_len * kv_dim));
    errdefer allocator.free(s.key_cache.raw_data);

    s.value_cache = tensor.Tensor(f32, 3).init(layout.right(3, .{ p.n_layers, p.seq_len, kv_dim }), try allocator.alloc(f32, p.n_layers * p.seq_len * kv_dim));
    errdefer allocator.free(s.value_cache.raw_data);

    s.attn_score = tensor.Tensor(f32, 2).init(layout.right(2, .{ p.n_heads, p.seq_len }), try allocator.alloc(f32, p.n_heads * p.seq_len));
    errdefer allocator.free(s.attn_score.raw_data);

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
    allocator.free(s.attn_score.raw_data);
    allocator.free(s.logits.raw_data);
    allocator.free(s.key_cache.raw_data);
    allocator.free(s.value_cache.raw_data);
}

pub fn forward(arg_transformer: *Transformer, token: u16, pos: usize) []f32 {
    const p = arg_transformer.config;
    const w = arg_transformer.weights;
    const s = &arg_transformer.state;

    const kv_dim = @divExact((p.dim * p.n_kv_heads), p.n_heads);
    const kv_mul = @divExact(p.n_heads, p.n_kv_heads); // integer multiplier of the kv sharing in multiquery
    const head_size = @divExact(p.dim, p.n_heads);
    const head_size_inv: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_size)));

    // copy the token embedding into x
    const token_embedding = w.token_embedding_table.sub_tensor(0, token);

    @memcpy(s.x.raw_data, token_embedding.raw_data);

    // forward all the layers
    for (0..p.n_layers) |layer_index| {
        rms_norm(s.xb, s.x, w.rms_attn.sub_tensor(0, layer_index), p.rms_epsilon);

        // key and value point to the kv cache
        const k_layer = s.key_cache.sub_tensor(0, layer_index);
        const v_layer = s.value_cache.sub_tensor(0, layer_index);

        const k_pos = k_layer.sub_tensor(0, pos);
        const v_pos = v_layer.sub_tensor(0, pos);

        gemm(s.q, s.xb, w.wq.sub_tensor(0, layer_index));

        gemm(k_pos, s.xb, w.wk.sub_tensor(0, layer_index));
        gemm(v_pos, s.xb, w.wv.sub_tensor(0, layer_index));

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        {
            var i: usize = 0;
            while (i < p.dim) : (i += @as(c_int, 2)) {
                const current_head_dim = i % head_size;
                const freq = 1.0 / std.math.pow(f32, p.rope_theta, @as(f32, @floatFromInt(current_head_dim)) / @as(f32, @floatFromInt(head_size)));
                const val = freq * @as(f32, @floatFromInt(pos));

                // FIXME There's a way to get both sin/cos in a better way IIRC
                const fcr = std.math.cos(val);
                const fci = std.math.sin(val);

                rotate_pair(s.q.raw_data[i .. i + 2], fcr, fci);

                if (i < kv_dim) {
                    rotate_pair(k_pos.raw_data[i .. i + 2], fcr, fci);
                }
            }
        }

        // Grouped-query attention (GQA)
        for (0..p.n_heads) |head_index| {
            // Create aliased tensor to allow manipulating individual heads
            const q_head_wise = tensor.Tensor(f32, 2).init(layout.right(2, .{ p.n_heads, p.dim / p.n_heads }), s.q.raw_data);

            const q_head = q_head_wise.sub_tensor(0, head_index);
            const attn_score_head = s.attn_score.sub_tensor(0, head_index);

            const kv_head_index = head_index / kv_mul;

            // iterate over all timesteps, including the current one
            for (0..pos + 1) |t| {
                // get the key vector for this head and at this timestep
                const k_t_pos = k_layer.sub_tensor(0, t);
                const k_head_wise = tensor.Tensor(f32, 2).init(layout.right(2, .{ p.n_kv_heads, head_size }), k_t_pos.raw_data);
                const k_head = k_head_wise.sub_tensor(0, kv_head_index);

                // calculate the attention score as the dot product of q and k
                attn_score_head.raw_data[t] = dot(q_head, k_head) * head_size_inv;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(attn_score_head.raw_data[0 .. pos + 1]);

            // Create aliased tensor to allow manipulating individual heads
            const xb_head_wise = tensor.Tensor(f32, 2).init(layout.right(2, .{ p.n_heads, p.dim / p.n_heads }), s.xb.raw_data);
            const xb_head = xb_head_wise.sub_tensor(0, head_index);

            // weighted sum of the values, store back into xb
            @memset(xb_head.raw_data, 0.0);

            for (0..pos + 1) |t| {
                // get the value vector for this head and at this timestep
                const v_t_pos = v_layer.sub_tensor(0, t);
                const v_head_wise = tensor.Tensor(f32, 2).init(layout.right(2, .{ p.n_kv_heads, head_size }), v_t_pos.raw_data);
                const v_head = v_head_wise.sub_tensor(0, kv_head_index);

                // accumulate the weighted value into xb
                for (0..head_size) |i| {
                    xb_head.raw_data[i] += attn_score_head.raw_data[t] * v_head.raw_data[i];
                }
            }
        }

        // final matmul to get the output of the attention
        gemm(s.xb2, s.xb, w.wo.sub_tensor(0, layer_index));

        // residual connection back into x
        for (s.x.raw_data, s.xb2.raw_data) |*x_elt, xb2_elt| {
            x_elt.* += xb2_elt;
        }

        rms_norm(s.xb, s.x, w.rms_ffn.sub_tensor(0, layer_index), p.rms_epsilon);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        gemm(s.hb, s.xb, w.w1.sub_tensor(0, layer_index));
        gemm(s.hb2, s.xb, w.w3.sub_tensor(0, layer_index));

        swiglu(s.hb, s.hb2);

        // final matmul to get the output of the ffn
        gemm(s.xb, s.hb, w.w2.sub_tensor(0, layer_index));

        // residual connection
        for (s.x.raw_data, s.xb.raw_data) |*x_elt, xb_elt| {
            x_elt.* += xb_elt;
        }
    }

    rms_norm(s.x, s.x, w.rms_final, p.rms_epsilon);

    // classifier into logits
    // Assuming we're reusing the same embedding table for this step
    gemm(s.logits, s.x, w.token_embedding_table);

    return s.logits.raw_data;
}

fn rotate_pair(pair: []f32, r: f32, i: f32) void {
    const rslt: [2]f32 = .{
        pair[0] * r - pair[1] * i,
        pair[0] * i + pair[1] * r,
    };

    @memcpy(pair, &rslt);
}

fn rms_norm(o: tensor.Tensor(f32, 1), x: tensor.Tensor(f32, 1), w: tensor.ConstTensor(f32, 1), epsilon: f32) void {
    const len = o.layout.shape[0];

    // Make sure tensors are continuous
    std.debug.assert(o.layout.stride[0] == 1);
    std.debug.assert(x.layout.stride[0] == 1);
    // Make sure the shape is consistent
    std.debug.assert(o.layout.shape[0] == len);
    std.debug.assert(x.layout.shape[0] == len);
    std.debug.assert(o.raw_data.len == len);
    std.debug.assert(x.raw_data.len == len);

    std.debug.assert(w.layout.shape[0] == len);
    std.debug.assert(w.layout.shape[0] == len);

    // calculate sum of squares
    var ss: f32 = dot(x, x);
    ss /= @as(f32, @floatFromInt(len));
    ss = 1.0 / std.math.sqrt(ss + epsilon);

    // normalize and scale
    for (0..len) |i| {
        o.raw_data[i] = w.raw_data[i] * (ss * x.raw_data[i]);
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

fn swiglu(a: tensor.Tensor(f32, 1), b: tensor.Tensor(f32, 1)) void {
    const len = a.layout.shape[0];

    // Make sure tensors are continuous
    std.debug.assert(a.layout.stride[0] == 1);
    std.debug.assert(b.layout.stride[0] == 1);
    // Make sure the shape is consistent
    std.debug.assert(a.layout.shape[0] == len);
    std.debug.assert(b.layout.shape[0] == len);

    for (a.raw_data, b.raw_data) |*hb_elt, hb2_elt| {
        var val = hb_elt.*;
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0 / (1.0 + std.math.exp(-val)));
        // elementwise multiply with w3(x)
        val *= hb2_elt;
        hb_elt.* = val;
    }
}

// W (d,n) @ x (n,) -> xout (d,)
// FIXME Tensor n2 must be const!
fn gemm(o: tensor.Tensor(f32, 1), x: tensor.Tensor(f32, 1), w: tensor.ConstTensor(f32, 2)) void {
    const o_size = o.layout.shape[0];
    const x_size = x.layout.shape[0];

    std.debug.assert(o_size == w.layout.shape[0]);
    std.debug.assert(x_size == w.layout.shape[1]);

    for (0..o_size) |i| {
        o.raw_data[i] = dot_const_b(x, w.sub_tensor(0, i));
    }
}

// FIXME Improve typing to reuse same Tensor class for const and non-const
fn dot_const_b(a: tensor.Tensor(f32, 1), b: tensor.ConstTensor(f32, 1)) f32 {
    // Make sure tensors are continuous
    std.debug.assert(a.layout.stride[0] == 1);
    std.debug.assert(b.layout.stride[0] == 1);
    // Make sure the shape is consistent
    std.debug.assert(a.layout.shape[0] == a.raw_data.len);
    std.debug.assert(b.layout.shape[0] == b.raw_data.len);

    var result: f32 = 0.0;

    for (a.raw_data, b.raw_data) |a_elt, b_elt| {
        result += a_elt * b_elt;
    }

    return result;
}

fn dot(a: tensor.Tensor(f32, 1), b: tensor.Tensor(f32, 1)) f32 {
    // Make sure tensors are continuous
    std.debug.assert(a.layout.stride[0] == 1);
    std.debug.assert(b.layout.stride[0] == 1);
    // Make sure the shape is consistent
    std.debug.assert(a.layout.shape[0] == a.raw_data.len);
    std.debug.assert(b.layout.shape[0] == b.raw_data.len);

    var result: f32 = 0.0;

    for (a.raw_data, b.raw_data) |a_elt, b_elt| {
        result += a_elt * b_elt;
    }

    return result;
}

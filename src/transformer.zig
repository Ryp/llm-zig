const llama = @import("llama.zig");

pub const Config = extern struct {
    dim: c_int,
    hidden_dim: c_int,
    n_layers: c_int,
    n_heads: c_int,
    n_kv_heads: c_int,
    vocab_size: c_int,
    seq_len: c_int,
};

pub const Transformer = extern struct {
    config: Config = @import("std").mem.zeroes(Config),
    weights: llama.TransformerWeights = @import("std").mem.zeroes(llama.TransformerWeights),
    state: llama.RunState = @import("std").mem.zeroes(llama.RunState),
    fd: c_int = @import("std").mem.zeroes(c_int),
    data: [*c]f32 = @import("std").mem.zeroes([*c]f32),
    file_size: isize = @import("std").mem.zeroes(isize),
};

pub fn build_transformer(arg_t: [*c]Transformer, arg_checkpoint_path: [*c]const u8) void {
    var t = arg_t;
    _ = &t;
    var checkpoint_path = arg_checkpoint_path;
    _ = &checkpoint_path;
    llama.read_checkpoint(checkpoint_path, &t.*.config, &t.*.weights, &t.*.fd, &t.*.data, &t.*.file_size);
    llama.malloc_run_state(&t.*.state, &t.*.config);
}

pub fn free_transformer(arg_t: [*c]Transformer) void {
    var t = arg_t;
    _ = &t;
    //if (t.*.data != @as([*c]f32, @ptrCast(@as(?*anyopaque, @ptrFromInt(@as(usize, 0) -% 1))))) {
        _ = llama.munmap(@as(?*anyopaque, @ptrCast(t.*.data)), @as(usize, @bitCast(t.*.file_size)));
    //}
    if (t.*.fd != -@as(c_int, 1)) {
        _ = llama.close(t.*.fd);
    }
    llama.free_run_state(&t.*.state);
}

pub fn rmsnorm(arg_o: [*c]f32, arg_x: [*c]f32, arg_weight: [*c]f32, arg_size: c_int) void {
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
    ss += 0.000009999999747378752;
    ss = 1.0 / llama.sqrtf(ss);
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

pub fn softmax(arg_x: [*c]f32, arg_size: c_int) void {
    var x = arg_x;
    _ = &x;
    var size = arg_size;
    _ = &size;
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
            }).* = llama.expf((blk: {
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

pub fn matmul(arg_xout: [*c]f32, arg_x: [*c]f32, arg_w: [*c]f32, arg_n: c_int, arg_d: c_int) void {
    var xout = arg_xout;
    _ = &xout;
    var x = arg_x;
    _ = &x;
    var w = arg_w;
    _ = &w;
    var n = arg_n;
    _ = &n;
    var d = arg_d;
    _ = &d;
    var i: c_int = undefined;
    _ = &i;
    {
        i = 0;
        while (i < d) : (i += 1) {
            var val: f32 = 0.0;
            _ = &val;
            {
                var j: c_int = 0;
                _ = &j;
                while (j < n) : (j += 1) {
                    val += (blk: {
                        const tmp = (i * n) + j;
                        if (tmp >= 0) break :blk w + @as(usize, @intCast(tmp)) else break :blk w - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                    }).* * (blk: {
                        const tmp = j;
                        if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                    }).*;
                }
            }
            (blk: {
                const tmp = i;
                if (tmp >= 0) break :blk xout + @as(usize, @intCast(tmp)) else break :blk xout - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* = val;
        }
    }
}

pub fn forward(arg_transformer: [*c]Transformer, arg_token: c_int, arg_pos: c_int) [*c]f32 {
    var transformer = arg_transformer;
    _ = &transformer;
    var token = arg_token;
    _ = &token;
    var pos = arg_pos;
    _ = &pos;
    var p: [*c]Config = &transformer.*.config;
    _ = &p;
    var w: [*c]llama.TransformerWeights = &transformer.*.weights;
    _ = &w;
    var s: [*c]llama.RunState = &transformer.*.state;
    _ = &s;
    var x: [*c]f32 = s.*.x;
    _ = &x;
    var dim: c_int = p.*.dim;
    _ = &dim;
    var kv_dim: c_int = @divTrunc(p.*.dim * p.*.n_kv_heads, p.*.n_heads);
    _ = &kv_dim;
    var kv_mul: c_int = @divTrunc(p.*.n_heads, p.*.n_kv_heads);
    _ = &kv_mul;
    var hidden_dim: c_int = p.*.hidden_dim;
    _ = &hidden_dim;
    var head_size: c_int = @divTrunc(dim, p.*.n_heads);
    _ = &head_size;
    var content_row: [*c]f32 = w.*.token_embedding_table + @as(usize, @bitCast(@as(isize, @intCast(token * dim))));
    _ = &content_row;
    _ = llama.memcpy(@as(?*anyopaque, @ptrCast(x)), @as(?*const anyopaque, @ptrCast(content_row)), @as(c_ulong, @bitCast(@as(c_long, dim))) *% @sizeOf(f32));
    {
        var l: c_ulonglong = 0;
        _ = &l;
        while (l < @as(c_ulonglong, @bitCast(@as(c_longlong, p.*.n_layers)))) : (l +%= 1) {
            rmsnorm(s.*.xb, x, w.*.rms_att_weight + (l *% @as(c_ulonglong, @bitCast(@as(c_longlong, dim)))), dim);
            var loff: c_int = @as(c_int, @bitCast(@as(c_uint, @truncate((l *% @as(c_ulonglong, @bitCast(@as(c_longlong, p.*.seq_len)))) *% @as(c_ulonglong, @bitCast(@as(c_longlong, kv_dim)))))));
            _ = &loff;
            s.*.k = (s.*.key_cache + @as(usize, @bitCast(@as(isize, @intCast(loff))))) + @as(usize, @bitCast(@as(isize, @intCast(pos * kv_dim))));
            s.*.v = (s.*.value_cache + @as(usize, @bitCast(@as(isize, @intCast(loff))))) + @as(usize, @bitCast(@as(isize, @intCast(pos * kv_dim))));
            matmul(s.*.q, s.*.xb, w.*.wq + ((l *% @as(c_ulonglong, @bitCast(@as(c_longlong, dim)))) *% @as(c_ulonglong, @bitCast(@as(c_longlong, dim)))), dim, dim);
            matmul(s.*.k, s.*.xb, w.*.wk + ((l *% @as(c_ulonglong, @bitCast(@as(c_longlong, dim)))) *% @as(c_ulonglong, @bitCast(@as(c_longlong, kv_dim)))), dim, kv_dim);
            matmul(s.*.v, s.*.xb, w.*.wv + ((l *% @as(c_ulonglong, @bitCast(@as(c_longlong, dim)))) *% @as(c_ulonglong, @bitCast(@as(c_longlong, kv_dim)))), dim, kv_dim);
            {
                var i: c_int = 0;
                _ = &i;
                while (i < dim) : (i += @as(c_int, 2)) {
                    var head_dim: c_int = @import("std").zig.c_translation.signedRemainder(i, head_size);
                    _ = &head_dim;
                    var freq: f32 = 1.0 / llama.powf(10000.0, @as(f32, @floatFromInt(head_dim)) / @as(f32, @floatFromInt(head_size)));
                    _ = &freq;
                    var val: f32 = @as(f32, @floatFromInt(pos)) * freq;
                    _ = &val;
                    var fcr: f32 = llama.cosf(val);
                    _ = &fcr;
                    var fci: f32 = llama.sinf(val);
                    _ = &fci;
                    var rotn: c_int = if (i < kv_dim) @as(c_int, 2) else @as(c_int, 1);
                    _ = &rotn;
                    {
                        var v: c_int = 0;
                        _ = &v;
                        while (v < rotn) : (v += 1) {
                            var vec: [*c]f32 = if (v == @as(c_int, 0)) s.*.q else s.*.k;
                            _ = &vec;
                            var v0: f32 = (blk: {
                                const tmp = i;
                                if (tmp >= 0) break :blk vec + @as(usize, @intCast(tmp)) else break :blk vec - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                            }).*;
                            _ = &v0;
                            var v1: f32 = (blk: {
                                const tmp = i + @as(c_int, 1);
                                if (tmp >= 0) break :blk vec + @as(usize, @intCast(tmp)) else break :blk vec - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                            }).*;
                            _ = &v1;
                            (blk: {
                                const tmp = i;
                                if (tmp >= 0) break :blk vec + @as(usize, @intCast(tmp)) else break :blk vec - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                            }).* = (v0 * fcr) - (v1 * fci);
                            (blk: {
                                const tmp = i + @as(c_int, 1);
                                if (tmp >= 0) break :blk vec + @as(usize, @intCast(tmp)) else break :blk vec - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                            }).* = (v0 * fci) + (v1 * fcr);
                        }
                    }
                }
            }
            var h: c_int = undefined;
            _ = &h;
            {
                h = 0;
                while (h < p.*.n_heads) : (h += 1) {
                    var q: [*c]f32 = s.*.q + @as(usize, @bitCast(@as(isize, @intCast(h * head_size))));
                    _ = &q;
                    var att: [*c]f32 = s.*.att + @as(usize, @bitCast(@as(isize, @intCast(h * p.*.seq_len))));
                    _ = &att;
                    {
                        var t: c_int = 0;
                        _ = &t;
                        while (t <= pos) : (t += 1) {
                            var k: [*c]f32 = ((s.*.key_cache + @as(usize, @bitCast(@as(isize, @intCast(loff))))) + @as(usize, @bitCast(@as(isize, @intCast(t * kv_dim))))) + @as(usize, @bitCast(@as(isize, @intCast(@divTrunc(h, kv_mul) * head_size))));
                            _ = &k;
                            var score: f32 = 0.0;
                            _ = &score;
                            {
                                var i: c_int = 0;
                                _ = &i;
                                while (i < head_size) : (i += 1) {
                                    score += (blk: {
                                        const tmp = i;
                                        if (tmp >= 0) break :blk q + @as(usize, @intCast(tmp)) else break :blk q - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                                    }).* * (blk: {
                                        const tmp = i;
                                        if (tmp >= 0) break :blk k + @as(usize, @intCast(tmp)) else break :blk k - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                                    }).*;
                                }
                            }
                            score /= llama.sqrtf(@as(f32, @floatFromInt(head_size)));
                            (blk: {
                                const tmp = t;
                                if (tmp >= 0) break :blk att + @as(usize, @intCast(tmp)) else break :blk att - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                            }).* = score;
                        }
                    }
                    softmax(att, pos + @as(c_int, 1));
                    var xb: [*c]f32 = s.*.xb + @as(usize, @bitCast(@as(isize, @intCast(h * head_size))));
                    _ = &xb;
                    _ = llama.memset(@as(?*anyopaque, @ptrCast(xb)), @as(c_int, 0), @as(c_ulong, @bitCast(@as(c_long, head_size))) *% @sizeOf(f32));
                    {
                        var t: c_int = 0;
                        _ = &t;
                        while (t <= pos) : (t += 1) {
                            var v: [*c]f32 = ((s.*.value_cache + @as(usize, @bitCast(@as(isize, @intCast(loff))))) + @as(usize, @bitCast(@as(isize, @intCast(t * kv_dim))))) + @as(usize, @bitCast(@as(isize, @intCast(@divTrunc(h, kv_mul) * head_size))));
                            _ = &v;
                            var a: f32 = (blk: {
                                const tmp = t;
                                if (tmp >= 0) break :blk att + @as(usize, @intCast(tmp)) else break :blk att - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                            }).*;
                            _ = &a;
                            {
                                var i: c_int = 0;
                                _ = &i;
                                while (i < head_size) : (i += 1) {
                                    (blk: {
                                        const tmp = i;
                                        if (tmp >= 0) break :blk xb + @as(usize, @intCast(tmp)) else break :blk xb - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                                    }).* += a * (blk: {
                                        const tmp = i;
                                        if (tmp >= 0) break :blk v + @as(usize, @intCast(tmp)) else break :blk v - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                                    }).*;
                                }
                            }
                        }
                    }
                }
            }
            matmul(s.*.xb2, s.*.xb, w.*.wo + ((l *% @as(c_ulonglong, @bitCast(@as(c_longlong, dim)))) *% @as(c_ulonglong, @bitCast(@as(c_longlong, dim)))), dim, dim);
            {
                var i: c_int = 0;
                _ = &i;
                while (i < dim) : (i += 1) {
                    (blk: {
                        const tmp = i;
                        if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                    }).* += (blk: {
                        const tmp = i;
                        if (tmp >= 0) break :blk s.*.xb2 + @as(usize, @intCast(tmp)) else break :blk s.*.xb2 - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                    }).*;
                }
            }
            rmsnorm(s.*.xb, x, w.*.rms_ffn_weight + (l *% @as(c_ulonglong, @bitCast(@as(c_longlong, dim)))), dim);
            matmul(s.*.hb, s.*.xb, w.*.w1 + ((l *% @as(c_ulonglong, @bitCast(@as(c_longlong, dim)))) *% @as(c_ulonglong, @bitCast(@as(c_longlong, hidden_dim)))), dim, hidden_dim);
            matmul(s.*.hb2, s.*.xb, w.*.w3 + ((l *% @as(c_ulonglong, @bitCast(@as(c_longlong, dim)))) *% @as(c_ulonglong, @bitCast(@as(c_longlong, hidden_dim)))), dim, hidden_dim);
            {
                var i: c_int = 0;
                _ = &i;
                while (i < hidden_dim) : (i += 1) {
                    var val: f32 = (blk: {
                        const tmp = i;
                        if (tmp >= 0) break :blk s.*.hb + @as(usize, @intCast(tmp)) else break :blk s.*.hb - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                    }).*;
                    _ = &val;
                    val *= 1.0 / (1.0 + llama.expf(-val));
                    val *= (blk: {
                        const tmp = i;
                        if (tmp >= 0) break :blk s.*.hb2 + @as(usize, @intCast(tmp)) else break :blk s.*.hb2 - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                    }).*;
                    (blk: {
                        const tmp = i;
                        if (tmp >= 0) break :blk s.*.hb + @as(usize, @intCast(tmp)) else break :blk s.*.hb - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                    }).* = val;
                }
            }
            matmul(s.*.xb, s.*.hb, w.*.w2 + ((l *% @as(c_ulonglong, @bitCast(@as(c_longlong, dim)))) *% @as(c_ulonglong, @bitCast(@as(c_longlong, hidden_dim)))), hidden_dim, dim);
            {
                var i: c_int = 0;
                _ = &i;
                while (i < dim) : (i += 1) {
                    (blk: {
                        const tmp = i;
                        if (tmp >= 0) break :blk x + @as(usize, @intCast(tmp)) else break :blk x - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                    }).* += (blk: {
                        const tmp = i;
                        if (tmp >= 0) break :blk s.*.xb + @as(usize, @intCast(tmp)) else break :blk s.*.xb - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                    }).*;
                }
            }
        }
    }
    rmsnorm(x, x, w.*.rms_final_weight, dim);
    matmul(s.*.logits, x, w.*.wcls, p.*.dim, p.*.vocab_size);
    return s.*.logits;
}

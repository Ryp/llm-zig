const llama = @import("llama.zig");

const transformer = @import("transformer.zig");

pub const ProbIndex = extern struct {
    prob: f32 = 0,
    index: usize = 0,
};

pub const Sampler = extern struct {
    vocab_size: usize = 0,
    probindex: [*c]ProbIndex = @import("std").mem.zeroes([*c]ProbIndex),
    temperature: f32 = @import("std").mem.zeroes(f32),
    topp: f32 = @import("std").mem.zeroes(f32),
    rng_state: c_ulonglong = @import("std").mem.zeroes(c_ulonglong),
};

fn sample_argmax(arg_probabilities: [*c]f32, n: usize) usize {
    var probabilities = arg_probabilities;
    _ = &probabilities;
    var max_i: usize = 0;
    _ = &max_i;
    var max_p: f32 = probabilities[@as(c_uint, @intCast(@as(c_int, 0)))];
    _ = &max_p;
    {
        var i: usize = 1;
        _ = &i;
        while (i < n) : (i += 1) {
            if ((blk: {
                const tmp = i;
                if (tmp >= 0) break :blk probabilities + @as(usize, @intCast(tmp)) else break :blk probabilities - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* > max_p) {
                max_i = i;
                max_p = (blk: {
                    const tmp = i;
                    if (tmp >= 0) break :blk probabilities + @as(usize, @intCast(tmp)) else break :blk probabilities - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*;
            }
        }
    }
    return max_i;
}

fn sample_mult(arg_probabilities: [*c]f32, arg_n: usize, arg_coin: f32) usize {
    var probabilities = arg_probabilities;
    _ = &probabilities;
    var n = arg_n;
    _ = &n;
    var coin = arg_coin;
    _ = &coin;
    var cdf: f32 = 0.0;
    _ = &cdf;

    {
        var i: usize = 0;
        _ = &i;
        while (i < n) : (i += 1) {
            cdf += (blk: {
                const tmp = i;
                if (tmp >= 0) break :blk probabilities + @as(usize, @intCast(tmp)) else break :blk probabilities - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).*;
            if (coin < cdf) {
                return i;
            }
        }
    }

    return n - 1;
}

export fn compare(arg_a: ?*const anyopaque, arg_b: ?*const anyopaque) c_int {
    var a = arg_a;
    _ = &a;
    var b = arg_b;
    _ = &b;
    var a_: [*c]ProbIndex = @as([*c]ProbIndex, @alignCast(@ptrCast(@volatileCast(@constCast(a)))));
    _ = &a_;
    var b_: [*c]ProbIndex = @as([*c]ProbIndex, @alignCast(@ptrCast(@volatileCast(@constCast(b)))));
    _ = &b_;
    if (a_.*.prob > b_.*.prob) return -@as(c_int, 1);
    if (a_.*.prob < b_.*.prob) return 1;
    return 0;
}

fn sample_topp(arg_probabilities: [*c]f32, arg_n: usize, arg_topp: f32, arg_probindex: [*c]ProbIndex, arg_coin: f32) usize {
    var probabilities = arg_probabilities;
    _ = &probabilities;
    var n = arg_n;
    _ = &n;
    var topp = arg_topp;
    _ = &topp;
    var probindex = arg_probindex;
    _ = &probindex;
    var coin = arg_coin;
    _ = &coin;
    var n0: c_int = 0;
    _ = &n0;
    const cutoff: f32 = (1.0 - topp) / @as(f32, @floatFromInt(n - @as(c_int, 1)));
    _ = &cutoff;
    {
        var i: usize = 0;
        _ = &i;
        while (i < n) : (i += 1) {
            if ((blk: {
                const tmp = i;
                if (tmp >= 0) break :blk probabilities + @as(usize, @intCast(tmp)) else break :blk probabilities - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* >= cutoff) {
                (blk: {
                    const tmp = n0;
                    if (tmp >= 0) break :blk probindex + @as(usize, @intCast(tmp)) else break :blk probindex - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*.index = i;
                (blk: {
                    const tmp = n0;
                    if (tmp >= 0) break :blk probindex + @as(usize, @intCast(tmp)) else break :blk probindex - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*.prob = (blk: {
                    const tmp = i;
                    if (tmp >= 0) break :blk probabilities + @as(usize, @intCast(tmp)) else break :blk probabilities - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*;
                n0 += 1;
            }
        }
    }

    llama.qsort(@as(?*anyopaque, @ptrCast(probindex)), @as(usize, @bitCast(@as(c_long, n0))), @sizeOf(ProbIndex), &compare);

    var cumulative_prob: f32 = 0.0;
    _ = &cumulative_prob;
    var last_idx: c_int = n0 - @as(c_int, 1);
    _ = &last_idx;
    {
        var i: c_int = 0;
        _ = &i;
        while (i < n0) : (i += 1) {
            cumulative_prob += (blk: {
                const tmp = i;
                if (tmp >= 0) break :blk probindex + @as(usize, @intCast(tmp)) else break :blk probindex - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).*.prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break;
            }
        }
    }
    var r: f32 = coin * cumulative_prob;
    _ = &r;
    var cdf: f32 = 0.0;
    _ = &cdf;
    {
        var i: c_int = 0;
        _ = &i;
        while (i <= last_idx) : (i += 1) {
            cdf += (blk: {
                const tmp = i;
                if (tmp >= 0) break :blk probindex + @as(usize, @intCast(tmp)) else break :blk probindex - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).*.prob;
            if (r < cdf) {
                return (blk: {
                    const tmp = i;
                    if (tmp >= 0) break :blk probindex + @as(usize, @intCast(tmp)) else break :blk probindex - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*.index;
            }
        }
    }
    return (blk: {
        const tmp = last_idx;
        if (tmp >= 0) break :blk probindex + @as(usize, @intCast(tmp)) else break :blk probindex - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
    }).*.index;
}
pub fn build_sampler(sampler: *Sampler, vocab_size: usize, temperature: f32, topp: f32, rng_seed: c_ulonglong) void {
    sampler.vocab_size = vocab_size;
    sampler.temperature = temperature;
    sampler.topp = topp;
    sampler.rng_state = rng_seed;
    sampler.probindex = @as([*c]ProbIndex, @ptrCast(@alignCast(llama.malloc(sampler.vocab_size * @sizeOf(ProbIndex)))));
}

pub fn free_sampler(sampler: *Sampler) void {
    llama.free(@as(?*anyopaque, @ptrCast(sampler.probindex)));
}

fn random_u32(state: [*c]c_ulonglong) c_uint {
    state.* ^= state.* >> @intCast(12);
    state.* ^= state.* << @intCast(25);
    state.* ^= state.* >> @intCast(27);
    return @as(c_uint, @bitCast(@as(c_uint, @truncate((state.* *% @as(c_ulonglong, 2685821657736338717)) >> @intCast(32)))));
}

fn random_f32(state: [*c]c_ulonglong) f32 {
    return @as(f32, @floatFromInt(random_u32(state) >> @intCast(8))) / 16777216.0;
}

pub fn sample(sampler: [*c]Sampler, logits: [*c]f32) u16 {
    var next: usize = undefined;
    _ = &next;
    if (sampler.*.temperature == 0.0) {
        next = sample_argmax(logits, sampler.*.vocab_size);
    } else {
        {
            var q: c_int = 0;
            _ = &q;
            while (q < sampler.*.vocab_size) : (q += 1) {
                (blk: {
                    const tmp = q;
                    if (tmp >= 0) break :blk logits + @as(usize, @intCast(tmp)) else break :blk logits - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).* /= sampler.*.temperature;
            }
        }
        transformer.softmax(logits, sampler.*.vocab_size);
        var coin: f32 = random_f32(&sampler.*.rng_state);
        _ = &coin;
        if ((sampler.*.topp <= @as(f32, @floatFromInt(@as(c_int, 0)))) or (sampler.*.topp >= @as(f32, @floatFromInt(@as(c_int, 1))))) {
            next = sample_mult(logits, sampler.*.vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler.*.vocab_size, sampler.*.topp, sampler.*.probindex, coin);
        }
    }
    return @intCast(next);
}


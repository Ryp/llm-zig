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

fn sample_argmax(probabilities: []f32, n: usize) usize {
    var max_i: usize = 0;
    var max_p = probabilities[0];

    for (1..n) |i| {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }

    return max_i;
}

fn sample_mult(probabilities: []f32, n: usize, coin: f32) usize {
    var cdf: f32 = 0.0;

    for (0..n) |i| {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }

    return n - 1; // in case of rounding errors
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

fn sample_topp(probabilities: []f32, n: usize, topp: f32, probindex: [*c]ProbIndex, coin: f32) usize {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    var n0: usize = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const cutoff = (1.0 - topp) / @as(f32, @floatFromInt(n - 1));
    for (0..n) |i| {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0 += 1;
        }
    }

    llama.qsort(@as(?*anyopaque, @ptrCast(probindex)), n0, @sizeOf(ProbIndex), &compare);

    // truncate the list where cumulative probability exceeds topp
    var cumulative_prob: f32 = 0.0;
    var last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (0..n0) |i| {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    const r = coin * cumulative_prob;
    var cdf: f32 = 0.0;
    for (0..last_idx + 1) |i| {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }

    return probindex[last_idx].index; // in case of rounding errors
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

pub fn sample(sampler: [*c]Sampler, logits: []f32) u16 {
    var next: usize = undefined;
    _ = &next;
    if (sampler.*.temperature == 0.0) {
        next = sample_argmax(logits, sampler.*.vocab_size);
    } else {
        for (0..sampler.*.vocab_size) |q| {
            logits[q] /= sampler.*.temperature;
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

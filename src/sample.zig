const std = @import("std");

const transformer = @import("transformer.zig");
const tokenizer = @import("tokenizer.zig");

const ProbIndex = struct {
    prob: f32,
    id: tokenizer.TokenId,
};

pub const Sampler = struct {
    vocab_size: usize,
    probindex: []ProbIndex,
    temperature: f32,
    topp: f32,
    rng_state: c_ulonglong,
};

fn sample_argmax(probabilities: []f32, n: usize) tokenizer.TokenId {
    var max_i: usize = 0;
    var max_p = probabilities[0];

    for (1..n) |i| {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }

    return @enumFromInt(max_i);
}

fn sample_mult(probabilities: []f32, n: usize, coin: f32) tokenizer.TokenId {
    var cdf: f32 = 0.0;

    for (0..n) |i| {
        cdf += probabilities[i];
        if (coin < cdf) {
            return @enumFromInt(i);
        }
    }

    return @enumFromInt(n - 1); // in case of rounding errors
}

fn prob_index_less_than(context: void, lhs: ProbIndex, rhs: ProbIndex) bool {
    _ = context;

    // Sort in decreasing order
    return lhs.prob > rhs.prob;
}

fn sample_topp(probabilities: []f32, n: usize, topp: f32, probindex: []ProbIndex, coin: f32) tokenizer.TokenId {
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
            probindex[n0].id = @enumFromInt(i);
            probindex[n0].prob = probabilities[i];
            n0 += 1;
        }
    }

    std.sort.pdq(ProbIndex, probindex[0..n0], {}, prob_index_less_than);

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
            return probindex[i].id;
        }
    }

    return probindex[last_idx].id; // in case of rounding errors
}

pub fn create_sampler(allocator: *std.mem.Allocator, vocab_size: usize, temperature: f32, topp: f32, rng_seed: c_ulonglong) !Sampler {
    return Sampler{
        .vocab_size = vocab_size,
        .temperature = temperature,
        .topp = topp,
        .rng_state = rng_seed,
        .probindex = try allocator.alloc(ProbIndex, vocab_size),
    };
}

pub fn destroy_sampler(allocator: *std.mem.Allocator, sampler: *Sampler) void {
    allocator.free(sampler.probindex);
}

fn random_u32(state: *u64) c_uint {
    state.* ^= state.* >> 12;
    state.* ^= state.* << 25;
    state.* ^= state.* >> 27;

    return @as(c_uint, @truncate((state.* *% 2685821657736338717) >> 32));
}

fn random_f32(state: *u64) f32 {
    return @as(f32, @floatFromInt(random_u32(state) >> 8)) / 16777216.0;
}

pub fn sample(sampler: *Sampler, logits: []f32) tokenizer.TokenId {
    var next: tokenizer.TokenId = undefined;

    if (sampler.temperature == 0.0) {
        next = sample_argmax(logits, sampler.vocab_size);
    } else {
        for (0..sampler.vocab_size) |q| {
            logits[q] /= sampler.temperature;
        }

        transformer.softmax(logits[0..sampler.vocab_size]);

        var coin: f32 = random_f32(&sampler.rng_state);
        _ = &coin;
        if ((sampler.topp <= @as(f32, @floatFromInt(@as(c_int, 0)))) or (sampler.topp >= @as(f32, @floatFromInt(@as(c_int, 1))))) {
            next = sample_mult(logits, sampler.vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler.vocab_size, sampler.topp, sampler.probindex, coin);
        }
    }
    return next;
}

const std = @import("std");

const llama = @import("llama.zig");

const token = @import("tokenizer.zig");
const sample = @import("sample.zig");
const transformer = @import("transformer.zig");

pub fn error_usage() void {
    std.debug.print("Usage:   run <checkpoint> [options]\n", .{});
    std.debug.print("Example: run model.bin -n 256 -i \"Once upon a time\"\n", .{});
    std.debug.print("Options:\n", .{});
    std.debug.print("  -t <float>  temperature in [0,inf], default 1.0\n", .{});
    std.debug.print("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n", .{});
    std.debug.print("  -s <int>    random seed, default time(NULL)\n", .{});
    std.debug.print("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n", .{});
    std.debug.print("  -i <string> input prompt\n", .{});
    std.debug.print("  -z <string> optional path to custom tokenizer\n", .{});
    std.debug.print("  -m <string> mode: generate|chat, default: generate\n", .{});
    std.debug.print("  -y <string> (optional) system prompt in chat mode\n", .{});
}

pub fn main() !void {
    const checkpoint_path = "../stories15M.bin";

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const check = gpa.deinit();
        std.debug.assert(check == .ok);
    }

    var allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    const tokenizer_path: [*c]u8 = @as([*c]u8, @ptrCast(@volatileCast(@constCast("tokenizer.bin"))));
    const temperature: f32 = 1.0;
    const topp: f32 = 0.9;
    var steps: usize = 16;

    const prompt: [*c]const u8 = "What does Claire like?"; // FIXME
    const rng_seed: c_ulonglong = 0; // FIXME

    // FIXME Args parsing goes here

    if (@as(f64, @floatCast(temperature)) < 0.0) {
        temperature = @as(f32, @floatCast(0.0));
    }

    if ((@as(f64, @floatCast(topp)) < 0.0) or (1.0 < @as(f64, @floatCast(topp)))) {
        topp = @as(f32, @floatCast(0.9));
    }

    var llama2_transformer: transformer.Transformer = undefined;

    try transformer.build_transformer(&allocator, &llama2_transformer, checkpoint_path);
    defer transformer.free_transformer(&allocator, &llama2_transformer);

    if (steps == 0 or steps > llama2_transformer.config.seq_len) {
        steps = llama2_transformer.config.seq_len;
    }

    var tokenizer: token.Tokenizer = undefined;
    token.build_tokenizer(&tokenizer, tokenizer_path, llama2_transformer.config.vocab_size);
    defer token.free_tokenizer(&tokenizer);

    var sampler: sample.Sampler = undefined;
    sample.build_sampler(&sampler, llama2_transformer.config.vocab_size, temperature, topp, rng_seed);
    defer sample.free_sampler(&sampler);

    llama.generate(&llama2_transformer, &tokenizer, &sampler, prompt, steps);
}

const std = @import("std");
const llama = @import("llama.zig");
const token = @import("tokenizer.zig");
const sample = @import("sample.zig");

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

pub fn main() void {
    const checkpoint_path: [*c]const u8 = "../stories15M.bin";

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    var tokenizer_path: [*c]u8 = @as([*c]u8, @ptrCast(@volatileCast(@constCast("tokenizer.bin"))));
    _ = &tokenizer_path;
    var temperature: f32 = 0.0;
    _ = &temperature;

    var topp: f32 = 0.8999999761581421;
    _ = &topp;

    var steps: c_int = 256;
    _ = &steps;

    const prompt: [*c]const u8 = "What does Claire like?"; // FIXME
    const rng_seed: c_ulonglong = 0; // FIXME
    //
    var mode: [*c]u8 = @as([*c]u8, @ptrCast(@volatileCast(@constCast("generate"))));
    _ = &mode;
    var system_prompt: [*c]u8 = null;
    _ = &system_prompt;

    // FIXME Args parsing goes here
    if (@as(f64, @floatCast(temperature)) < 0.0) {
        temperature = @as(f32, @floatCast(0.0));
    }
    if ((@as(f64, @floatCast(topp)) < 0.0) or (1.0 < @as(f64, @floatCast(topp)))) {
        topp = @as(f32, @floatCast(0.9));
    }
    if (steps < @as(c_int, 0)) {
        steps = 0;
    }
    var transformer: llama.Transformer = undefined;
    llama.build_transformer(&transformer, checkpoint_path);
    if ((steps == @as(c_int, 0)) or (steps > transformer.config.seq_len)) {
        steps = transformer.config.seq_len;
    }

    var tokenizer: token.Tokenizer = undefined;
    token.build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    var sampler: sample.Sampler = undefined;
    sample.build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (llama.strcmp(mode, "generate") == @as(c_int, 0)) {
        llama.generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (llama.strcmp(mode, "chat") == @as(c_int, 0)) {
        llama.chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        _ = llama.fprintf(llama.stderr, "unknown mode: %s\n", mode);
        error_usage();
        return;
    }

    sample.free_sampler(&sampler);
    token.free_tokenizer(&tokenizer);
    llama.free_transformer(&transformer);
}

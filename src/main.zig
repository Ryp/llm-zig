const std = @import("std");

const llama = @import("llama.zig");

const tokenizer = @import("tokenizer.zig");
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

    const tokenizer_path = "tokenizer.bin";
    const temperature: f32 = 1.0;
    const topp: f32 = 0.9;
    var steps: usize = 160;

    const prompt = "What does Claire like?"; // FIXME
    const rng_seed: c_ulonglong = 0; // FIXME

    // FIXME Args parsing goes here

    if (@as(f64, @floatCast(temperature)) < 0.0) {
        temperature = @as(f32, @floatCast(0.0));
    }

    if ((@as(f64, @floatCast(topp)) < 0.0) or (1.0 < @as(f64, @floatCast(topp)))) {
        topp = @as(f32, @floatCast(0.9));
    }

    var llama2_transformer: transformer.Transformer = undefined;

    try transformer.create_transformer(&allocator, &llama2_transformer, checkpoint_path);
    defer transformer.destroy_transformer(&allocator, &llama2_transformer);

    if (steps == 0 or steps > llama2_transformer.config.seq_len) {
        steps = llama2_transformer.config.seq_len;
    }

    var tok: tokenizer.Tokenizer = undefined;
    tokenizer.build_tokenizer(&tok, tokenizer_path, llama2_transformer.config.vocab_size);
    defer tokenizer.free_tokenizer(&tok);

    var sampler = try sample.create_sampler(&allocator, llama2_transformer.config.vocab_size, temperature, topp, rng_seed);
    defer sample.free_sampler(&allocator, &sampler);

    try generate(&allocator, &llama2_transformer, &tok, &sampler, prompt, steps);
}

fn generate(allocator: *std.mem.Allocator, arg_transformer: *transformer.Transformer, arg_tokenizer: *tokenizer.Tokenizer, sampler: *sample.Sampler, prompt: [:0]const u8, steps: usize) !void {
    const prompt_len = llama.strlen(prompt);
    const prompt_tokens = try allocator.alloc(c_int, prompt_len + 3); // +3 for '\0', ?BOS, ?EOS
    defer allocator.free(prompt_tokens);

    var num_prompt_tokens: usize = 0;
    tokenizer.encode(arg_tokenizer, prompt, 1, 0, prompt_tokens.ptr, &num_prompt_tokens); // FIXME

    if (num_prompt_tokens < 1) {
        std.debug.print("Something is wrong, expected at least 1 prompt token\n", .{});
        return error.Usage;
    }

    var start: c_long = 0;
    var next: u16 = undefined;
    var token: u16 = @intCast(prompt_tokens[0]);
    var pos: usize = 0;

    while (pos < steps) {
        const logits = transformer.forward(arg_transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = @intCast(prompt_tokens[pos + 1]);
        } else {
            next = sample.sample(sampler, logits[0..arg_transformer.config.vocab_size]);
        }

        pos += 1;

        if (next == 1) {
            break;
        }

        const piece: [*c]u8 = tokenizer.decode(arg_tokenizer, token, next);
        tokenizer.safe_printf(piece);

        _ = llama.fflush(llama.stdout);

        token = next;

        if (start == @as(c_long, @bitCast(@as(c_long, @as(c_int, 0))))) {
            start = llama.time_in_ms();
        }
    }

    _ = llama.printf("\n");

    if (pos > 1) {
        const end: c_long = llama.time_in_ms();
        std.debug.print("achieved tok/s: {}\n", .{(@as(f64, @floatFromInt(pos - @as(c_int, 1))) / @as(f64, @floatFromInt(end - start))) * @as(f64, @floatFromInt(@as(c_int, 1000)))});
    }

    llama.free(@as(?*anyopaque, @ptrCast(prompt_tokens)));
}

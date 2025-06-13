const std = @import("std");

const tokenizer = @import("tokenizer.zig");
const sample = @import("sample.zig");
const transformer = @import("transformer.zig");

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
    var steps: usize = 256;

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

    var tok = try tokenizer.create_tokenizer(&allocator, tokenizer_path, llama2_transformer.config.vocab_size);
    defer tokenizer.destroy_tokenizer(&allocator, &tok);

    var sampler = try sample.create_sampler(&allocator, llama2_transformer.config.vocab_size, temperature, topp, rng_seed);
    defer sample.destroy_sampler(&allocator, &sampler);

    try generate(&allocator, &llama2_transformer, &tok, &sampler, prompt, steps);
}

fn generate(allocator: *std.mem.Allocator, arg_transformer: *transformer.Transformer, arg_tokenizer: *tokenizer.Tokenizer, sampler: *sample.Sampler, prompt: [:0]const u8, steps: usize) !void {
    const prompt_tokens = try allocator.alloc(tokenizer.TokenId, prompt.len + 3); // +3 for '\0', ?BOS, ?EOS
    defer allocator.free(prompt_tokens);

    const num_prompt_tokens = try tokenizer.encode(allocator, arg_tokenizer, prompt, true, false, prompt_tokens);

    if (num_prompt_tokens < 1) {
        std.debug.print("Something is wrong, expected at least 1 prompt token\n", .{});
        return error.Usage;
    }

    var start: i64 = 0;
    var next: tokenizer.TokenId = undefined;
    var token: tokenizer.TokenId = prompt_tokens[0];
    var pos: usize = 0;

    while (pos < steps) {
        const logits = transformer.forward(arg_transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample.sample(sampler, logits[0..arg_transformer.config.vocab_size]);
        }

        pos += 1;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == .BOS) {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        const piece = tokenizer.decode(arg_tokenizer, token, next);
        tokenizer.safe_printf(piece);

        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) {
            start = std.time.milliTimestamp();
        }
    }

    std.debug.print("\n", .{});

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        const end = std.time.milliTimestamp();
        std.debug.print("achieved tok/s: {d:.1}\n", .{(@as(f64, @floatFromInt(pos - @as(c_int, 1))) / @as(f64, @floatFromInt(end - start))) * @as(f64, @floatFromInt(@as(c_int, 1000)))});
    }
}

const std = @import("std");

pub const Tokenizer = struct {
    vocab: [][:0]const u8,
    vocab_scores: []f32,
    sorted_vocab: []TokenIndex,
    vocab_size: usize,
    max_token_length: usize,
    byte_pieces: [256][1:0]u8,
};

pub const TokenId = enum(u16) {
    Unknown = 0,
    BOS = 1,
    EOS = 2,
    _,
};

const TokenIndex = struct {
    str: [:0]const u8,
    id: TokenId,
};

pub fn create_tokenizer(allocator: *std.mem.Allocator, tokenizer_path: [:0]const u8, vocab_size: usize) !Tokenizer {
    var t: Tokenizer = undefined;

    t.vocab_size = vocab_size;

    t.vocab = try allocator.alloc([:0]const u8, vocab_size);
    errdefer allocator.free(t.vocab);

    t.vocab_scores = try allocator.alloc(f32, vocab_size);
    errdefer allocator.free(t.vocab_scores);

    for (&t.byte_pieces, 0..) |*byte_piece, i| {
        byte_piece[0] = @intCast(i);
    }

    const tokenizer_file = if (std.fs.cwd().openFile(tokenizer_path, .{ .mode = .read_only })) |f| f else |err| {
        std.debug.print("error: couldn't open file: '{s}'\n", .{tokenizer_path});
        return err;
    };
    defer tokenizer_file.close();

    var max_token_length: i32 = undefined;

    {
        const max_token_length_bytes = std.mem.asBytes(&max_token_length);
        const bytes_read = try tokenizer_file.read(max_token_length_bytes);
        std.debug.assert(bytes_read == max_token_length_bytes.len);
    }

    t.max_token_length = @intCast(max_token_length);

    for (0..vocab_size) |i| {
        {
            const score_bytes = std.mem.asBytes(&t.vocab_scores[i]);
            const bytes_read = try tokenizer_file.read(score_bytes);
            std.debug.assert(bytes_read == score_bytes.len);
        }

        var len_i32: i32 = undefined;

        {
            const len_bytes = std.mem.asBytes(&len_i32);
            const bytes_read = try tokenizer_file.read(len_bytes);
            std.debug.assert(bytes_read == len_bytes.len);
        }

        const len: usize = @intCast(len_i32);
        const vocab_i = try allocator.allocSentinel(u8, len, 0);
        errdefer allocator.free(vocab_i);

        const bytes_read = try tokenizer_file.read(vocab_i);
        std.debug.assert(bytes_read == len);

        t.vocab[i] = vocab_i;
    }

    t.sorted_vocab = try allocator.alloc(TokenIndex, vocab_size);
    errdefer allocator.free(t.sorted_vocab);

    for (0..t.vocab_size) |i| {
        t.sorted_vocab[i] = .{
            .str = t.vocab[i],
            .id = @enumFromInt(i),
        };
    }

    std.mem.sort(TokenIndex, t.sorted_vocab, {}, compare_tokens_less_than);

    return t;
}

fn compare_tokens_less_than(_: void, a: TokenIndex, b: TokenIndex) bool {
    return std.mem.orderZ(u8, a.str, b.str) == .lt;
}

pub fn destroy_tokenizer(allocator: *std.mem.Allocator, t: *Tokenizer) void {
    {
        var i: usize = 0;
        _ = &i;
        while (i < t.*.vocab_size) : (i += 1) {
            allocator.free(t.vocab[i]);
        }
    }

    allocator.free(t.vocab);
    allocator.free(t.vocab_scores);
    allocator.free(t.sorted_vocab);
}

pub fn decode(t: *Tokenizer, prev_token: TokenId, token: TokenId) [:0]const u8 {
    var piece = t.vocab[@intFromEnum(token)];

    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == .BOS and piece[0] == ' ') {
        piece = piece[1..];
    }

    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    const length = std.mem.sliceTo(piece, 0).len;

    if (length == 6) {
        if (piece[0] == '<' and piece[5] == '>') {
            if (std.fmt.parseUnsigned(u8, piece[1..5], 0) catch null) |n| {
                piece = &t.byte_pieces[n];
            }
        }
    }

    return piece;
}

pub fn safe_printf(piece: [:0]const u8) void {
    if (piece[0] == '\x00') {
        return;
    }

    if (piece[1] == '\x00') {
        const byte_val = piece[0];

        if (!(std.ascii.isPrint(byte_val) or std.ascii.isWhitespace(byte_val))) {
            return; // bad byte, don't print it
        }
    }

    std.debug.print("{s}", .{piece});
}

fn token_index_cmp(context: [:0]const u8, elt: TokenIndex) std.math.Order {
    return std.mem.orderZ(u8, context, elt.str);
}

fn str_lookup(str: [:0]const u8, sorted_vocab: []const TokenIndex) ?TokenId {
    // FIXME cast to null terminated string
    const search_index = std.sort.binarySearch(TokenIndex, sorted_vocab, str, token_index_cmp);

    if (search_index) |index| {
        return sorted_vocab[index].id;
    } else {
        return null;
    }
}

pub fn encode(allocator: *std.mem.Allocator, t: *Tokenizer, text: [:0]const u8, put_bos: bool, put_eos: bool, tokens: []TokenId) !usize {
    const str_buffer = try allocator.allocSentinel(u8, t.max_token_length * 2 + 1 + 2 *% @sizeOf(u8), 0);
    defer allocator.free(str_buffer);

    var str_len: usize = 0;
    var n_tokens: usize = 0;

    if (put_bos) {
        tokens[n_tokens] = .BOS;
        n_tokens += 1;
    }

    if (text[0] != '\x00') {
        if (str_lookup(" ", t.sorted_vocab)) |id| {
            tokens[n_tokens] = id;
        } else {
            @panic("Hell no");
        }
        n_tokens += 1;
    }

    // process the raw (UTF-8) byte sequence of the input string
    {
        var c = text;
        while (c[0] != '\x00') : (c = c[1..]) {
            // reset buffer if the current byte is ASCII or a leading byte
            // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
            // 0x80 is 10000000
            // in UTF-8, all continuation bytes start with "10" in first two bits
            // so in English this is: "if this byte is not a continuation byte"
            if ((c[0] & 0xC0) != 0x80) {
                // this byte must be either a leading byte (11...) or an ASCII char (0x...)
                // => reset our location, as we're starting a new UTF-8 codepoint
                str_len = 0;
            }

            // append the current byte to the buffer
            str_buffer[str_len] = c[0];
            str_len += 1;
            str_buffer[str_len] = '\x00';

            // while the next character is a continuation byte, continue appending
            // but if there are too many of them, just stop to avoid overruning str_buffer size.
            if ((c[1] & 0xC0) == 0x80 and str_len < 4) {
                continue;
            }

            // ok c+1 is not a continuation byte, so we've read in a full codepoint
            if (str_lookup(str_buffer, t.sorted_vocab)) |id| {
                // we found this codepoint in vocab, add it as a token
                tokens[n_tokens] = id;
                n_tokens += 1;
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3
                for (0..str_len) |i| {
                    tokens[n_tokens] = @enumFromInt(str_buffer[i] + 3);
                    n_tokens += 1;
                }
            }

            str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
        }
    }

    while (true) {
        var best_score: f32 = -10000000000.0;
        var best_id: TokenId = undefined;
        var best_idx_opt: ?usize = null;

        for (0..n_tokens - 1) |i| {
            _ = try std.fmt.bufPrintZ(str_buffer, "{s}{s}", .{ t.vocab[@intFromEnum(tokens[i])], t.vocab[@intFromEnum(tokens[i + 1])] });

            if (str_lookup(str_buffer, t.sorted_vocab)) |id| {
                if (t.vocab_scores[@intFromEnum(id)] > best_score) {
                    best_score = t.vocab_scores[@intFromEnum(id)];
                    best_id = id;
                    best_idx_opt = i;
                }
            }
        }

        if (best_idx_opt) |best_idx| {
            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id;

            for (best_idx + 1..n_tokens - 1) |i| {
                tokens[i] = tokens[i + 1];
            }

            n_tokens -= 1; // token length decreased
        } else {
            break; // we couldn't find any more pairs to merge, so we're done
        }
    }

    // add optional EOS (=2) token, if desired
    if (put_eos) {
        tokens[n_tokens] = .EOS;
        n_tokens += 1;
    }

    return n_tokens;
}

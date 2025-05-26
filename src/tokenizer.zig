const llama = @import("llama.zig");

pub const TokenIndex = extern struct {
    str: [*c]u8,
    id: c_int,
};

pub const Tokenizer = extern struct {
    vocab: [*c][*c]u8 = @import("std").mem.zeroes([*c][*c]u8),
    vocab_scores: [*c]f32 = @import("std").mem.zeroes([*c]f32),
    sorted_vocab: [*c]TokenIndex,
    vocab_size: c_int = @import("std").mem.zeroes(c_int),
    max_token_length: c_uint = @import("std").mem.zeroes(c_uint),
    byte_pieces: [512]u8 = @import("std").mem.zeroes([512]u8),
};

export fn compare_tokens(arg_a: ?*const anyopaque, arg_b: ?*const anyopaque) c_int {
    const a: *const TokenIndex = @alignCast(@ptrCast(arg_a));
    const b: *const TokenIndex = @alignCast(@ptrCast(arg_b));
    return llama.strcmp(a.str, b.str);
}

pub fn build_tokenizer(arg_t: [*c]Tokenizer, arg_tokenizer_path: [*c]u8, arg_vocab_size: usize) void {
    var t = arg_t;
    _ = &t;
    var tokenizer_path = arg_tokenizer_path;
    _ = &tokenizer_path;
    var vocab_size: c_int = @intCast(arg_vocab_size);
    _ = &vocab_size;
    t.*.vocab_size = vocab_size;
    t.*.vocab = @as([*c][*c]u8, @ptrCast(@alignCast(llama.malloc(@as(c_ulong, @bitCast(@as(c_long, vocab_size))) *% @sizeOf([*c]u8)))));
    t.*.vocab_scores = @as([*c]f32, @ptrCast(@alignCast(llama.malloc(@as(c_ulong, @bitCast(@as(c_long, vocab_size))) *% @sizeOf(f32)))));
    t.*.sorted_vocab = null;
    {
        var i: c_int = 0;
        _ = &i;
        while (i < @as(c_int, 256)) : (i += 1) {
            t.*.byte_pieces[@as(c_uint, @intCast(i * @as(c_int, 2)))] = @as(u8, @bitCast(@as(i8, @truncate(i))));
            t.*.byte_pieces[@as(c_uint, @intCast((i * @as(c_int, 2)) + @as(c_int, 1)))] = '\x00';
        }
    }
    var file: ?*llama.FILE = llama.fopen(tokenizer_path, "rb");
    _ = &file;
    if (!(file != null)) {
        _ = llama.fprintf(llama.stderr, "couldn't load %s\n", tokenizer_path);
        llama.exit(@as(c_int, 1));
    }
    if (llama.fread(@as(?*anyopaque, @ptrCast(&t.*.max_token_length)), @sizeOf(c_int), @as(c_ulong, @bitCast(@as(c_long, @as(c_int, 1)))), file) != @as(c_ulong, @bitCast(@as(c_long, @as(c_int, 1))))) {
        _ = llama.fprintf(llama.stderr, "failed read\n");
        llama.exit(@as(c_int, 1));
    }
    var len: c_int = undefined;
    _ = &len;
    {
        var i: c_int = 0;
        _ = &i;
        while (i < vocab_size) : (i += 1) {
            if (llama.fread(@as(?*anyopaque, @ptrCast(t.*.vocab_scores + @as(usize, @bitCast(@as(isize, @intCast(i)))))), @sizeOf(f32), @as(c_ulong, @bitCast(@as(c_long, @as(c_int, 1)))), file) != @as(c_ulong, @bitCast(@as(c_long, @as(c_int, 1))))) {
                _ = llama.fprintf(llama.stderr, "failed read\n");
                llama.exit(@as(c_int, 1));
            }
            if (llama.fread(@as(?*anyopaque, @ptrCast(&len)), @sizeOf(c_int), @as(c_ulong, @bitCast(@as(c_long, @as(c_int, 1)))), file) != @as(c_ulong, @bitCast(@as(c_long, @as(c_int, 1))))) {
                _ = llama.fprintf(llama.stderr, "failed read\n");
                llama.exit(@as(c_int, 1));
            }
            (blk: {
                const tmp = i;
                if (tmp >= 0) break :blk t.*.vocab + @as(usize, @intCast(tmp)) else break :blk t.*.vocab - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* = @as([*c]u8, @ptrCast(@alignCast(llama.malloc(@as(c_ulong, @bitCast(@as(c_long, len + @as(c_int, 1))))))));
            if (llama.fread(@as(?*anyopaque, @ptrCast((blk: {
                const tmp = i;
                if (tmp >= 0) break :blk t.*.vocab + @as(usize, @intCast(tmp)) else break :blk t.*.vocab - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).*)), @as(c_ulong, @bitCast(@as(c_long, len))), @as(c_ulong, @bitCast(@as(c_long, @as(c_int, 1)))), file) != @as(c_ulong, @bitCast(@as(c_long, @as(c_int, 1))))) {
                _ = llama.fprintf(llama.stderr, "failed read\n");
                llama.exit(@as(c_int, 1));
            }
            (blk: {
                const tmp = len;
                if (tmp >= 0) break :blk (blk_1: {
                    const tmp_2 = i;
                    if (tmp_2 >= 0) break :blk_1 t.*.vocab + @as(usize, @intCast(tmp_2)) else break :blk_1 t.*.vocab - ~@as(usize, @bitCast(@as(isize, @intCast(tmp_2)) +% -1));
                }).* + @as(usize, @intCast(tmp)) else break :blk (blk_1: {
                    const tmp_2 = i;
                    if (tmp_2 >= 0) break :blk_1 t.*.vocab + @as(usize, @intCast(tmp_2)) else break :blk_1 t.*.vocab - ~@as(usize, @bitCast(@as(isize, @intCast(tmp_2)) +% -1));
                }).* - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).* = '\x00';
        }
    }
    _ = llama.fclose(file);
}
pub fn free_tokenizer(arg_t: [*c]Tokenizer) void {
    var t = arg_t;
    _ = &t;
    {
        var i: c_int = 0;
        _ = &i;
        while (i < t.*.vocab_size) : (i += 1) {
            llama.free(@as(?*anyopaque, @ptrCast((blk: {
                const tmp = i;
                if (tmp >= 0) break :blk t.*.vocab + @as(usize, @intCast(tmp)) else break :blk t.*.vocab - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
            }).*)));
        }
    }
    llama.free(@as(?*anyopaque, @ptrCast(t.*.vocab)));
    llama.free(@as(?*anyopaque, @ptrCast(t.*.vocab_scores)));
    llama.free(@as(?*anyopaque, @ptrCast(t.*.sorted_vocab)));
}
pub fn decode(arg_t: [*c]Tokenizer, arg_prev_token: c_int, token: c_int) [*c]u8 {
    var t = arg_t;
    _ = &t;
    var prev_token = arg_prev_token;
    _ = &prev_token;
    var piece: [*c]u8 = (blk: {
        const tmp = token;
        if (tmp >= 0) break :blk t.*.vocab + @as(usize, @intCast(tmp)) else break :blk t.*.vocab - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
    }).*;
    _ = &piece;
    if ((prev_token == @as(c_int, 1)) and (@as(c_int, @bitCast(@as(c_uint, piece[@as(c_uint, @intCast(@as(c_int, 0)))]))) == @as(c_int, ' '))) {
        piece += 1;
    }
    var byte_val: u8 = undefined;
    _ = &byte_val;
    if (llama.sscanf(piece, "<0x%02hhX>", &byte_val) == @as(c_int, 1)) {
        piece = @as([*c]u8, @ptrCast(@alignCast(@as([*c]u8, @ptrCast(@alignCast(&t.*.byte_pieces)))))) + @as(usize, @bitCast(@as(isize, @intCast(@as(c_int, @bitCast(@as(c_uint, byte_val))) * @as(c_int, 2)))));
    }
    return piece;
}

pub fn safe_printf(arg_piece: [*c]u8) void {
    var piece = arg_piece;
    _ = &piece;
    if (piece == @as([*c]u8, @ptrCast(@alignCast(@as(?*anyopaque, @ptrFromInt(@as(c_int, 0))))))) {
        return;
    }
    if (@as(c_int, @bitCast(@as(c_uint, piece[@as(c_uint, @intCast(@as(c_int, 0)))]))) == @as(c_int, '\x00')) {
        return;
    }
    if (@as(c_int, @bitCast(@as(c_uint, piece[@as(c_uint, @intCast(@as(c_int, 1)))]))) == @as(c_int, '\x00')) {
        var byte_val: u8 = @as(u8, @bitCast(piece[@as(c_uint, @intCast(@as(c_int, 0)))]));
        _ = &byte_val;
        if (!(((@as(c_int, @bitCast(@as(c_uint, (blk: {
            const tmp = @as(c_int, @bitCast(@as(c_uint, byte_val)));
            if (tmp >= 0) break :blk llama.__ctype_b_loc().* + @as(usize, @intCast(tmp)) else break :blk llama.__ctype_b_loc().* - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
        }).*))) & @as(c_int, @bitCast(@as(c_uint, @as(c_ushort, @bitCast(@as(c_short, @truncate(llama._ISprint)))))))) != 0) or ((@as(c_int, @bitCast(@as(c_uint, (blk: {
            const tmp = @as(c_int, @bitCast(@as(c_uint, byte_val)));
            if (tmp >= 0) break :blk llama.__ctype_b_loc().* + @as(usize, @intCast(tmp)) else break :blk llama.__ctype_b_loc().* - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
        }).*))) & @as(c_int, @bitCast(@as(c_uint, @as(c_ushort, @bitCast(@as(c_short, @truncate(llama._ISspace)))))))) != 0))) {
            return;
        }
    }
    _ = llama.printf("%s", piece);
}

fn str_lookup(arg_str: [*c]u8, arg_sorted_vocab: [*c]TokenIndex, arg_vocab_size: c_int) c_int {
    var str = arg_str;
    _ = &str;
    var sorted_vocab = arg_sorted_vocab;
    _ = &sorted_vocab;
    var vocab_size = arg_vocab_size;
    _ = &vocab_size;
    var tok: TokenIndex = TokenIndex{
        .str = str,
        .id = 0,
    };
    _ = &tok;
    var res: [*c]TokenIndex = @as([*c]TokenIndex, @ptrCast(@alignCast(llama.bsearch(@as(?*const anyopaque, @ptrCast(&tok)), @as(?*const anyopaque, @ptrCast(sorted_vocab)), @as(usize, @bitCast(@as(c_long, vocab_size))), @sizeOf(TokenIndex), &compare_tokens))));
    _ = &res;
    return if (res != @as([*c]TokenIndex, @ptrCast(@alignCast(@as(?*anyopaque, @ptrFromInt(@as(c_int, 0))))))) res.*.id else -@as(c_int, 1);
}

pub fn encode(arg_t: [*c]Tokenizer, arg_text: [*c]const u8, arg_bos: i8, arg_eos: i8, arg_tokens: [*c]c_int, arg_n_tokens: *usize) void {
    var t = arg_t;
    _ = &t;
    var text = arg_text;
    _ = &text;
    var bos = arg_bos;
    _ = &bos;
    var eos = arg_eos;
    _ = &eos;
    var tokens = arg_tokens;
    _ = &tokens;
    var n_tokens = arg_n_tokens;
    _ = &n_tokens;
    if (text == @as([*c]u8, @ptrCast(@alignCast(@as(?*anyopaque, @ptrFromInt(@as(c_int, 0))))))) {
        _ = llama.fprintf(llama.stderr, "cannot encode NULL text\n");
        llama.exit(@as(c_int, 1));
    }
    if (t.*.sorted_vocab == @as([*c]TokenIndex, @ptrCast(@alignCast(@as(?*anyopaque, @ptrFromInt(@as(c_int, 0))))))) {
        t.*.sorted_vocab = @as([*c]TokenIndex, @ptrCast(@alignCast(llama.malloc(@as(c_ulong, @bitCast(@as(c_long, t.*.vocab_size))) *% @sizeOf(TokenIndex)))));
        {
            var i: c_int = 0;
            _ = &i;
            while (i < t.*.vocab_size) : (i += 1) {
                (blk: {
                    const tmp = i;
                    if (tmp >= 0) break :blk t.*.sorted_vocab + @as(usize, @intCast(tmp)) else break :blk t.*.sorted_vocab - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*.str = (blk: {
                    const tmp = i;
                    if (tmp >= 0) break :blk t.*.vocab + @as(usize, @intCast(tmp)) else break :blk t.*.vocab - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*;
                (blk: {
                    const tmp = i;
                    if (tmp >= 0) break :blk t.*.sorted_vocab + @as(usize, @intCast(tmp)) else break :blk t.*.sorted_vocab - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*.id = i;
            }
        }
        llama.qsort(@as(?*anyopaque, @ptrCast(t.*.sorted_vocab)), @as(usize, @bitCast(@as(c_long, t.*.vocab_size))), @sizeOf(TokenIndex), &compare_tokens);
    }
    var str_buffer: [*c]u8 = @as([*c]u8, @ptrCast(@alignCast(llama.malloc(@as(c_ulong, @bitCast(@as(c_ulong, ((t.*.max_token_length *% @as(c_uint, @bitCast(@as(c_int, 2)))) +% @as(c_uint, @bitCast(@as(c_int, 1)))) +% @as(c_uint, @bitCast(@as(c_int, 2)))))) *% @sizeOf(u8)))));
    _ = &str_buffer;
    var str_len: usize = 0;
    _ = &str_len;
    n_tokens.* = 0;
    if (bos != 0) {
        (blk: {
            const tmp = blk_1: {
                const ref = &n_tokens.*;
                const tmp_2 = ref.*;
                ref.* += 1;
                break :blk_1 tmp_2;
            };
            if (tmp >= 0) break :blk tokens + @as(usize, @intCast(tmp)) else break :blk tokens - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
        }).* = 1;
    }
    if (@as(c_int, @bitCast(@as(c_uint, text[@as(c_uint, @intCast(@as(c_int, 0)))]))) != @as(c_int, '\x00')) {
        var dummy_prefix: c_int = str_lookup(@as([*c]u8, @ptrCast(@volatileCast(@constCast(" ")))), t.*.sorted_vocab, t.*.vocab_size);
        _ = &dummy_prefix;
        (blk: {
            const tmp = blk_1: {
                const ref = &n_tokens.*;
                const tmp_2 = ref.*;
                ref.* += 1;
                break :blk_1 tmp_2;
            };
            if (tmp >= 0) break :blk tokens + @as(usize, @intCast(tmp)) else break :blk tokens - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
        }).* = dummy_prefix;
    }
    {
        var c: [*c]const u8 = text;
        _ = &c;
        while (@as(c_int, @bitCast(@as(c_uint, c.*))) != @as(c_int, '\x00')) : (c += 1) {
            if ((@as(c_int, @bitCast(@as(c_uint, c.*))) & @as(c_int, 192)) != @as(c_int, 128)) {
                str_len = 0;
            }
            str_buffer[blk: {
                const ref = &str_len;
                const tmp = ref.*;
                ref.* +%= 1;
                break :blk tmp;
            }] = c.*;
            str_buffer[str_len] = '\x00';
            if (((@as(c_int, @bitCast(@as(c_uint, (c + @as(usize, @bitCast(@as(isize, @intCast(@as(c_int, 1)))))).*))) & @as(c_int, 192)) == @as(c_int, 128)) and (str_len < @as(usize, @bitCast(@as(c_long, @as(c_int, 4)))))) {
                continue;
            }
            var id: c_int = str_lookup(str_buffer, t.*.sorted_vocab, t.*.vocab_size);
            _ = &id;
            if (id != -@as(c_int, 1)) {
                (blk: {
                    const tmp = blk_1: {
                        const ref = &n_tokens.*;
                        const tmp_2 = ref.*;
                        ref.* += 1;
                        break :blk_1 tmp_2;
                    };
                    if (tmp >= 0) break :blk tokens + @as(usize, @intCast(tmp)) else break :blk tokens - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).* = id;
            } else {
                {
                    var i: c_int = 0;
                    _ = &i;
                    while (@as(usize, @bitCast(@as(c_long, i))) < str_len) : (i += 1) {
                        (blk: {
                            const tmp = blk_1: {
                                const ref = &n_tokens.*;
                                const tmp_2 = ref.*;
                                ref.* += 1;
                                break :blk_1 tmp_2;
                            };
                            if (tmp >= 0) break :blk tokens + @as(usize, @intCast(tmp)) else break :blk tokens - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                        }).* = @as(c_int, @bitCast(@as(c_uint, @as(u8, @bitCast((blk: {
                            const tmp = i;
                            if (tmp >= 0) break :blk str_buffer + @as(usize, @intCast(tmp)) else break :blk str_buffer - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                        }).*))))) + @as(c_int, 3);
                    }
                }
            }
            str_len = 0;
        }
    }
    while (true) {
        var best_score: f32 = @as(f32, @floatCast(-10000000000.0));
        _ = &best_score;
        var best_id: c_int = -@as(c_int, 1);
        _ = &best_id;
        var best_idx: c_int = -@as(c_int, 1);
        _ = &best_idx;
        {
            var i: c_int = 0;
            _ = &i;
            while (i < (n_tokens.* - @as(c_int, 1))) : (i += 1) {
                _ = llama.sprintf(str_buffer, "%s%s", (blk: {
                    const tmp = (blk_1: {
                        const tmp_2 = i;
                        if (tmp_2 >= 0) break :blk_1 tokens + @as(usize, @intCast(tmp_2)) else break :blk_1 tokens - ~@as(usize, @bitCast(@as(isize, @intCast(tmp_2)) +% -1));
                    }).*;
                    if (tmp >= 0) break :blk t.*.vocab + @as(usize, @intCast(tmp)) else break :blk t.*.vocab - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*, (blk: {
                    const tmp = (blk_1: {
                        const tmp_2 = i + @as(c_int, 1);
                        if (tmp_2 >= 0) break :blk_1 tokens + @as(usize, @intCast(tmp_2)) else break :blk_1 tokens - ~@as(usize, @bitCast(@as(isize, @intCast(tmp_2)) +% -1));
                    }).*;
                    if (tmp >= 0) break :blk t.*.vocab + @as(usize, @intCast(tmp)) else break :blk t.*.vocab - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*);
                var id: c_int = str_lookup(str_buffer, t.*.sorted_vocab, t.*.vocab_size);
                _ = &id;
                if ((id != -@as(c_int, 1)) and ((blk: {
                    const tmp = id;
                    if (tmp >= 0) break :blk t.*.vocab_scores + @as(usize, @intCast(tmp)) else break :blk t.*.vocab_scores - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).* > best_score)) {
                    best_score = (blk: {
                        const tmp = id;
                        if (tmp >= 0) break :blk t.*.vocab_scores + @as(usize, @intCast(tmp)) else break :blk t.*.vocab_scores - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                    }).*;
                    best_id = id;
                    best_idx = i;
                }
            }
        }
        if (best_idx == -@as(c_int, 1)) {
            break;
        }
        (blk: {
            const tmp = best_idx;
            if (tmp >= 0) break :blk tokens + @as(usize, @intCast(tmp)) else break :blk tokens - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
        }).* = best_id;
        {
            var i: c_int = best_idx + @as(c_int, 1);
            _ = &i;
            while (i < (n_tokens.* - @as(c_int, 1))) : (i += 1) {
                (blk: {
                    const tmp = i;
                    if (tmp >= 0) break :blk tokens + @as(usize, @intCast(tmp)) else break :blk tokens - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).* = (blk: {
                    const tmp = i + @as(c_int, 1);
                    if (tmp >= 0) break :blk tokens + @as(usize, @intCast(tmp)) else break :blk tokens - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
                }).*;
            }
        }
        n_tokens.* -= 1;
    }
    if (eos != 0) {
        (blk: {
            const tmp = blk_1: {
                const ref = &n_tokens.*;
                const tmp_2 = ref.*;
                ref.* += 1;
                break :blk_1 tmp_2;
            };
            if (tmp >= 0) break :blk tokens + @as(usize, @intCast(tmp)) else break :blk tokens - ~@as(usize, @bitCast(@as(isize, @intCast(tmp)) +% -1));
        }).* = 2;
    }
    llama.free(@as(?*anyopaque, @ptrCast(str_buffer)));
}

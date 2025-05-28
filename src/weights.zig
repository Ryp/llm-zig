const std = @import("std");

const transformer = @import("transformer.zig");
const llama = @import("llama.zig");

pub const TransformerWeights = struct {
    mmap_fd: c_int,
    mmap_handle: []align(std.heap.page_size_min) u8,

    token_embedding_table: [*c]f32,
    rms_att_weight: [*c]f32,
    rms_ffn_weight: [*c]f32,
    wq: [*c]f32,
    wk: [*c]f32,
    wv: [*c]f32,
    wo: [*c]f32,
    w1: [*c]f32,
    w2: [*c]f32,
    w3: [*c]f32,
    rms_final_weight: [*c]f32,
    wcls: [*c]f32,
};

pub const SerializedConfig = extern struct {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    seq_len: u32,
};

pub fn memory_map_weights(w: *TransformerWeights, p: transformer.Config, arg_ptr: [*]f32, shared_weights: bool) void {
    const head_size = @divTrunc(p.dim, p.n_heads);
    const n_layers = p.n_layers;

    var ptr = arg_ptr;
    w.token_embedding_table = ptr;
    ptr += p.vocab_size * p.dim;
    w.rms_att_weight = ptr;
    ptr += n_layers * p.dim;
    w.wq = ptr;
    ptr += (n_layers * p.dim * p.n_heads * head_size);
    w.wk = ptr;
    ptr += (n_layers * p.dim * p.n_kv_heads * head_size);
    w.wv = ptr;
    ptr += (n_layers * p.dim * p.n_kv_heads * head_size);
    w.wo = ptr;
    ptr += n_layers * p.n_heads * head_size * p.dim;
    w.rms_ffn_weight = ptr;
    ptr += n_layers * p.dim;
    w.w1 = ptr;
    ptr += n_layers * p.dim * p.hidden_dim;
    w.w2 = ptr;
    ptr += n_layers * p.hidden_dim * p.dim;
    w.w3 = ptr;
    ptr += n_layers * p.dim * p.hidden_dim;
    w.rms_final_weight = ptr;
    ptr += p.dim;
    ptr += @divTrunc(p.seq_len * head_size, 2);
    ptr += @divTrunc(p.seq_len * head_size, 2);

    w.wcls = if (shared_weights) w.token_embedding_table else ptr;
}

pub fn open_weights_from_file(checkpoint: [:0]const u8, config: *transformer.Config, file_size: *isize) !TransformerWeights {
    var weights: TransformerWeights = undefined;

    var file: ?*llama.FILE = llama.fopen(checkpoint, "rb");
    _ = &file;
    if (file == null) {
        std.debug.print("Couldn't open file {s}\n", .{checkpoint});
        unreachable;
    }

    var serialized_config: SerializedConfig = undefined;

    if (llama.fread(@as(?*anyopaque, @ptrCast(&serialized_config)), @sizeOf(SerializedConfig), @as(c_ulong, @bitCast(@as(c_long, @as(c_int, 1)))), file) != @as(c_ulong, @bitCast(@as(c_long, @as(c_int, 1))))) {
        llama.exit(@as(c_int, 1));
    }

    const shared_weights: bool = serialized_config.vocab_size > 0; // FIXME
    std.debug.assert(@as(i32, @bitCast(serialized_config.vocab_size)) > 0);

    config.* = .{
        .dim = serialized_config.dim,
        .hidden_dim = serialized_config.hidden_dim,
        .n_layers = serialized_config.n_layers,
        .n_heads = serialized_config.n_heads,
        .n_kv_heads = serialized_config.n_kv_heads,
        .vocab_size = serialized_config.vocab_size,
        .seq_len = serialized_config.seq_len,
    };

    std.debug.print("config: {}\n", .{config});

    _ = llama.fseek(file, 0, 2); // move file pointer to end of file
    // fseek(file, 0, SEEK_END);
    file_size.* = llama.ftell(file);
    _ = llama.fclose(file);

    weights.mmap_fd = try std.posix.open(checkpoint, .{ .ACCMODE = .RDONLY }, 0);
    errdefer std.posix.close(weights.mmap_fd);

    weights.mmap_handle = try std.posix.mmap(null, @bitCast(file_size.*), std.posix.PROT.READ, .{ .TYPE = .PRIVATE }, weights.mmap_fd, 0);
    errdefer std.posix.munmap(weights.mmap_handle);

    // FIXME use slice
    const data = @as([*]f32, @ptrCast(weights.mmap_handle));

    const weights_ptr = data + (@sizeOf(SerializedConfig) / @sizeOf(f32));

    memory_map_weights(&weights, config.*, weights_ptr, shared_weights);

    return weights;
}

pub fn close_weights_from_file(weights: TransformerWeights) void {
    std.posix.munmap(weights.mmap_handle);
    std.posix.close(weights.mmap_fd);
}

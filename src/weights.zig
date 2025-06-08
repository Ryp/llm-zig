const std = @import("std");

const transformer = @import("transformer.zig");
const tensor = @import("tensor/tensor.zig");
const layout = @import("tensor/layout.zig");

// Only used for getting data from disk
const SerializedConfig = extern struct {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: u32,
    seq_len: u32,
};

pub const TransformerWeights = struct {
    checkpoint_file: std.fs.File,
    checkpoint_mmap_ptr: []align(std.heap.page_size_min) u8,

    token_embedding_table: tensor.ConstTensor(f32, 2),
    rms_att_weight: tensor.ConstTensor(f32, 2),
    rms_ffn_weight: tensor.ConstTensor(f32, 2),
    wq: tensor.ConstTensor(f32, 3),
    wk: tensor.ConstTensor(f32, 3),
    wv: tensor.ConstTensor(f32, 3),
    wo: []const f32,
    w1: []const f32,
    w2: []const f32,
    w3: []const f32,
    rms_final_weight: tensor.ConstTensor(f32, 1),
};

pub fn open_weights_from_file(checkpoint: [:0]const u8, config: *transformer.Config) !TransformerWeights {
    var weights: TransformerWeights = undefined;

    weights.checkpoint_file = if (std.fs.cwd().openFile(checkpoint, .{ .mode = .read_only })) |f| f else |err| {
        std.debug.print("error: couldn't open checkpoint file: '{s}'\n", .{checkpoint});
        return err;
    };
    errdefer weights.checkpoint_file.close();

    const checkpoint_size_bytes = try weights.checkpoint_file.getEndPos();

    // Read config first
    var serialized_config: SerializedConfig = undefined;

    const config_bytes: []u8 = std.mem.asBytes(&serialized_config);
    const bytes_read = try weights.checkpoint_file.read(config_bytes);
    std.debug.assert(bytes_read == config_bytes.len);

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

    weights.checkpoint_mmap_ptr = try std.posix.mmap(null, checkpoint_size_bytes, std.posix.PROT.READ, .{ .TYPE = .PRIVATE }, weights.checkpoint_file.handle, 0);
    errdefer std.posix.munmap(weights.checkpoint_mmap_ptr);

    const weights_slice = std.mem.bytesAsSlice(f32, weights.checkpoint_mmap_ptr[@sizeOf(SerializedConfig)..]);

    memory_map_weights(&weights, config.*, weights_slice);

    return weights;
}

pub fn close_weights_from_file(weights: TransformerWeights) void {
    std.posix.munmap(weights.checkpoint_mmap_ptr);

    weights.checkpoint_file.close();
}

fn memory_map_weights(w: *TransformerWeights, p: transformer.Config, arg_ptr: []const f32) void {
    const head_size = @divTrunc(p.dim, p.n_heads);
    const n_layers = p.n_layers;

    var ptr = arg_ptr;

    w.token_embedding_table = tensor.ConstTensor(f32, 2).init(layout.right(2, .{p.vocab_size, p.dim}), read_advance_ptr(&ptr, p.vocab_size * p.dim));
    w.rms_att_weight = tensor.ConstTensor(f32, 2).init(layout.right(2, .{n_layers, p.dim}), read_advance_ptr(&ptr, n_layers * p.dim));
    w.wq = tensor.ConstTensor(f32, 3).init(layout.right(3, .{n_layers, p.dim, p.n_heads * head_size}), read_advance_ptr(&ptr, n_layers * p.dim * p.n_heads * head_size));
    w.wk = tensor.ConstTensor(f32, 3).init(layout.right(3, .{n_layers, p.dim, p.n_kv_heads * head_size}), read_advance_ptr(&ptr, n_layers * p.dim * p.n_kv_heads * head_size));
    w.wv = tensor.ConstTensor(f32, 3).init(layout.right(3, .{n_layers, p.dim, p.n_kv_heads * head_size}), read_advance_ptr(&ptr, n_layers * p.dim * p.n_kv_heads * head_size));
    w.wo = read_advance_ptr(&ptr, n_layers * p.n_heads * head_size * p.dim);
    w.rms_ffn_weight = tensor.ConstTensor(f32, 2).init(layout.right(2, .{n_layers, p.dim}), read_advance_ptr(&ptr, n_layers * p.dim));
    w.w1 = read_advance_ptr(&ptr, n_layers * p.dim * p.hidden_dim);
    w.w2 = read_advance_ptr(&ptr, n_layers * p.hidden_dim * p.dim);
    w.w3 = read_advance_ptr(&ptr, n_layers * p.dim * p.hidden_dim);
    w.rms_final_weight = tensor.ConstTensor(f32, 1).init(layout.right(1, .{p.dim}), read_advance_ptr(&ptr, p.dim));
}

// Helper to load weights easily
fn read_advance_ptr(ptr: *[]const f32, size_elements: usize) []const f32 {
    const slice =  ptr.*[0..size_elements];
    ptr.* = ptr.*[size_elements..];
    return slice;
}

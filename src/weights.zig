const std = @import("std");

const transformer = @import("transformer.zig");
const tensor = @import("tensor/tensor.zig");
const layout = @import("tensor/layout.zig");

const native_os = @import("builtin").os.tag;

// Only used for getting data from disk
const SerializedConfig = extern struct {
    dim: u32,
    hidden_dim: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    vocab_size: i32,
    seq_len: u32,
};

pub const TransformerWeights = struct {
    checkpoint_file: std.fs.File,
    checkpoint_mmap_ptr: []align(std.heap.page_size_min) u8,

    token_embedding_table: tensor.ConstTensor(f32, 2),
    rms_attn: tensor.ConstTensor(f32, 2),
    rms_ffn: tensor.ConstTensor(f32, 2),
    wq: tensor.ConstTensor(f32, 3),
    wk: tensor.ConstTensor(f32, 3),
    wv: tensor.ConstTensor(f32, 3),
    wo: tensor.ConstTensor(f32, 3),
    w1: tensor.ConstTensor(f32, 3),
    w2: tensor.ConstTensor(f32, 3),
    w3: tensor.ConstTensor(f32, 3),
    rms_final: tensor.ConstTensor(f32, 1),
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

    config.* = .{
        .dim = serialized_config.dim,
        .hidden_dim = serialized_config.hidden_dim,
        .n_layers = serialized_config.n_layers,
        .n_heads = serialized_config.n_heads,
        .n_kv_heads = serialized_config.n_kv_heads,
        .vocab_size = @intCast(serialized_config.vocab_size),
        .seq_len = serialized_config.seq_len,
    };

    std.debug.print("config: {}\n", .{config});

    weights.checkpoint_mmap_ptr = try switch (native_os) {
        .linux, .macos => std.posix.mmap(null, checkpoint_size_bytes, std.posix.PROT.READ, .{ .TYPE = .PRIVATE }, weights.checkpoint_file.handle, 0),
        .windows => @panic("Unsupported OS"),
        else => @panic("Unsupported OS"),
    };
    errdefer std.posix.munmap(weights.checkpoint_mmap_ptr);

    const weights_slice = std.mem.bytesAsSlice(f32, weights.checkpoint_mmap_ptr[@sizeOf(SerializedConfig)..]);

    memory_map_weights(&weights, config.*, weights_slice);

    return weights;
}

pub fn close_weights_from_file(weights: TransformerWeights) void {
    switch (native_os) {
        .linux, .macos => {
            std.posix.munmap(weights.checkpoint_mmap_ptr);
        },
        .windows => @panic("Unsupported OS"),
        else => @panic("Unsupported OS"),
    }

    weights.checkpoint_file.close();
}

fn memory_map_weights(w: *TransformerWeights, p: transformer.Config, arg_ptr: []const f32) void {
    const head_dim = @divExact(p.dim, p.n_heads);

    var ptr = arg_ptr;

    w.token_embedding_table = tensor.ConstTensor(f32, 2).init(layout.right(2, .{ p.vocab_size, p.dim }), read_advance_ptr(&ptr, p.vocab_size * p.dim));
    w.rms_attn = tensor.ConstTensor(f32, 2).init(layout.right(2, .{ p.n_layers, p.dim }), read_advance_ptr(&ptr, p.n_layers * p.dim));
    w.wq = tensor.ConstTensor(f32, 3).init(layout.right(3, .{ p.n_layers, p.n_heads * head_dim, p.dim }), read_advance_ptr(&ptr, p.n_layers * p.dim * (p.n_heads * head_dim)));
    w.wk = tensor.ConstTensor(f32, 3).init(layout.right(3, .{ p.n_layers, p.n_kv_heads * head_dim, p.dim }), read_advance_ptr(&ptr, p.n_layers * p.dim * (p.n_kv_heads * head_dim)));
    w.wv = tensor.ConstTensor(f32, 3).init(layout.right(3, .{ p.n_layers, p.n_kv_heads * head_dim, p.dim }), read_advance_ptr(&ptr, p.n_layers * p.dim * (p.n_kv_heads * head_dim)));
    w.wo = tensor.ConstTensor(f32, 3).init(layout.right(3, .{ p.n_layers, p.dim, p.n_heads * head_dim }), read_advance_ptr(&ptr, p.n_layers * (p.n_heads * head_dim) * p.dim));
    w.rms_ffn = tensor.ConstTensor(f32, 2).init(layout.right(2, .{ p.n_layers, p.dim }), read_advance_ptr(&ptr, p.n_layers * p.dim));
    w.w1 = tensor.ConstTensor(f32, 3).init(layout.right(3, .{ p.n_layers, p.hidden_dim, p.dim }), read_advance_ptr(&ptr, p.n_layers * p.dim * p.hidden_dim));
    w.w2 = tensor.ConstTensor(f32, 3).init(layout.right(3, .{ p.n_layers, p.dim, p.hidden_dim }), read_advance_ptr(&ptr, p.n_layers * p.hidden_dim * p.dim));
    w.w3 = tensor.ConstTensor(f32, 3).init(layout.right(3, .{ p.n_layers, p.hidden_dim, p.dim }), read_advance_ptr(&ptr, p.n_layers * p.dim * p.hidden_dim));
    w.rms_final = tensor.ConstTensor(f32, 1).init(layout.right(1, .{p.dim}), read_advance_ptr(&ptr, p.dim));
}

// Helper to load weights easily
fn read_advance_ptr(ptr: *[]const f32, size_elements: usize) []const f32 {
    const slice = ptr.*[0..size_elements];
    ptr.* = ptr.*[size_elements..];
    return slice;
}

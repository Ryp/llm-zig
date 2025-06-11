comptime {
    _ = @import("tensor/tuple.zig");
    _ = @import("tensor/shape.zig");
    _ = @import("tensor/tensor.zig");
    _ = @import("tensor/stride.zig");
}

test {
    @import("std").testing.refAllDeclsRecursive(@This());
}

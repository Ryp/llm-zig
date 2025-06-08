const std = @import("std");

const tuple = @import("tuple.zig");

pub fn Shape(comptime rank: usize) type {
    return tuple.Vector(usize, rank);
}

test "Simple shape" {
    const Shape3 = Shape(3);
    const shape = Shape3{ 1, 2, 3 };

    try std.testing.expectEqual(1, shape[0]);
    try std.testing.expectEqual(2, shape[1]);
    try std.testing.expectEqual(3, shape[2]);
}

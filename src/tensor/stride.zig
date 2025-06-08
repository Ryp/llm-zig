const std = @import("std");

const tuple = @import("tuple.zig");
const Shape = @import("shape.zig").Shape;

pub fn Stride(comptime rank: usize) type {
    return tuple.Vector(usize, rank);
}

pub fn right(rank: comptime_int, shape: Shape(rank)) Stride(rank) {
    var stride: Stride(rank) = undefined;

    stride[rank - 1] = 1;

    // Prefix product
    for (1..rank) |index| {
        const back_index = rank - index;
        stride[back_index - 1] = shape[back_index] * stride[back_index];
    }

    return stride;
}

pub fn left(rank: comptime_int, shape: Shape(rank)) Stride(rank) {
    var stride: Stride(rank) = undefined;

    stride[0] = 1;

    // Prefix product
    for (shape[0 .. rank - 1], 0..) |shape_elt, index| {
        stride[index + 1] = shape_elt * stride[index];
    }

    return stride;
}

test "Simple stride" {
    const Stride3 = Stride(3);
    const stride = Stride3{ 1, 2, 3 };

    try std.testing.expectEqual(1, stride[0]);
    try std.testing.expectEqual(2, stride[1]);
    try std.testing.expectEqual(3, stride[2]);
}

test "Stride Right" {
    const rank = 2;
    const ShapeType = Shape(rank);
    const shape = ShapeType{ 8, 4 };
    const stride = right(rank, shape);

    try std.testing.expectEqual(4, stride[0]);
    try std.testing.expectEqual(1, stride[1]);
}

test "Stride Left" {
    const rank = 2;
    const ShapeType = Shape(rank);
    const shape = ShapeType{ 8, 4 };
    const stride = left(rank, shape);

    try std.testing.expectEqual(1, stride[0]);
    try std.testing.expectEqual(8, stride[1]);
}

test "Stride Left Dim3" {
    const rank = 3;
    const ShapeType = Shape(rank);
    const shape = ShapeType{ 8, 4, 2 };
    const stride = left(rank, shape);

    try std.testing.expectEqual(1, stride[0]);
    try std.testing.expectEqual(8, stride[1]);
    try std.testing.expectEqual(32, stride[2]);
}

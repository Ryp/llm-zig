const std = @import("std");

const Tuple = @import("tuple.zig").Vector;
const Shape = @import("shape.zig").Shape;
const Stride = @import("stride.zig").Stride;
const stride_left = @import("stride.zig").left;
const stride_right = @import("stride.zig").right;

pub fn Layout(comptime rank: usize) type {
    return struct {
        shape: Shape(rank),
        stride: Stride(rank),

        pub fn size_elements(self: @This()) usize {
            var size: usize = 0;

            inline for (self.shape, self.stride) |shape_elt, stride_elt| {
                size = @max(size, shape_elt * stride_elt);
            }

            return size;
        }

        pub fn element_offset(self: @This(), element_index: Tuple(usize, rank)) usize {
            var offset: usize = 0;

            inline for (element_index, self.stride) |index_elt, stride_elt| {
                offset += index_elt * stride_elt;
            }

            return offset;
        }

        pub fn coord_to_index(self: @This(), coord: Shape(rank)) usize {
            var index: usize = 0;
            inline for (coord, self.stride) |coord_elt, stride_elt| {
                index += coord_elt * stride_elt;
            }
            return index;
        }
    };
}

pub fn left(rank: comptime_int, shape: Shape(rank)) Layout(rank) {
    return .{
        .shape = shape,
        .stride = stride_left(rank, shape),
    };
}

pub fn right(rank: comptime_int, shape: Shape(rank)) Layout(rank) {
    return .{
        .shape = shape,
        .stride = stride_right(rank, shape),
    };
}

test "Simple layout" {
    const Layout3 = Layout(3);
    const layout = Layout3{
        .shape = .{ 1, 2, 3 },
        .stride = .{ 1, 2, 3 },
    };

    try std.testing.expectEqual(1, layout.stride[0]);
    try std.testing.expectEqual(2, layout.stride[1]);
    try std.testing.expectEqual(3, layout.stride[2]);
}

test "Layout Size" {
    const layout = Layout(2){
        .shape = .{ 2, 4 },
        .stride = .{ 1, 2 },
    };

    try std.testing.expectEqual(8, layout.size_elements());
    try std.testing.expectEqual(7, layout.element_offset(.{1, 3}));
}

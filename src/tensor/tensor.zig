const std = @import("std");

const layout = @import("layout.zig");
const shape = @import("shape.zig");
const stride = @import("stride.zig");
const tuple = @import("tuple.zig");

// DataType -> comptime
// Dim -> comptime
// Shape -> runtime
// Stride -> runtime

pub fn Tensor(comptime arg_data_type: type, comptime rank: comptime_int) type {
    return GenericTensor(false, arg_data_type, rank);
}

pub fn ConstTensor(comptime arg_data_type: type, comptime rank: comptime_int) type {
    return GenericTensor(true, arg_data_type, rank);
}

pub fn GenericTensor(comptime is_const: bool, comptime scalar_type: type, comptime rank: comptime_int) type {
    return struct {
        const Self = @This();
        const LayoutType = layout.Layout(rank);
        const SliceType = if (is_const) []const scalar_type else []scalar_type;

        comptime scalar_type: @TypeOf(scalar_type) = scalar_type,
        comptime rank: comptime_int = rank,

        layout: LayoutType,
        raw_data: SliceType,

        pub fn init(arg_layout: LayoutType, raw_data: SliceType) Self {
            return .{
                .layout = arg_layout,
                .raw_data = raw_data,
            };
        }

        const Vector = tuple.Vector(usize, rank);

        pub fn at(self: *Self, element: Vector) *scalar_type {
            const element_offset = self.layout.element_offset(element);
            return &self.raw_data[element_offset];
        }

        pub fn sub_tensor(self: Self, comptime sub_rank: usize, offset: usize) GenericTensor(is_const, self.scalar_type, self.rank - 1) {
            const sub_shape = self.layout.shape[0..sub_rank] ++ self.layout.shape[sub_rank + 1 .. self.rank];
            const sub_stride = self.layout.stride[0..sub_rank] ++ self.layout.stride[sub_rank + 1 .. self.rank];

            const sub_offset_start = self.layout.stride[sub_rank] * offset;
            // FIXME What happens if the tensor is not contiguous?
            const sub_offset_end = self.layout.stride[sub_rank] * (offset + 1);

            return .{
                .layout = layout.Layout(self.rank - 1){
                    .shape = sub_shape.*,
                    .stride = sub_stride.*,
                },
                .raw_data = self.raw_data[sub_offset_start..sub_offset_end],
            };
        }
    };
}

test "Tensor" {
    var raw_data = [_]f32{0.0} ** (4 * 8);

    var tensor_a = Tensor(f32, 2).init(layout.left(2, .{ 4, 8 }), &raw_data);

    try std.testing.expectEqual(.{ 4, 8 }, tensor_a.layout.shape);

    tensor_a.at(.{ 0, 0 }).* = 1.0;
    try std.testing.expectEqual(1.0, tensor_a.at(.{ 0, 0 }).*);
    tensor_a.raw_data[0] = 0.0;
    try std.testing.expectEqual(0.0, tensor_a.at(.{ 0, 0 }).*);

    tensor_a.raw_data[1] = 2.0;
    try std.testing.expectEqual(2.0, tensor_a.at(.{ 1, 0 }).*);

    tensor_a.raw_data[4] = 3.0;
    try std.testing.expectEqual(3.0, tensor_a.at(.{ 0, 1 }).*);
}

test "Subtensor" {
    var raw_data = [_]f32{0.0} ** (2 * 4 * 8);

    var tensor_a = Tensor(f32, 3).init(layout.right(3, .{ 2, 4, 8 }), &raw_data);

    try std.testing.expectEqual(4 * 8, tensor_a.layout.stride[0]);

    var tensor_b = tensor_a.sub_tensor(0, 1);

    try std.testing.expectEqual(.{ 4, 8 }, tensor_b.layout.shape);
    try std.testing.expectEqual(.{ 8, 1 }, tensor_b.layout.stride);

    tensor_a.at(.{ 1, 2, 3 }).* = 15.0;

    try std.testing.expectEqual(15.0, tensor_b.at(.{ 2, 3 }).*);
}

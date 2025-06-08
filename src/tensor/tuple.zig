const std = @import("std");

pub fn Vector(comptime T: type, comptime rank: usize) type {
    return [rank]T;
}

test "Simple tuple" {
    const Size = Vector(u8, 3);
    const size = Size{ 1, 2, 3 };

    try std.testing.expectEqual(1, size[0]);
    try std.testing.expectEqual(2, size[1]);
    try std.testing.expectEqual(3, size[2]);
}

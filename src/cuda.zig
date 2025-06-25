const std = @import("std");

pub const c = @cImport({
    @cInclude("cuda.h");
    @cInclude("nvrtc.h");
    //@cInclude("cuda_runtime_api.h");
});

const increment_kernel = @embedFile("increment.cu");

pub fn cuda_main(allocator: *std.mem.Allocator) void {
    var device_count: i32 = 0;
    const device_index: i32 = 0;

    var device: c.CUdevice = undefined;
    var context: c.CUcontext = undefined;

    _ = c.cuInit(0);

    _ = c.cuDeviceGetCount(&device_count);
    std.debug.assert(device_count > 0);
    std.debug.print("Device count = {}\n", .{device_count});

    _ = c.cuDeviceGet(&device, device_index);

    _ = c.cuDeviceGet(&device, device_index);

    const attribute = c.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH;
    var attribute_value: i32 = undefined;
    _ = c.cuDeviceGetAttribute(&attribute_value, attribute, device);
    std.debug.print("CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = {}\n", .{attribute_value});

    // ?
    _ = c.cuDevicePrimaryCtxRetain(&context, device);
    _ = c.cuCtxSetCurrent(context);

    // Copy data from host to GPU
    const data = [_]f32{ 1.2, 2.8, 0.123 };
    const cu_slice = try device.htodCopy(f32, &data);
    defer cu_slice.free();
    std.debug.print("Copied array {d:.3} from system to GPU\n", .{data});

    // Compile and load the Kernel
    std.debug.print("Kernel program:\n{s}\n\n", .{increment_kernel});
    const ptx = try c.CuCompile.cudaText(increment_kernel, .{}, allocator);
    defer allocator.free(ptx);
    const module = try c.CuDevice.loadPtxText(ptx);
    const function = try module.getFunc("increment");
    std.debug.print("Compiled Cuda Kernel that increments each value by 1 and loaded into GPU\n", .{});

    // Run the kernel on the data
    try function.run(.{&cu_slice.device_ptr}, c.CuLaunchConfig{ .block_dim = .{ 3, 1, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 });
    std.debug.print("Ran the Kernel against the array in GPU\n", .{});

    // Retrieve incremented data back to the system
    const incremented_arr = try c.CuDevice.syncReclaim(f32, allocator, cu_slice);
    defer incremented_arr.deinit();
    std.debug.print("Retrieved incremented data {d:.3} from GPU to system\n", .{incremented_arr.items});
}

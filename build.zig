const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{});

    const exe = b.addExecutable(.{
        .name = "llm",
        .root_source_file = b.path("src/main.zig"),
        .optimize = optimize,
        .target = target,
    });

    const no_bin = b.option(bool, "no-bin", "skip emitting binary") orelse false;
    const enable_cuda = b.option(bool, "cuda", "enable CUDA support") orelse false;

    const exe_options = b.addOptions();
    exe_options.addOption(bool, "enable_cuda", enable_cuda);
    exe.root_module.addOptions("build_options", exe_options);

    if (no_bin) {
        b.getInstallStep().dependOn(&exe.step);
    } else {
        b.installArtifact(exe);
    }

    if (enable_cuda) {
        exe.linkSystemLibrary("cuda");
        exe.linkSystemLibrary("nvrtc");
        //exe.linkSystemLibrary("cudart"); // Link against the CUDA runtime library
        exe.addLibraryPath(std.Build.LazyPath{ .cwd_relative = "/usr/local/cuda/lib64" });
        exe.addIncludePath(std.Build.LazyPath{ .cwd_relative = "/usr/local/cuda/include" });
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the program");
    run_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/test.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}

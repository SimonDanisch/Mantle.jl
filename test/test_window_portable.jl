# The window questions that are the same on every backend, and the quickest ones.
#
# Split out of `test_window.jl` while that file still reached into the Vulkan
# backend, so that the property below — exactly the one that had no test on
# Metal — ran per backend. Both files are portable now. This one stays separate
# because `runtests.jl` runs each window file in a child process of its own with
# a deadline: a hang in the long file (see its header) does not hide these.
#
# `Window(backend, w, h)` is in the BACKEND_VOCABULARY and was answered by Lava
# alone. The Metal side said it could not, for a reason that turned out to be
# false — ColorTypes IS a Mantle dependency.

using Test
import Mantle
import GLFW
const M = Mantle
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8

include(joinpath(@__DIR__, "testbackend.jl"))

@testset "the portable Window is answered by this backend" begin
    # `Window(backend, w, h)` is in the BACKEND_VOCABULARY and was answered by
    # Lava alone; the Metal side said it could not, for a reason that turned out
    # to be false (ColorTypes IS a Mantle dependency). This is the test that
    # says every backend answers it, rather than one backend's tests passing.
    w = Mantle.Window(TESTBACKEND, 320, 200; title = "portable")
    try
        @test w isa Mantle.Window
        @test size(w) == (320, 200)
        @test eltype(w) == BGRA{N0f8}
        @test isopen(w)
        # The extent it reports IS the extent of the texture it hands out. A
        # window that lies about that breaks every pass that strides a buffer
        # by hand, and on a Retina panel the naive answer is twice too large.
        @test Mantle.target_extent(w) == (320, 200)
        Mantle.beginframe!(w)
        Mantle.acquire_next_image!(w)
        # The window hands out a target between acquire and present. Its EXTENT
        # is `target_extent` above and not a field of the thing handed out: a
        # `MTLTexture` carries its size, a `VK.ImageView` is a handle and the
        # extent lives on the image. Asserting `t.width` asked the Metal shape
        # of every backend, and a vendor-conditional test is not the fix --
        # `target_extent` is the portable question and is already checked.
        t = Mantle.target_view(w)
        @test t !== nothing
        # A manual pass draws into the acquired image through `currentimage`,
        # which a backend must answer for its window.
        @test Mantle.pass!(p -> :drawn, Mantle.Device(TESTBACKEND), Mantle.WindowTarget(w);
                           clear = (0f0, 0f0, 0f0, 1f0)) === :drawn
        Mantle.present_frame!(Mantle.Device(TESTBACKEND), w)
    finally
        close(w)
    end
end

struct YieldingWindowSurface
    entered::Channel{Nothing}
    resume::Channel{Nothing}
    fail::Bool
end
Mantle.fitsurface!(::YieldingWindowSurface, ::Int, ::Int) = false
Mantle.closesurface!(::YieldingWindowSurface) = nothing
function Mantle.acquire_next_image!(surface::YieldingWindowSurface)
    surface.fail && error("acquire failed")
    put!(surface.entered, nothing)
    take!(surface.resume)
    return nothing
end

@testset "a window is owned while its acquire yields" begin
    surface = YieldingWindowSurface(Channel{Nothing}(1), Channel{Nothing}(1), false)
    w = Mantle.Window(Mantle.platformwindow(16, 16, "acquire ownership"), surface, nothing, true)
    owner = @async Mantle.acquire_next_image!(w)
    take!(surface.entered)
    try
        @test w.acquirer === owner
        if w.acquirer === owner
            @test_throws ErrorException Mantle.acquire_next_image!(w)
            @test_throws ErrorException Mantle.fitwindow!(w)
            @test_throws ErrorException close(w)
            @test isopen(w)
        end
    finally
        put!(surface.resume, nothing)
        wait(owner)
        Mantle.presented!(w)
        close(w)
    end

    surface = YieldingWindowSurface(Channel{Nothing}(1), Channel{Nothing}(1), true)
    w = Mantle.Window(Mantle.platformwindow(16, 16, "failed acquire"), surface, nothing, true)
    try
        @test_throws ErrorException Mantle.acquire_next_image!(w)
        @test w.acquirer === nothing
    finally
        close(w)
    end
end

include(joinpath(@__DIR__, "spinwork.jl"))

@testset "a frame's waits let the thread's other tasks run" begin
    # A frame waits for its slot's previous frame, for an image, and in the present,
    # where NVIDIA's X11 driver waits out the whole GPU frame. All of them yield
    # (`hostwait`), so an input loop beside the render loop runs while frames are in
    # flight (MakieOrg/Makie.jl#5856). GPU-bound frames, 20 ms each, so the frame
    # waits whatever the display does; without vsync, nothing waits for it.
    dev = M.Device(TESTBACKEND)
    n = spinsteps(dev, 0.02)
    win = M.Window(TESTBACKEND, 64, 64; title = "waits yield", vsync = false)
    stop = Ref(false)
    ticks, task = ticker(stop)
    try
        g = M.Graph(dev)
        M.dispatch!(g, hw_spin!, (M.Buffer(dev, zeros(UInt32, 1)), n), 1; name = "spin")
        M.render!(g, "present", M.Surface(g, win) => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        end
        plan = M.Plan(g)
        foreach(_ -> M.run!(plan), 1:3)                 # compiled, swapchain settled
        k, s = progress_during(() -> foreach(_ -> M.run!(plan), 1:20), ticks)
        Mantle.flush!(dev)
        # Twenty frames of 20 ms; at least one tick per 10 ms of them, where the
        # ticker manages about five and a thread held in the waits none.
        @test s > 0.2
        @test k >= s / 0.01
    finally
        stop[] = true
        wait(task)
        close(win)
    end
end

# Window hints are GLFW-global. A Mantle window asks for `CLIENT_API = NO_API`,
# because it presents through Vulkan or Metal, and it left the hint set: the next
# OpenGL window anything in the process opened had no context to make current.
# RayMakie's comparison against GLMakie failed that way, with NO_WINDOW_CONTEXT.
@testset "a Mantle window leaves GLFW's hints as it found them" begin
    w = Mantle.Window(TESTBACKEND, 64, 64; title = "hints")
    close(w)
    # Hidden: a visible window can hang on XWayland (see `runtests.jl`).
    gl = GLFW.Window(; name = "gl", resolution = (16, 16), visible = false, focus = false)
    try
        @test GLFW.GetCurrentContext().handle == gl.handle
    finally
        GLFW.DestroyWindow(gl)
    end
end

# Integers of a width a device has no type for compute on the device what they
# compute on the host.
#
# LLVM's switch-to-lookup builds a bitmap as wide as its table: Hikari's shader
# materials tested a closure lobe against four of seven values as
# `lshr i7 -50, (trunc i8 %x to i7)`, and Lava emitted the `trunc` as a conversion
# between two 8-bit integers, which spirv-val refuses. Lava now carries every odd
# width in the next native one (`widen_odd_integers!`). Its `test_odd_integers.jl`
# checks the pass on the host; this runs the same IR on the device, through
# `llvmcall`, against the host's LLVM computing in the odd widths, for every pair
# of 8-bit inputs.

using Test, Mantle
import KernelAbstractions as KA
include(joinpath(@__DIR__, "testbackend.jl"))

const ODD_INTEGER_BODIES = (
    # Hikari's bitmap: whether `x` is not one of 1, 2, 3, 6.
    bitmap = """
        %c = icmp ugt i8 %x, 6
        %cast = trunc i8 %x to i7
        %down = lshr i7 -50, %cast
        %bit = and i7 %down, 1
        %m = icmp eq i7 %bit, 0
        %s = select i1 %c, i1 true, i1 %m
        %r = zext i1 %s to i32
        ret i32 %r""",
    # Every operation that can set the bits above the width, then read as signed.
    wrapping = """
        %a = trunc i8 %x to i7
        %b = trunc i8 %y to i7
        %s = add i7 %a, %b
        %m = mul i7 %s, %b
        %h = shl i7 %m, 3
        %d = sub i7 %h, %a
        %r = sext i7 %d to i32
        ret i32 %r""",
    # Reading as signed: a shift, division, a compare.
    signed = """
        %a = trunc i8 %x to i7
        %b = trunc i8 %y to i7
        %q = ashr i7 %a, 2
        %dl = and i7 %b, 15
        %dv = or i7 %dl, 1
        %v = sdiv i7 %a, %dv
        %lt = icmp slt i7 %a, %b
        %q32 = sext i7 %q to i32
        %v32 = sext i7 %v to i32
        %lt32 = zext i1 %lt to i32
        %r1 = mul i32 %q32, 10007
        %r2 = mul i32 %v32, 101
        %s1 = add i32 %r1, %r2
        %r = add i32 %s1, %lt32
        ret i32 %r""",
    # Around a loop, and a switch on one.
    control = """
        %a = trunc i8 %x to i7
        %n = trunc i8 %y to i3
        br label %loop
    loop:
        %i = phi i3 [ 0, %entry ], [ %i1, %loop ]
        %acc = phi i7 [ %a, %entry ], [ %acc2, %loop ]
        %acc2 = mul i7 %acc, 3
        %i1 = add i3 %i, 1
        %done = icmp eq i3 %i1, %n
        br i1 %done, label %exit, label %loop
    exit:
        switch i7 %acc2, label %other [ i7 3, label %three
                                        i7 -10, label %minus ]
    three:
        br label %join
    minus:
        br label %join
    other:
        br label %join
    join:
        %k = phi i32 [ 1, %three ], [ 2, %minus ], [ 0, %other ]
        %z = zext i7 %acc2 to i32
        %z4 = shl i32 %z, 4
        %r = or i32 %z4, %k
        ret i32 %r""",
)

for (name, body) in pairs(ODD_INTEGER_BODIES)
    ir = "define i32 @f(i8 %x, i8 %y) alwaysinline {\nentry:\n" * body * "\n}\n"
    @eval $(Symbol("odd_", name))(x::UInt8, y::UInt8) = Base.llvmcall(($ir, "f"), Int32, Tuple{UInt8, UInt8}, x, y)
end

KA.@kernel function odd_integers!(bitmap, wrapping, signed, control, @Const(x), @Const(y))
    i = KA.@index(Global)
    @inbounds begin
        bitmap[i] = odd_bitmap(x[i], y[i])
        wrapping[i] = odd_wrapping(x[i], y[i])
        signed[i] = odd_signed(x[i], y[i])
        control[i] = odd_control(x[i], y[i])
    end
end

@testset "odd-width integers compute as on the host" begin
    xs = vec([x for x in 0x00:0xff, _ in 0x00:0xff])
    ys = vec([y for _ in 0x00:0xff, y in 0x00:0xff])
    n = length(xs)
    out = [Mantle.devicearray(TESTBACKEND, zeros(Int32, n)) for _ in 1:4]
    odd_integers!(TESTBACKEND)(out..., Mantle.devicearray(TESTBACKEND, xs), Mantle.devicearray(TESTBACKEND, ys);
                               ndrange = n)
    @test Array(out[1]) == odd_bitmap.(xs, ys)
    @test Array(out[2]) == odd_wrapping.(xs, ys)
    @test Array(out[3]) == odd_signed.(xs, ys)
    @test Array(out[4]) == odd_control.(xs, ys)
end

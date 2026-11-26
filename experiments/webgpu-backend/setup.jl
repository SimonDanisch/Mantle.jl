# The WebGPU test device on this desktop: `Mantle.webgpudevice()` opens its own
# Electron window, which needs the desktop session's X display. Electron inherits
# FONTCONFIG_FILE: the system's conf.d syntax is newer than the bundled fontconfig.
using Mantle, ElectronCall, Bonito
import KernelAbstractions as KA
using Test

if Sys.islinux() && !haskey(ENV, "DISPLAY")
    ENV["DISPLAY"] = ":1"
    ENV["XAUTHORITY"] = joinpath("/run/user/1000", first(filter(startswith("xauth_"), readdir("/run/user/1000"))))
end
Sys.islinux() && (ENV["FONTCONFIG_FILE"] = joinpath(@__DIR__, "fonts.conf"))

# Two heaps, so every kernel goes through the heap switch (`heaps.jl` puts arrays in both).
const dev = Mantle.webgpudevice(; memory = 3 * 2^30)
println(Mantle.devicename(dev))

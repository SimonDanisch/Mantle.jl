"""
    foreachbackend(path)

Run a portable test file once for every backend this machine can use.

Each run gets a fresh module, so the file's own `const`s are defined once per
backend instead of redefined; the file reads `Main.MANTLE_TEST_BACKEND`.

This exists because the files it is called on hardcoded `VulkanAPI()` 57 times
between them — `test_window.jl` 35 times — so windows, surfaces, resize,
presentation, arena recording and device ranges were only ever checked against
one backend. Both bugs a person found by looking at a window in September 2026
were in that blind spot.
"""
function foreachbackend(path)
    bes = Mantle.eachbackend()
    isempty(bes) && @info "no usable backend, skipping" file = basename(path)
    for be in bes
        @eval Main MANTLE_TEST_BACKEND = $be
        @info "portable tests" file = basename(path) backend = nameof(typeof(be))
        @eval Main module $(gensym(:PortableRun))
            using Test
            include($path)
        end
    end
end

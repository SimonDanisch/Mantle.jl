# The backend a device test runs on. `runtests.jl` includes each such file once per
# usable backend (`foreachbackend`), which sets `Main.MANTLE_TEST_BACKEND`; a bare
# `include` from the REPL gets the default one. Nothing in a test names a backend.
import Mantle
const TESTBACKEND = isdefined(Main, :MANTLE_TEST_BACKEND) ? Main.MANTLE_TEST_BACKEND :
                                                           Mantle.defaultbackend()

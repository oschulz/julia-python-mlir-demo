import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.18"  # cap GPU memory before importing jax

import re
import numpy as np
import jax
import jax.numpy as jnp
import jaxlib.mlir.ir as ir
from jax._src.interpreters import mlir as jmlir   # make_ir_context: registers func + stablehlo
from jax._src.lib import xla_client as xc
from enzyme_ad.jax import hlo_call

# Reactant's optimized @code_hlo output can attach its own dialect attribute to FFT/slice ops
# as a real/imaginary analysis hint, e.g.:
#   stablehlo.fft ... {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed NOTGUARANTEED>]}
# Reactant.Ops.hlo_call (Julia) parses it fine (its own registered dialect); enzyme_ad.jax's
# MLIR context doesn't know the `enzymexla` custom attribute VALUE type and fails to parse the
# module at all ("unregistered dialect"). It's a pure, self-contained analysis hint (doesn't
# change the op's semantics), so stripping the clause is safe. NOTE: `@code_hlo optimize=false`
# is NOT an alternative fix here — it avoids this attribute but unwinds FFT lowering into an
# `enzyme.batch` op instead, a whole extra dialect enzyme_ad.jax's parser doesn't know at all.
_STRIP_ENZYMEXLA_HINT_RE = re.compile(
    r"\s*\{enzymexla\.complex_is_purely_(?:real|imaginary) = \[#enzymexla<guaranteed [A-Z]+>\]\}"
)


def save_mlir(path, fn, *args):
    """Lower `fn(*args)` to StableHLO/MLIR text and write it to `path`.
    Counterpart of load_mlir in common.jl."""
    text = jax.jit(fn).lower(*args).as_text()
    with open(path, "w") as f:
        f.write(text)


def load_mlir(path):
    """Eager PJRT load: RUN only. Returns a function that runs the module's @main on
    the default device. Opaque executable, outside JAX — not differentiable."""
    ctx = jmlir.make_ir_context()
    with ctx, open(path) as f:
        # Parse text -> live MLIR module object. compile_and_load needs the module
        # object (the nanobind binding rejects a raw str/bytes); a bare ir.Context
        # only knows builtin+stablehlo, so make_ir_context also registers `func`.
        module = ir.Module.parse(f.read(), ctx)
    device = jax.devices()[0]
    loaded = device.client.compile_and_load(
        module,
        executable_devices=xc.DeviceList((device,)),
        compile_options=xc.CompileOptions(),
    )

    def run(*arrays):
        outs = loaded.execute([jax.device_put(a) for a in arrays])
        return [np.asarray(o) for o in outs]

    return run


def load_mlir_enzyme(path):
    """enzyme-ad load: returns a *traceable* JAX function that calls the module's @main
    via enzyme_ad.jax.hlo_call. Needs jit to run (like Reactant.Ops.hlo_call on the
    Julia side) and is differentiable with jax.grad (Enzyme-on-MLIR). hlo_call returns
    a tuple of outputs.

    Strips Reactant's `enzymexla.complex_is_purely_*` FFT analysis hint (see
    _STRIP_ENZYMEXLA_HINT_RE above) so modules containing FFT ops parse at all."""
    code = _STRIP_ENZYMEXLA_HINT_RE.sub("", open(path).read())

    def f(*arrays):
        return hlo_call(*arrays, source=code)

    return f

# Julia/Python MLIR exchange and Enzyme-based AD demos

## Setup

[Install Julia](https://github.com/oschulz/julia-setup).

Edit `CondaPkg.toml` to control which GPU environment to use (e.g.
system-wide CUDA, TensorFlow/JAX-provided CUDA, CPU only, etc.).

Make this directory your Julia project environment. Then run

```julia
julia> import Pkg; Pkg.instantiate()
julia> import CondaPkg; CondaPkg.resolve()
```

CondaPkg (with default settings) will set up a Pixi environment in the
sub-directory `.CondaPkg`. Make sure this is the Python installation you'll
use.

In VS-Code, for example set

```json
"python.defaultInterpreterPath": "${workspaceFolder}/.CondaPkg/.pixi/envs/default/bin/python"
```

in your workspace settings.

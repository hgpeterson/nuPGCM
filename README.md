# 𝜈PGCM

[![GNU GPLv3 License](https://img.shields.io/badge/License-GNU%20GPL-blue)](https://www.gnu.org/licenses/gpl-3.0.en.html)

This repository hosts a planetary geostrophic circulation model (𝜈PGCM) that solves the 3D planetary geostrophic (PG) equations using finite elements.
The model is written in [Julia](https://julialang.org/) and can run on a GPU.

*[See [PGModels1Dand2D](https://github.com/hgpeterson/PGModels1Dand2D) for 1D and 2D PG models from [Peterson & Callies (2022)](https://doi.org/10.1175/JPO-D-21-0173.1) and [Peterson & Callies (2023)](https://doi.org/10.1175/JPO-D-22-0082.1)]*

## Installation

If you don't have Julia installed already, [download it here](https://julialang.org/downloads/) (currently, the model has only been verified with Julia version 1.10).

The model is still in active development and has not yet been added tot he official Julia package registry.
To try it out, manually download the repository:
```
git clone git@github.com:hgpeterson/nuPGCM.git
cd nuPGCM
```
To install the required dependencies, launch Julia, type `]` to start the Pkg REPL, and type
```
(@v1.10) pkg> activate nuPGCM
Activating project at `~/nuPGCM`

(nuPGCM) pkg> instantiate
  No Changes to `~/nuPGCM/Project.toml`
  No Changes to `~/nuPGCM/Manifest.toml`

(nuPGCM) pkg> precompile
```

## Examples

To run the example:
```
cd examples
julia --project=../run.jl
```
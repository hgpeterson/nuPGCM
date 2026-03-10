# $\nu$PGCM

Interested in simulating the large-scale ocean circulation over long timescales? 
Then look no further!
Some highlights of the planetary geostrophic circulation model ($\nu$PGCM):
- Capable of simulating arbitrary domains thanks to the finite elements formulation
- Written in [Julia](https://julialang.org/) 
- Runs on a GPU

## Installation

If you don't have Julia installed already, [download it here](https://julialang.org/downloads/) (currently, the package environment is built with Julia version 1.10).

The model is still in active development and has not yet been added to the official Julia package registry.
To try it out, manually download the repository:
```
git clone git@github.com:hgpeterson/nuPGCM.git
cd nuPGCM
julia --project
```
To install the required dependencies, type `]` to start the Pkg REPL and type
```
(nuPGCM) pkg> instantiate
```
To run the example, you can backspace out of the Pkg REPL and run 
```
julia> include("examples/run.jl")
```
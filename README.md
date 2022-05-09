# νPGCM 

This repository hosts Julia code for obtaining numerical solutions to the planetary geostrophic equations (with momentum fluxes parameterized by Fickian diffusion) for general topography.
For a similar model using Rayleigh drag, see [https://github.com/joernc/pgcm](https://github.com/joernc/pgcm).
The model is described in Peterson & Callies (2022): Rapid spin up and spin down of flow along slopes, https://doi.org/10.1175/JPO-D-21-0173.1.

The 1D, 2D, and 3D PG models are in `src/`. 
See `examples/` for some examples for running them. 
`non_pg_models/` also has some code a non-PG 1D model (both dimensional and non-dimensional).

Please don't hesitate to contact me if you have any questions!
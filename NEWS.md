# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Optional separate order for buoyancy
- Optional order-1 timestepping scheme
- Optimization for inversion matrix rebuild when `EddyParameterization` is used
- Some more time printing
- Support for Julia v1.11 and v1.12
- Improvements to `estimated time remaining` info
- `Base.show()` and `Base.summary()` implementations for more types

### Fixed

- Optimized how Dirichlet boundary conditions are handled for advection RHS builds
- Fixed inversion-only `Model` initialization error

### Changed

- Switched representation of velocity from three scalar FE fields to a `VectorValue`d field

## [0.5.0] - 2025-11-25

### Added

- Surface flux boundary condition for buoyancy
- Parameterization for convection (off by default)
- Parameterization for eddies (off by default)
- New `Forcings` type
- Prettier printing of various types
- A few more info/warning messages and timings

### Fixed

- Fixed definition of wind stress boundary condition

### Changed

- Split diffusivity `\kappa` into vertical and horizontal components `\kappa\_v` and `\kappa\_h`
- Lots of boilerplate initialization code (building matrices, re-ordering DOFs, etc.) moved to `InversionToolkit` and `EvolutionToolkit` constructors
- More general treatment of Dirichlet boundary conditions in `Spaces` constructor to allow for more general problems and mesh tags
- Rename `Model` contructors
- Added `f` and `H` to `Parameters`
- Added `surface_tags` keyword argument to `Mesh` constructor

## [0.4.2] - 2025-09-12

### Added

- Allow for non-constant ν [#7](https://github.com/hgpeterson/nuPGCM/issues/7)
- Added an updated channel+basin mesh script in `meshes/`

### Fixes

- Removed erroneous factor of `\alpha` in wind stress surface integral

## [0.4.1] - 2025-09-08

### Fixes

- Fixed Missing `.msh` files for `test/runtests.jl` [#3](https://github.com/hgpeterson/nuPGCM/issues/3)
- Fixed Missing `.msh` file and `matrices` folder for example [#6](https://github.com/hgpeterson/nuPGCM/issues/6)

## [0.4.0] - 2025-08-26

### Added

- Allow for general Dirichlet boundary condition on buoyancy
- Add wind stress forcing

### Fixed

- Fixed Strang splitting time stepping scheme (commit [1d641ea](https://github.com/hgpeterson/nuPGCM/commit/1d641ea4449a46dcc0190dda4faf52bce70d44a4))
- Closed issue [#4](https://github.com/hgpeterson/nuPGCM/issues/4) 
- Closed issue [#5](https://github.com/hgpeterson/nuPGCM/issues/5)

## [0.3.0] - 2025-04-22

### Changed

- Move to isotropic mesh formulation PR[#2](https://github.com/hgpeterson/nuPGCM/pull/2)

## [0.2.0] - 2025-04-14

### Added

- Some end-to-end tests

### Changed

- Refactor package to export `Model` struct for cleaner initialization and interaction user to more easily initialize and interact with

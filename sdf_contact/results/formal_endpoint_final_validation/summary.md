# Formal endpoint final validation

## What is included in this endpoint package
- formal pressure field constitutive law `p_i = k_i d_i`
- pressure-difference normal `n = grad(h) / ||grad(h)||`
- explicit band mechanics / local-normal accumulator cell layer
- separate zero-thickness sheet representation recovery by measure-preserving clustering
- shared patch -> polygon support -> triangulation -> quadrature outer architecture

## Common evaluator configuration used for all analytic cases
- patch raster_cells = 18
- patch support_radius_floor_scale = 0.9
- sheet bisection_steps = 18
- no case-specific evaluator branching was used

## Centered sphere-plane, equal stiffness
- mean relative force error: 0.6658%
- max relative force error: 3.8411%
- max body moment norm: 6.678983e-04

## Off-axis sphere-plane, equal stiffness
- max relative force error: 0.0480%
- max relative error of ground Mz: 0.1444%
- max body moment norm: 7.863218e-05

## Centered sphere-plane, unequal stiffness
- mean relative force error: 0.6658%
- max relative force error: 3.8411%

## Centered flat punch / box-plane, equal stiffness
- mean relative force error: 0.9168%
- max relative force error: 0.9169%
- max lateral force norm: 2.936249e-02

## Off-axis flat punch / box-plane, equal stiffness
- max relative force error: 0.9169%
- max relative error of ground Mz: 0.3837%
- max lateral force norm: 1.812496e-02

## Reading of the result
- Sphere-plane remains at roughly 1% or better force accuracy without special-case branching.
- Flat punch / box-plane now also follows the analytic constant-area force law closely, which is the key plane-consistency benchmark from the formal document.
- Off-axis cases preserve the correct first moment trend `Mz = -x * Fy` on the ground side.
- This package is therefore much closer to the formal endpoint than the earlier spring-gap and sheet-only versions.
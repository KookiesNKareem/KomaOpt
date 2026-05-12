# KomaOpt

Gradient-based RF pulse design on top of [KomaMRI](https://github.com/JuliaHealth/KomaMRI.jl). Reverse-mode AD through the Bloch simulator runs on three backends: CPU via Enzyme, GPU/XLA via [Reactant](https://github.com/EnzymeAD/Reactant.jl), and native CUDA via KernelAbstractions + Enzyme.

## Setup

Requires Julia ≥ 1.10 ([juliaup](https://github.com/JuliaLang/juliaup) recommended):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

First instantiation downloads the LLVM/Reactant artifacts and precompiles a few hundred packages — expect 10–15 min. Subsequent loads are fast.

CUDA.jl is **not** a project dependency. It is lazily installed and loaded the first time you run with `backend=:cuda`.

## Scripts

Run either script with the project environment active:

```bash
julia --project=. simple_cpu.jl
julia --project=. 2d_excitation.jl
```

- **[`simple_cpu.jl`](simple_cpu.jl)** — 1-D slice-selective RF design on the CPU. Optimizes a complex RF pulse against a target slice profile using `KomaMRICore` + Enzyme. Minimal end-to-end example.
- **[`2d_excitation.jl`](2d_excitation.jl)** — 2-D selective excitation with a spiral k-space trajectory. Optimizes a complex RF pulse to produce a target 2-D image. Choose a backend via `main(backend=…)`:
  - `:cpu` — Reactant CPU
  - `:reactant_gpu` — Reactant CUDA
  - `:cuda` — native KernelAbstractions + Enzyme CUDA

Shared utilities (control→timeline interpolation, CSR gradient gather, Reactant + KA contexts) live in [`utils.jl`](utils.jl).

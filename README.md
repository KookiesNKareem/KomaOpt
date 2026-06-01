# KomaOpt

Gradient-based RF pulse design on top of [KomaMRI](https://github.com/JuliaHealth/KomaMRI.jl). Reverse-mode AD through the Bloch simulator runs on three backends: CPU via Enzyme, GPU/XLA via [Reactant](https://github.com/EnzymeAD/Reactant.jl), and native CUDA via KernelAbstractions + Enzyme.

## Setup

Requires Julia ≥ 1.10 ([juliaup](https://github.com/JuliaLang/juliaup) recommended):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

First instantiation downloads the LLVM/Reactant artifacts and precompiles a few hundred packages — expect 10–15 min. Subsequent loads are fast.

CUDA.jl is **not** a project dependency. It is lazily installed and loaded the first time you run with `backend=:cuda` (or its equivalent `BACKEND` const in the other scripts).

## Scripts

Run any script with the project environment active:

```bash
julia --project=. simple_cpu.jl
julia --project=. 2d_excitation.jl
julia --project=. fatsat.jl
julia --project=. off_resonance.jl
```

- **[`simple_cpu.jl`](simple_cpu.jl)** — 1-D slice-selective RF design on the CPU. Optimizes a complex RF pulse against a target slice profile using `KomaMRICore` + Enzyme. Minimal end-to-end example.
- **[`2d_excitation.jl`](2d_excitation.jl)** — 2-D selective excitation with a spiral k-space trajectory. Optimizes a complex RF pulse to produce a target 2-D image. Select a backend via `main(backend=…)`.
- **[`fatsat.jl`](fatsat.jl)** — Fat-saturation RF pulse design (T1-aware). Optimizes a frequency-selective pulse against a Butterworth or Gaussian target Mz profile, with a post-pulse delay accounted for. Also runs a naive (T1=∞) optimization and a Gaussian reference for the comparison figure. Backend via the `BACKEND` const at the top.
- **[`off_resonance.jl`](off_resonance.jl)** — 2-D excitation with B0 inhomogeneity. Loads a B0 map (DICOM, a Hz-file image/JLD2, or a synthetic Gaussian bump), then runs two optimizations — a B0-ignorant one and a B0-aware one — and forward-evaluates both against the true field. Backend via the `BACKEND` const at the top.

The backend-selectable scripts default to `:cpu`, which is Reactant CPU. Their backend menu is:

- `:cpu` — Reactant CPU
- `:reactant_gpu` — Reactant CUDA
- `:cuda` — native KernelAbstractions + Enzyme CUDA

Shared utilities (control→timeline interpolation, CSR gradient gather, Reactant + KA contexts, Bloch kernels) live in [`utils.jl`](utils.jl). Pulseq sequence generators called by `fatsat.jl` and `off_resonance.jl` live in [`pulseq_utils/`](pulseq_utils/).

## Outputs

Each script writes its optimized pulse(s) and figures to `pulses/<name>/`:

```
pulses/simple_cpu/        # simple_cpu.png, simple_cpu.jld2
pulses/<target_stem>/     # from 2d_excitation.jl (e.g. pulses/stanford_logo/)
pulses/fatsat/            # fatsat_figure.png, fatsat_comparison.png, *.jld2, *.seq
pulses/off_resonance/     # results.png, b0_ignorant.{jld2,seq}, b0_aware.{jld2,seq}
```

Each `.jld2` bundle contains `B1`, `Gx`, `Gy`, `T`, and the full `seq`. `.jld2` and `.seq` outputs are gitignored.

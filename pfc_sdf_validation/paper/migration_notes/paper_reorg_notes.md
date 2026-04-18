# Paper Reorganization Notes

## Scope

This note records how the submission workspace under `paper/` was assembled from the existing repository without modifying the original project layout.

## Main File Mapping

| Original path | New path | Reason |
|---|---|---|
| `pfc_sdf_dynamics_paper_sprint1.tex` | `sections/00_abstract.tex` through `sections/08_conclusion.tex` | Main body source was the most direct starting point for the submission split. |
| `pfc_sdf_dynamics_paper_sprint3_appendix.tex` | `sections/09_appendix.tex` | This file contains the most complete appendix material. |
| `pfc_sdf_dynamics_paper_sprint1.tex` | `abstract.txt` | Plain-text abstract snapshot for quick reuse in forms and submission portals. |
| n/a | `main.tex` | New internal-only entrypoint with normalized section order and internal paths. |
| n/a | `main_anonymous.tex` | Anonymous-ready entrypoint without author metadata. |
| n/a | `sections/02_related_work.tex` | Placeholder added because no integrated related-work section existed in the source manuscript. |
| n/a | `sections/06_external_comparison.tex` | Added to isolate the 6-DoF reference-comparison material into a dedicated section. |
| n/a | `bib/main.bib` | Placeholder added because no bibliography file was present. |

## Figures Migrated

| Original path | New path |
|---|---|
| `results/figures/flat_force_time.pdf` | `figures/flat_force_time.pdf` |
| `results/figures/sphere_force_time.pdf` | `figures/sphere_force_time.pdf` |
| `results/figures/flat_energy_time.pdf` | `figures/flat_energy_time.pdf` |
| `results/figures/native_band_force_time.pdf` | `figures/native_band_force_time.pdf` |
| `results/figures/native_band_active_measure_time.pdf` | `figures/native_band_active_measure_time.pdf` |
| `results/figures/native_band_controller_statistics.pdf` | `figures/native_band_controller_statistics.pdf` |

These were copied because they are directly referenced by the reorganized paper body.

## Tables Migrated

| Original path | New path |
|---|---|
| `results/tables/main_ablation.tex` | `tables/main_ablation.tex` |
| `results/tables/efficiency_continuity.tex` | `tables/efficiency_continuity.tex` |
| `results/tables/efficiency_traction.tex` | `tables/efficiency_traction.tex` |
| `complex6dof_ablation.tex` | `tables/complex6dof_ablation.tex` |
| `complex6dof_efficiency.tex` | `tables/complex6dof_efficiency.tex` |
| `complex6dof_reference_comparison.tex` | `tables/complex6dof_reference_comparison.tex` |
| `results/tables/eccentric_flat.tex` | `tables/eccentric_flat.tex` |
| `results/tables/multipatch_dynamic.tex` | `tables/multipatch_dynamic.tex` |
| `results/tables/controller_statistics.tex` | `tables/controller_statistics.tex` |
| `results/tables/reference_comparison.tex` | `tables/reference_comparison.tex` |
| `results/tables/sensitivity.tex` | `tables/sensitivity.tex` |

Root-level `complex6dof_*.tex` files were kept because the current manuscript refers to those table labels and there was no newer table version under `results/tables/`.

## Supporting Notes Migrated

| Original path | New path | Reason |
|---|---|---|
| `results/tables/appendix_plan.md` | `migration_notes/appendix_plan.md` | Records prior appendix planning context. |
| `results/tables/step1_2_6dof_summary.md` | `migration_notes/step1_2_6dof_summary.md` | Keeps 6-DoF rollout context. |
| `results/tables/step3_6dof_summary.md` | `migration_notes/step3_6dof_summary.md` | Keeps 6-DoF benchmark rollout context. |
| `pfc_sdf_dynamics_paper_sprint1.tex` | `migration_notes/source_main_sprint1.tex` | Verbatim source snapshot for diff/reference. |
| `pfc_sdf_dynamics_paper_sprint3_appendix.tex` | `migration_notes/source_appendix_sprint3.tex` | Verbatim appendix snapshot for diff/reference. |

The cell-centered native-band statement was preserved directly inside the reorganized manuscript rather than split into a separate note because the original repository only carried that wording inside the LaTeX sources.

## What Was Migrated Without Rewriting

- most Chinese manuscript paragraphs were copied structurally and only lightly cleaned
- table bodies were copied as-is
- figure filenames were kept stable where they were already submission-friendly
- appendix material was preserved as appendix material rather than merged into the body

## What Still Needs Manual Work

- full English rewrite
- related work
- bibliography
- official venue template migration
- anonymous polishing
- final supplementary package selection
- final cover-letter / dual-track packaging

## Candidate Files Not Included

| Candidate | Why it was not included |
|---|---|
| `pfc_sdf_dynamics_paper_sprint1.pdf` | Old compiled output, not a source artifact. |
| `pfc_sdf_dynamics_paper_sprint3_appendix.pdf` | Old compiled output, not a source artifact. |
| `*.aux`, `*.log`, `*.out`, `*.toc`, `*.fdb_latexmk`, `*.fls`, `*.xdv` | Compile byproducts only. |
| root `*.csv` files and `results/tables/*.csv` | Data artifacts, not submission-facing manuscript sources. |
| `results/tables/*.md` other than the migration notes copied above | Mostly raw summaries, planning notes, or intermediate reporting artifacts. |
| `paper_ablation_table.md` | Scratch-style markdown, superseded by `tables/main_ablation.tex`. |
| `results/figures/complex*.pdf` and `results/figures/complex6dof_*.pdf` | Not directly referenced by the current reorganized manuscript or appendix. |
| `experiments/`, `configs/`, `src/`, `tests/` | Required for reproducibility, but intentionally left outside the clean submission workspace. |

## Path Normalization

- All manuscript `\input{}` references now resolve inside `paper/`.
- All figure references now resolve through `paper/figures/`.
- `main.tex` and `main_anonymous.tex` do not cross-reference `../results`, `../src`, or other repository-root paths.

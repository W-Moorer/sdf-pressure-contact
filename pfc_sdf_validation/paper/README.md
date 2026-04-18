# Paper Workspace

Target venue:

- SIGGRAPH Asia 2026 Technical Papers
- dual-track submission
- conference-track intent

This `paper/` directory is a clean submission workspace assembled from the current `pfc_sdf_validation/` repository. The reorganization is limited to structure cleanup, path normalization, and migration of paper-relevant assets. It does not modify the original source, tests, configs, or results directories.

## Directory Structure

```text
paper/
├── README.md
├── Makefile
├── latexmkrc
├── main.tex
├── main_anonymous.tex
├── abstract.txt
├── sections/
├── tables/
├── figures/
├── bib/
├── supplementary/
├── cover/
└── migration_notes/
```

## Source Origins

Main sources used for the reorganization:

- `pfc_sdf_dynamics_paper_sprint1.tex` -> source for the abstract, introduction, method, setup, main results, discussion, and conclusion sections
- `pfc_sdf_dynamics_paper_sprint3_appendix.tex` -> source for the appendix body and supplementary comparison tables
- `results/figures/*.pdf` -> main figure assets now copied into `paper/figures/`
- `results/tables/*.tex` and root-level `complex6dof_*.tex` -> table assets now copied into `paper/tables/`
- `results/tables/appendix_plan.md`, `results/tables/step1_2_6dof_summary.md`, `results/tables/step3_6dof_summary.md` -> retained under `paper/migration_notes/` as supporting migration context

Verbatim source snapshots are kept in:

- `migration_notes/source_main_sprint1.tex`
- `migration_notes/source_appendix_sprint3.tex`

## Current State

What is already done:

- paper-facing directory structure created
- all currently referenced figures copied into `paper/figures/`
- all currently referenced LaTeX tables copied into `paper/tables/`
- new `main.tex` and `main_anonymous.tex` entrypoints now reference only files inside `paper/`
- section files split under `paper/sections/`
- the cell-centered native-band wording is preserved in the manuscript

What is only structural for now:

- the paper still uses `ctexart` instead of the official SIGGRAPH Asia 2026 template
- `02_related_work.tex` is a placeholder skeleton
- `bib/main.bib` is a placeholder
- `main.tex` author metadata is still TODO
- `main_anonymous.tex` still needs venue-grade anonymous polishing
- `supplementary/` and `cover/` are placeholders only

## TODO

- rewrite the manuscript into final submission-grade English
- replace the temporary `ctexart` wrapper with the official SIGGRAPH Asia 2026 paper template
- complete `sections/02_related_work.tex`
- populate `bib/main.bib` and wire up citations
- verify whether `reference_comparison.tex` should remain appendix-only or be partially promoted into the main paper
- finish anonymous polishing for `main_anonymous.tex`
- decide the final supplementary package contents and add them under `supplementary/`
- prepare submission cover material under `cover/`

## Minimal Compile

Assuming `xelatex` and `latexmk` are installed:

```powershell
cd pfc_sdf_validation/paper
make pdf
make anonymous
```

Fallback direct commands:

```powershell
latexmk -r latexmkrc -pdf main.tex
latexmk -r latexmkrc -pdf main_anonymous.tex
```

Outputs are configured to land in `paper/build/`.

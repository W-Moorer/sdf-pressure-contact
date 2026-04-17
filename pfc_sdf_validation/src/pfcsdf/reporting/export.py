from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def _default_float_formatter(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    magnitude = abs(value)
    if magnitude == 0.0:
        return "0"
    if magnitude >= 1000 or magnitude < 1e-3:
        return f"{value:.3e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_df(df: pd.DataFrame, float_formatter: Callable[[float], str] | None = None) -> pd.DataFrame:
    fmt = _default_float_formatter if float_formatter is None else float_formatter
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]) or pd.api.types.is_integer_dtype(out[col]):
            out[col] = [fmt(float(v)) if pd.notna(v) else "nan" for v in out[col]]
    return out


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_markdown_table(df: pd.DataFrame, path: str | Path, *, title: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _format_df(df).to_markdown(index=False)
    text = body if title is None else f"# {title}\n\n{body}\n"
    path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    return path


def write_latex_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    caption: str | None = None,
    label: str | None = None,
    column_format: str | None = None,
    float_formatter: Callable[[float], str] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted = _format_df(df, float_formatter=float_formatter)
    latex = formatted.to_latex(index=False, escape=False, caption=caption, label=label, column_format=column_format)
    latex = latex.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}", 1)
    latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    path.write_text(latex, encoding="utf-8")
    return path

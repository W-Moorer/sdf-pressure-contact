#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-contained verification of the main SDF-pressure-contact chain
with built-in plotting.

This script verifies:
    SDF -> depth -> pressure fields -> equal-pressure sheet h=0
    -> narrow-band / coarea volume integral -> force and moment

It saves:
    - verify_sdf_pressure_contact_chain_results.csv
    - verify_sdf_pressure_contact_chain_convergence.png
    - verify_sdf_pressure_contact_chain_force_vs_exact.png
    - verify_sdf_pressure_contact_chain_sheet_vs_exact.png

Usage:
    python verify_sdf_pressure_contact_chain_with_plots.py

Notes:
    1) The script uses matplotlib's Agg backend, so it works on headless servers.
    2) It saves images to the current working directory by default.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def delta_cosine(s: np.ndarray, eta: float) -> np.ndarray:
    """Normalized compactly supported delta approximation."""
    a = np.abs(s)
    out = np.zeros_like(s, dtype=float)
    mask = a <= eta
    out[mask] = 0.5 / eta * (1.0 + np.cos(np.pi * s[mask] / eta))
    return out


def analytic_planar_contact(
    kA: float,
    kB: float,
    delta: float,
    Lx: float,
    Lz: float,
    x0: float = 0.0,
    z0: float = 0.0,
) -> dict:
    """
    Analytic solution for two opposing half-spaces with a finite rectangular patch.

    A: surface y = 0,    interior y < 0
    B: surface y = -delta, interior y > -delta
    Linear pressure fields:
        pA = kA * dA
        pB = kB * dB
    On the equal-pressure sheet:
        dA + dB = delta
        kA dA = kB dB
    """
    area = Lx * Lz
    y_star = -kB * delta / (kA + kB)
    p_star = (kA * kB / (kA + kB)) * delta
    Fy = area * p_star
    Mx = -z0 * Fy
    My = 0.0
    Mz = x0 * Fy
    return {
        "area": area,
        "y_star": y_star,
        "p_star": p_star,
        "Fy": Fy,
        "Mx": Mx,
        "My": My,
        "Mz": Mz,
    }


def volume_integral_planar_contact(
    Nx: int,
    Ny: int,
    Nz: int,
    *,
    kA: float,
    kB: float,
    delta: float,
    Lx: float,
    Lz: float,
    x0: float = 0.0,
    z0: float = 0.0,
    pad_x: float = 0.2,
    pad_z: float = 0.2,
    pad_y: float = 0.15,
    eta_factor: float = 2.0,
) -> dict:
    """
    Numerical verification of the chain:
        SDF -> depth -> pressure fields -> equal-pressure sheet
            -> coarea/narrow-band volume integral -> force & moment
    """
    x_min = x0 - 0.5 * Lx - pad_x
    x_max = x0 + 0.5 * Lx + pad_x
    z_min = z0 - 0.5 * Lz - pad_z
    z_max = z0 + 0.5 * Lz + pad_z
    y_min = -delta - pad_y
    y_max = pad_y

    xs = np.linspace(x_min, x_max, Nx, endpoint=False) + (x_max - x_min) / Nx / 2.0
    ys = np.linspace(y_min, y_max, Ny, endpoint=False) + (y_max - y_min) / Ny / 2.0
    zs = np.linspace(z_min, z_max, Nz, endpoint=False) + (z_max - z_min) / Nz / 2.0

    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    dz = (z_max - z_min) / Nz
    dV = dx * dy * dz

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # Opposing half-spaces
    phi_A = Y
    phi_B = -(Y + delta)

    dA = np.clip(-phi_A, 0.0, None)
    dB = np.clip(-phi_B, 0.0, None)

    pA = kA * dA
    pB = kB * dB

    h = pA - pB
    grad_h_norm = (kA + kB)

    overlap = (phi_A <= 0.0) & (phi_B <= 0.0)
    patch = (
        (np.abs(X - x0) <= 0.5 * Lx) &
        (np.abs(Z - z0) <= 0.5 * Lz)
    )

    # eta_y is a physical thickness in y; eta_h is the corresponding band width in h-space
    eta_y = eta_factor * dy
    eta_h = grad_h_norm * eta_y

    weight = delta_cosine(h, eta_h) * grad_h_norm * overlap * patch

    # In this planar test, the sheet traction is vertical.
    p_bar = 0.5 * (pA + pB)
    t_y = p_bar

    Fy = np.sum(t_y * weight) * dV
    Mx = np.sum((-Z * t_y) * weight) * dV
    Mz = np.sum((X * t_y) * weight) * dV

    area_est = np.sum(weight) * dV
    y_star_est = np.sum(Y * weight) * dV / area_est

    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "eta_y": eta_y,
        "eta_h": eta_h,
        "Fy": Fy,
        "Mx": Mx,
        "Mz": Mz,
        "area_est": area_est,
        "y_star_est": y_star_est,
    }


def build_convergence_table(case_name: str, resolutions: list[int], **kwargs) -> tuple[pd.DataFrame, dict]:
    ana = analytic_planar_contact(
        kwargs["kA"], kwargs["kB"], kwargs["delta"],
        kwargs["Lx"], kwargs["Lz"],
        kwargs.get("x0", 0.0), kwargs.get("z0", 0.0)
    )

    rows = []
    for N in resolutions:
        num = volume_integral_planar_contact(N, N, N, **kwargs)
        rows.append({
            "case": case_name,
            "N": N,
            "dy": num["dy"],
            "eta_y": num["eta_y"],
            "Fy_num": num["Fy"],
            "Fy_exact": ana["Fy"],
            "rel_err_Fy": abs(num["Fy"] - ana["Fy"]) / abs(ana["Fy"]),
            "Mz_num": num["Mz"],
            "Mz_exact": ana["Mz"],
            "rel_err_Mz": abs(num["Mz"] - ana["Mz"]) / abs(ana["Mz"]) if abs(ana["Mz"]) > 0 else np.nan,
            "y_star_num": num["y_star_est"],
            "y_star_exact": ana["y_star"],
            "abs_err_y_star": abs(num["y_star_est"] - ana["y_star"]),
            "area_num": num["area_est"],
            "area_exact": ana["area"],
            "rel_err_area": abs(num["area_est"] - ana["area"]) / abs(ana["area"]),
        })
    return pd.DataFrame(rows), ana


def make_plots(df_all: pd.DataFrame, out_dir: Path) -> None:
    # 1) Error convergence
    plt.figure(figsize=(7.0, 4.5))
    for case, grp in df_all.groupby("case"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["rel_err_Fy"], marker="o", label=f"{case}: force")
        plt.plot(grp["N"], grp["abs_err_y_star"], marker="s", label=f"{case}: sheet y")
    plt.yscale("log")
    plt.xlabel("grid resolution N = Nx = Ny = Nz")
    plt.ylabel("error")
    plt.title("Convergence of band-integral verification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "verify_sdf_pressure_contact_chain_convergence.png", dpi=180)
    plt.close()

    # 2) Numerical vs exact force
    plt.figure(figsize=(7.0, 4.5))
    for case, grp in df_all.groupby("case"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["Fy_num"], marker="o", label=f"{case}: numerical")
        plt.plot(grp["N"], grp["Fy_exact"], marker="x", label=f"{case}: exact")
    plt.xlabel("grid resolution N = Nx = Ny = Nz")
    plt.ylabel("Fy")
    plt.title("Numerical force vs exact force")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "verify_sdf_pressure_contact_chain_force_vs_exact.png", dpi=180)
    plt.close()

    # 3) Numerical vs exact sheet location
    plt.figure(figsize=(7.0, 4.5))
    for case, grp in df_all.groupby("case"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["y_star_num"], marker="o", label=f"{case}: numerical")
        plt.plot(grp["N"], grp["y_star_exact"], marker="x", label=f"{case}: exact")
    plt.xlabel("grid resolution N = Nx = Ny = Nz")
    plt.ylabel("equal-pressure sheet y*")
    plt.title("Recovered sheet location vs exact value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "verify_sdf_pressure_contact_chain_sheet_vs_exact.png", dpi=180)
    plt.close()


def main() -> None:
    out_dir = Path.cwd()
    figures_dir = out_dir / "result" / "figures"
    data_dir = out_dir / "result" / "data"

    df_sym, ana_sym = build_convergence_table(
        "symmetric_centered",
        resolutions=[24, 32, 48, 64, 96, 128],
        kA=40.0, kB=40.0, delta=0.10, Lx=1.20, Lz=0.80, x0=0.0, z0=0.0
    )

    df_asym, ana_asym = build_convergence_table(
        "asymmetric_offset",
        resolutions=[24, 32, 48, 64, 96, 128],
        kA=30.0, kB=90.0, delta=0.10, Lx=1.20, Lz=0.80, x0=0.35, z0=0.0
    )

    df_all = pd.concat([df_sym, df_asym], ignore_index=True)
    df_all.to_csv(data_dir / "verify_sdf_pressure_contact_chain_results.csv", index=False)
    make_plots(df_all, figures_dir)

    print("解析结果（对称居中 case）:")
    print(ana_sym)
    print("\n解析结果（非对称偏心 case）:")
    print(ana_asym)
    print("\n数值结果表已保存到:")
    print(data_dir / "verify_sdf_pressure_contact_chain_results.csv")
    print("\n图片已保存到:")
    print(figures_dir / "verify_sdf_pressure_contact_chain_convergence.png")
    print(figures_dir / "verify_sdf_pressure_contact_chain_force_vs_exact.png")
    print(figures_dir / "verify_sdf_pressure_contact_chain_sheet_vs_exact.png")


if __name__ == "__main__":
    main()

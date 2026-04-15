#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version A vs Version B on a truly 3D benchmark:
    rigid hemisphere pressing shallowly into the top face of a compliant cube.

Focus:
    shallow indentation accuracy

Version A:
    direct 3D voxel narrow-band integral in the cube volume

Version B:
    local-normal 3D accumulator on the exact spherical contact sheet
    (surface elements on the submerged spherical cap)

For this benchmark, the equal-pressure sheet is exactly the rigid sphere surface,
so the local-normal 1D band integral collapses to a sheet traction in the eta->0 limit:
    dF = -p(y) n dA

That makes this benchmark ideal for showing why the local-normal accumulator
is much more accurate for shallow indentation than a coarse voxel band integral.

Outputs:
    - hemisphere_cube_AB_shallow_results.csv
    - hemisphere_cube_AB_shallow_force_compare.png
    - hemisphere_cube_AB_error_vs_runtime.png
    - hemisphere_cube_AB_convergence.png
    - hemisphere_cube_AB_symmetry_residual.png
    - hemisphere_cube_AB_local_normal_geometry.png

Usage:
    python hemisphere_cube_AB_shallow_validation.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time


# -----------------------------
# Basic helpers
# -----------------------------
def delta_cosine(s: np.ndarray, eta: float) -> np.ndarray:
    a = np.abs(s)
    out = np.zeros_like(s, dtype=float)
    mask = a <= eta
    out[mask] = 0.5 / eta * (1.0 + np.cos(np.pi * s[mask] / eta))
    return out


def exact_cap_volume(R: float, delta: float) -> float:
    return np.pi * delta * delta * (R - delta / 3.0)


def exact_cap_surface_area(R: float, delta: float) -> float:
    return 2.0 * np.pi * R * delta


def exact_contact_radius(R: float, delta: float) -> float:
    return np.sqrt(max(0.0, 2.0 * R * delta - delta * delta))


def analytic_force(R: float, delta: float, k: float) -> float:
    return k * exact_cap_volume(R, delta)


# -----------------------------
# Version A: direct voxel band integral
# -----------------------------
def sphere_sdf_and_grad(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: tuple[float, float, float], R: float):
    cx, cy, cz = center
    DX = X - cx
    DY = Y - cy
    DZ = Z - cz
    RR = np.sqrt(DX * DX + DY * DY + DZ * DZ)
    phi = RR - R
    eps = 1e-15
    nx = DX / np.maximum(RR, eps)
    ny = DY / np.maximum(RR, eps)
    nz = DZ / np.maximum(RR, eps)
    return phi, nx, ny, nz


def version_A_direct_band(
    *,
    R: float,
    delta: float,
    k: float,
    cube_size: float,
    cube_height: float,
    N: int,
    eta_factor: float = 1.5,
):
    x_min, x_max = -cube_size / 2.0, cube_size / 2.0
    z_min, z_max = -cube_size / 2.0, cube_size / 2.0
    y_min, y_max = -cube_height, 0.0

    xs = np.linspace(x_min, x_max, N, endpoint=False) + (x_max - x_min) / N / 2.0
    ys = np.linspace(y_min, y_max, N, endpoint=False) + (y_max - y_min) / N / 2.0
    zs = np.linspace(z_min, z_max, N, endpoint=False) + (z_max - z_min) / N / 2.0

    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    dz = (z_max - z_min) / N
    dV = dx * dy * dz

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    center = (0.0, R - delta, 0.0)

    phi, nx, ny, nz = sphere_sdf_and_grad(X, Y, Z, center, R)

    p = k * np.clip(-Y, 0.0, None)

    h = max(dx, dy, dz)
    eta = eta_factor * h
    band = delta_cosine(phi, eta)

    tx = -p * nx
    ty = -p * ny
    tz = -p * nz

    Fx = np.sum(tx * band) * dV
    Fy = np.sum(ty * band) * dV
    Fz = np.sum(tz * band) * dV

    Mx = np.sum((Y * tz - Z * ty) * band) * dV
    My = np.sum((Z * tx - X * tz) * band) * dV
    Mz = np.sum((X * ty - Y * tx) * band) * dV

    return {
        "Fx": Fx, "Fy": Fy, "Fz": Fz,
        "Mx": Mx, "My": My, "Mz": Mz,
        "eta": eta,
    }


# -----------------------------
# Version B: local-normal 3D accumulator on spherical sheet elements
# -----------------------------
def version_B_local_normal_sheet(
    *,
    R: float,
    delta: float,
    k: float,
    Na: int,
    Nt: int,
):
    """
    Local-normal 3D accumulator.

    Surface parameterization of the submerged spherical cap:
        alpha in [0, alpha_max] measured from the bottom pole
        theta in [0, 2pi)

    Point:
        x = R sin(alpha) cos(theta)
        y = (R - delta) - R cos(alpha)
        z = R sin(alpha) sin(theta)

    Outward normal:
        n = (sin(alpha) cos(theta), -cos(alpha), sin(alpha) sin(theta))

    Local-normal accumulator in the eta->0 limit:
        dF = -p(y) n dA
    """
    alpha_max = np.arccos(1.0 - delta / R)
    if alpha_max <= 0.0:
        return {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0, "Mx": 0.0, "My": 0.0, "Mz": 0.0}

    dalpha = alpha_max / Na
    dtheta = 2.0 * np.pi / Nt

    alpha = (np.arange(Na) + 0.5) * dalpha
    theta = (np.arange(Nt) + 0.5) * dtheta

    A, T = np.meshgrid(alpha, theta, indexing="ij")

    cy = R - delta

    X = R * np.sin(A) * np.cos(T)
    Y = cy - R * np.cos(A)
    Z = R * np.sin(A) * np.sin(T)

    nx = np.sin(A) * np.cos(T)
    ny = -np.cos(A)
    nz = np.sin(A) * np.sin(T)

    p = k * np.clip(-Y, 0.0, None)
    dA = (R * R) * np.sin(A) * dalpha * dtheta

    dFx = -p * nx * dA
    dFy = -p * ny * dA
    dFz = -p * nz * dA

    Fx = np.sum(dFx)
    Fy = np.sum(dFy)
    Fz = np.sum(dFz)

    # moments about the origin
    Mx = np.sum(Y * dFz - Z * dFy)
    My = np.sum(Z * dFx - X * dFz)
    Mz = np.sum(X * dFy - Y * dFx)

    return {
        "Fx": Fx, "Fy": Fy, "Fz": Fz,
        "Mx": Mx, "My": My, "Mz": Mz,
    }


# -----------------------------
# Geometry demo
# -----------------------------
def make_geometry_demo(out_dir: Path, *, R: float, delta: float, Na: int, Nt: int):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    alpha_max = np.arccos(1.0 - delta / R)
    dalpha = alpha_max / Na
    dtheta = 2.0 * np.pi / Nt
    alpha = (np.arange(Na) + 0.5) * dalpha
    theta = (np.arange(Nt) + 0.5) * dtheta
    A, T = np.meshgrid(alpha, theta, indexing="ij")

    cy = R - delta
    X = R * np.sin(A) * np.cos(T)
    Y = cy - R * np.cos(A)
    Z = R * np.sin(A) * np.sin(T)

    # show sheet-element centers
    ax.scatter(X, Y, Z, s=6)

    # show a sparse set of local normals
    stride_a = max(1, Na // 8)
    stride_t = max(1, Nt // 16)
    Xq = X[::stride_a, ::stride_t]
    Yq = Y[::stride_a, ::stride_t]
    Zq = Z[::stride_a, ::stride_t]
    nq_x = np.sin(A[::stride_a, ::stride_t]) * np.cos(T[::stride_a, ::stride_t])
    nq_y = -np.cos(A[::stride_a, ::stride_t])
    nq_z = np.sin(A[::stride_a, ::stride_t]) * np.sin(T[::stride_a, ::stride_t])
    ax.quiver(Xq, Yq, Zq, nq_x, nq_y, nq_z, length=0.08, normalize=True)

    # exact contact circle on y=0
    a = exact_contact_radius(R, delta)
    t = np.linspace(0.0, 2.0 * np.pi, 300)
    ax.plot(a * np.cos(t), np.zeros_like(t), a * np.sin(t), linewidth=2.0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Version B: local sheet elements and local normals")
    ax.set_box_aspect((1, 0.6, 1))
    plt.tight_layout()
    plt.savefig(out_dir / "hemisphere_cube_AB_local_normal_geometry.png", dpi=180)
    plt.close()


# -----------------------------
# Main experiment
# -----------------------------
def main():
    out_dir = Path.cwd()
    figures_dir = out_dir / "result" / "figures"
    data_dir = out_dir / "result" / "data"

    R = 1.0
    k = 10.0

    # Keep the volume box modest: we focus on shallow indentation.
    cube_size = 1.6
    cube_height = 0.25

    shallow_deltas = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12])

    # fixed settings for shallow force-vs-penetration comparison
    N_A_fixed = 96
    Na_B_fixed = 24
    Nt_B_fixed = 96

    rows = []

    # 1) shallow penetration sweep
    for delta in shallow_deltas:
        Fy_exact = analytic_force(R, delta, k)

        t0 = time.perf_counter()
        Aout = version_A_direct_band(
            R=R, delta=delta, k=k,
            cube_size=cube_size, cube_height=cube_height,
            N=N_A_fixed, eta_factor=1.5
        )
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        Bout = version_B_local_normal_sheet(
            R=R, delta=delta, k=k,
            Na=Na_B_fixed, Nt=Nt_B_fixed
        )
        t3 = time.perf_counter()

        rows.append({
            "mode": "shallow_sweep",
            "method": "A_direct_band",
            "delta": delta,
            "resolution": N_A_fixed,
            "Fy_num": Aout["Fy"],
            "Fy_exact": Fy_exact,
            "rel_err_Fy": abs(Aout["Fy"] - Fy_exact) / abs(Fy_exact),
            "sym_resid": np.sqrt(Aout["Fx"]**2 + Aout["Fz"]**2 + Aout["Mx"]**2 + Aout["Mz"]**2),
            "time_sec": t1 - t0,
        })

        rows.append({
            "mode": "shallow_sweep",
            "method": "B_local_normal",
            "delta": delta,
            "resolution": Na_B_fixed,
            "Fy_num": Bout["Fy"],
            "Fy_exact": Fy_exact,
            "rel_err_Fy": abs(Bout["Fy"] - Fy_exact) / abs(Fy_exact),
            "sym_resid": np.sqrt(Bout["Fx"]**2 + Bout["Fz"]**2 + Bout["Mx"]**2 + Bout["Mz"]**2),
            "time_sec": t3 - t2,
        })

    # 2) shallow convergence at one particularly shallow case
    delta_conv = 0.03
    Fy_exact_conv = analytic_force(R, delta_conv, k)

    for N in [48, 64, 80, 96, 120, 144]:
        t0 = time.perf_counter()
        out = version_A_direct_band(
            R=R, delta=delta_conv, k=k,
            cube_size=cube_size, cube_height=cube_height,
            N=N, eta_factor=1.5
        )
        t1 = time.perf_counter()
        rows.append({
            "mode": "convergence",
            "method": "A_direct_band",
            "delta": delta_conv,
            "resolution": N,
            "Fy_num": out["Fy"],
            "Fy_exact": Fy_exact_conv,
            "rel_err_Fy": abs(out["Fy"] - Fy_exact_conv) / abs(Fy_exact_conv),
            "sym_resid": np.sqrt(out["Fx"]**2 + out["Fz"]**2 + out["Mx"]**2 + out["Mz"]**2),
            "time_sec": t1 - t0,
        })

    for Na in [6, 8, 10, 12, 16, 24, 32, 48]:
        Nt = 4 * Na
        t0 = time.perf_counter()
        out = version_B_local_normal_sheet(
            R=R, delta=delta_conv, k=k,
            Na=Na, Nt=Nt
        )
        t1 = time.perf_counter()
        rows.append({
            "mode": "convergence",
            "method": "B_local_normal",
            "delta": delta_conv,
            "resolution": Na,
            "Fy_num": out["Fy"],
            "Fy_exact": Fy_exact_conv,
            "rel_err_Fy": abs(out["Fy"] - Fy_exact_conv) / abs(Fy_exact_conv),
            "sym_resid": np.sqrt(out["Fx"]**2 + out["Fz"]**2 + out["Mx"]**2 + out["Mz"]**2),
            "time_sec": t1 - t0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "hemisphere_cube_AB_shallow_results.csv", index=False)

    # Plot 1: shallow force-vs-penetration
    sweep = df[df["mode"] == "shallow_sweep"].copy()
    plt.figure(figsize=(7.0, 4.6))
    for method, grp in sweep.groupby("method"):
        grp = grp.sort_values("delta")
        plt.plot(grp["delta"], grp["Fy_num"], marker="o", label=method)
    exact_curve = sweep.drop_duplicates("delta").sort_values("delta")
    plt.plot(exact_curve["delta"], exact_curve["Fy_exact"], marker="x", label="exact")
    plt.xlabel("penetration depth δ")
    plt.ylabel("vertical force Fy")
    plt.title("Shallow indentation: Version A vs Version B")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hemisphere_cube_AB_shallow_force_compare.png", dpi=180)
    plt.close()

    # Plot 2: convergence
    conv = df[df["mode"] == "convergence"].copy()
    plt.figure(figsize=(7.0, 4.6))
    for method, grp in conv.groupby("method"):
        grp = grp.sort_values("resolution")
        plt.plot(grp["resolution"], grp["rel_err_Fy"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("resolution parameter")
    plt.ylabel("relative Fy error")
    plt.title("Shallow case convergence at δ=0.03")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hemisphere_cube_AB_convergence.png", dpi=180)
    plt.close()

    # Plot 3: error vs runtime
    plt.figure(figsize=(7.0, 4.6))
    for method, grp in conv.groupby("method"):
        grp = grp.sort_values("time_sec")
        plt.plot(grp["time_sec"], grp["rel_err_Fy"], marker="o", label=method)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("runtime [s]")
    plt.ylabel("relative Fy error")
    plt.title("Accuracy vs runtime for the shallow case")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hemisphere_cube_AB_error_vs_runtime.png", dpi=180)
    plt.close()

    # Plot 4: symmetry residual
    plt.figure(figsize=(7.0, 4.6))
    for method, grp in conv.groupby("method"):
        grp = grp.sort_values("resolution")
        plt.plot(grp["resolution"], grp["sym_resid"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("resolution parameter")
    plt.ylabel("symmetry residual")
    plt.title("Centered axisymmetric case: residual non-vertical components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hemisphere_cube_AB_symmetry_residual.png", dpi=180)
    plt.close()

    # Plot 5: local-normal geometry
    make_geometry_demo(figures_dir, R=R, delta=delta_conv, Na=12, Nt=48)

    print("Saved:")
    print(data_dir / "hemisphere_cube_AB_shallow_results.csv")
    print(figures_dir / "hemisphere_cube_AB_shallow_force_compare.png")
    print(figures_dir / "hemisphere_cube_AB_error_vs_runtime.png")
    print(figures_dir / "hemisphere_cube_AB_convergence.png")
    print(figures_dir / "hemisphere_cube_AB_symmetry_residual.png")
    print(figures_dir / "hemisphere_cube_AB_local_normal_geometry.png")


if __name__ == "__main__":
    main()

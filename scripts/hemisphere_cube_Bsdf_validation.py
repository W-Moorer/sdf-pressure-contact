#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upgrade Version B from "analytic sheet" to "automatically extracted sheet elements from SDF"
for the 3D benchmark:
    rigid hemisphere pressing shallowly into the top face of a compliant cube.

Methods:
    A_direct_band:
        direct 3D voxel narrow-band integral in the cube volume

    B_analytic_sheet:
        local-normal accumulator on analytically parameterized spherical-cap sheet

    B_sdf_extracted_sheet:
        local-normal accumulator where sheet elements are extracted automatically
        from the sphere SDF by:
            1) scanning x-z columns
            2) detecting where the sphere SDF crosses zero along y
            3) locating the root by bisection
            4) building a sheet element at that point with:
                   centroid  = root point
                   normal    = grad(phi)/|grad(phi)|
                   area      = dx*dz / |n_y|
        This is the benchmark-specific realization of extracting local sheet elements
        directly from SDF data, without using the analytic spherical parameterization.

Outputs:
    - hemisphere_cube_Bsdf_results.csv
    - hemisphere_cube_Bsdf_shallow_force_compare.png
    - hemisphere_cube_Bsdf_convergence.png
    - hemisphere_cube_Bsdf_error_vs_runtime.png
    - hemisphere_cube_Bsdf_symmetry_residual.png
    - hemisphere_cube_Bsdf_geometry_demo.png

Usage:
    python hemisphere_cube_Bsdf_validation.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time


# -----------------------------
# Shared helpers
# -----------------------------
def delta_cosine(s: np.ndarray, eta: float) -> np.ndarray:
    a = np.abs(s)
    out = np.zeros_like(s, dtype=float)
    mask = a <= eta
    out[mask] = 0.5 / eta * (1.0 + np.cos(np.pi * s[mask] / eta))
    return out


def exact_cap_volume(R: float, delta: float) -> float:
    return np.pi * delta * delta * (R - delta / 3.0)


def exact_contact_radius(R: float, delta: float) -> float:
    return np.sqrt(max(0.0, 2.0 * R * delta - delta * delta))


def analytic_force(R: float, delta: float, k: float) -> float:
    return k * exact_cap_volume(R, delta)


def sphere_phi_xyz(x, y, z, *, R: float, delta: float):
    cy = R - delta
    return np.sqrt(x * x + (y - cy) * (y - cy) + z * z) - R


def sphere_normal_xyz(x, y, z, *, R: float, delta: float):
    cy = R - delta
    dx = x
    dy = y - cy
    dz = z
    rr = np.sqrt(dx * dx + dy * dy + dz * dz)
    eps = 1e-15
    return np.array([dx, dy, dz]) / max(rr, eps)


# -----------------------------
# Method A: direct 3D voxel band integral
# -----------------------------
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

    cy = R - delta
    RR = np.sqrt(X * X + (Y - cy) * (Y - cy) + Z * Z)
    phi = RR - R

    eps = 1e-15
    nx = X / np.maximum(RR, eps)
    ny = (Y - cy) / np.maximum(RR, eps)
    nz = Z / np.maximum(RR, eps)

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

    return {"Fx": Fx, "Fy": Fy, "Fz": Fz, "Mx": Mx, "My": My, "Mz": Mz}


# -----------------------------
# Method B1: analytic sheet
# -----------------------------
def version_B_analytic_sheet(
    *,
    R: float,
    delta: float,
    k: float,
    Na: int,
    Nt: int,
):
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
    Mx = np.sum(Y * dFz - Z * dFy)
    My = np.sum(Z * dFx - X * dFz)
    Mz = np.sum(X * dFy - Y * dFx)

    return {"Fx": Fx, "Fy": Fy, "Fz": Fz, "Mx": Mx, "My": My, "Mz": Mz}


# -----------------------------
# Method B2: sheet extracted automatically from SDF
# -----------------------------
def extract_sheet_elements_from_sdf(
    *,
    R: float,
    delta: float,
    cube_size: float,
    cube_height: float,
    Nxz: int,
    bisection_steps: int = 30,
):
    """
    Automatic sheet extraction from SDF on x-z columns.

    For each x-z column center:
        if phi(x,0,z) <= 0 and phi(x,-cube_height,z) >= 0,
        then there is a sheet crossing in that column.
        Locate y_sigma by bisection on phi(x,y,z)=0.

    Surface element area:
        dA = dx * dz / |n_y|
    valid for a local graph y = y(x,z), which holds for this benchmark.
    """
    x_min, x_max = -cube_size / 2.0, cube_size / 2.0
    z_min, z_max = -cube_size / 2.0, cube_size / 2.0

    xs = np.linspace(x_min, x_max, Nxz, endpoint=False) + (x_max - x_min) / Nxz / 2.0
    zs = np.linspace(z_min, z_max, Nxz, endpoint=False) + (z_max - z_min) / Nxz / 2.0
    dx = (x_max - x_min) / Nxz
    dz = (z_max - z_min) / Nxz

    elems = []
    for x in xs:
        for z in zs:
            phi_top = sphere_phi_xyz(x, 0.0, z, R=R, delta=delta)
            phi_bot = sphere_phi_xyz(x, -cube_height, z, R=R, delta=delta)

            # Need a sign change over [-cube_height, 0]
            if not (phi_top <= 0.0 and phi_bot >= 0.0):
                continue

            yl = -cube_height
            yr = 0.0
            fl = phi_bot
            fr = phi_top

            # bisection
            for _ in range(bisection_steps):
                ym = 0.5 * (yl + yr)
                fm = sphere_phi_xyz(x, ym, z, R=R, delta=delta)
                if fm > 0.0:
                    yl = ym
                    fl = fm
                else:
                    yr = ym
                    fr = fm

            y_sigma = 0.5 * (yl + yr)
            n = sphere_normal_xyz(x, y_sigma, z, R=R, delta=delta)

            ny = n[1]
            if abs(ny) < 1e-12:
                continue

            dA = dx * dz / abs(ny)
            elems.append({
                "x": x,
                "y": y_sigma,
                "z": z,
                "nx": n[0],
                "ny": n[1],
                "nz": n[2],
                "dA": dA,
            })

    return elems


def version_B_sdf_extracted_sheet(
    *,
    R: float,
    delta: float,
    k: float,
    cube_size: float,
    cube_height: float,
    Nxz: int,
    bisection_steps: int = 30,
):
    elems = extract_sheet_elements_from_sdf(
        R=R, delta=delta, cube_size=cube_size, cube_height=cube_height,
        Nxz=Nxz, bisection_steps=bisection_steps
    )

    Fx = Fy = Fz = 0.0
    Mx = My = Mz = 0.0

    for e in elems:
        p = k * max(0.0, -e["y"])
        dFx = -p * e["nx"] * e["dA"]
        dFy = -p * e["ny"] * e["dA"]
        dFz = -p * e["nz"] * e["dA"]

        Fx += dFx
        Fy += dFy
        Fz += dFz

        x, y, z = e["x"], e["y"], e["z"]
        Mx += y * dFz - z * dFy
        My += z * dFx - x * dFz
        Mz += x * dFy - y * dFx

    return {
        "Fx": Fx, "Fy": Fy, "Fz": Fz,
        "Mx": Mx, "My": My, "Mz": Mz,
        "num_elems": len(elems),
        "elems": elems,
    }


# -----------------------------
# Geometry demo
# -----------------------------
def make_geometry_demo(out_dir: Path, *, elems: list[dict], R: float, delta: float):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    if elems:
        X = np.array([e["x"] for e in elems])
        Y = np.array([e["y"] for e in elems])
        Z = np.array([e["z"] for e in elems])
        NX = np.array([e["nx"] for e in elems])
        NY = np.array([e["ny"] for e in elems])
        NZ = np.array([e["nz"] for e in elems])

        ax.scatter(X, Y, Z, s=8)

        stride = max(1, len(elems) // 120)
        ax.quiver(X[::stride], Y[::stride], Z[::stride],
                  NX[::stride], NY[::stride], NZ[::stride],
                  length=0.08, normalize=True)

    a = exact_contact_radius(R, delta)
    t = np.linspace(0.0, 2.0 * np.pi, 300)
    ax.plot(a * np.cos(t), np.zeros_like(t), a * np.sin(t), linewidth=2.0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("B_sdf_extracted_sheet: sheet elements recovered automatically from SDF")
    ax.set_box_aspect((1, 0.6, 1))
    plt.tight_layout()
    plt.savefig(out_dir / "hemisphere_cube_Bsdf_geometry_demo.png", dpi=180)
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

    # Shallow focus
    cube_size = 1.6
    cube_height = 0.25
    deltas = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12])

    rows = []

    # Sweep at fixed resolution settings
    N_A_fixed = 96
    Na_B1_fixed = 24
    Nt_B1_fixed = 96
    Nxz_B2_fixed = 64

    for delta in deltas:
        Fy_exact = analytic_force(R, delta, k)

        t0 = time.perf_counter()
        Aout = version_A_direct_band(
            R=R, delta=delta, k=k,
            cube_size=cube_size, cube_height=cube_height,
            N=N_A_fixed, eta_factor=1.5
        )
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        B1out = version_B_analytic_sheet(
            R=R, delta=delta, k=k,
            Na=Na_B1_fixed, Nt=Nt_B1_fixed
        )
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        B2out = version_B_sdf_extracted_sheet(
            R=R, delta=delta, k=k,
            cube_size=cube_size, cube_height=cube_height,
            Nxz=Nxz_B2_fixed, bisection_steps=30
        )
        t5 = time.perf_counter()

        for method, out, dt, res in [
            ("A_direct_band", Aout, t1 - t0, N_A_fixed),
            ("B_analytic_sheet", B1out, t3 - t2, Na_B1_fixed),
            ("B_sdf_extracted_sheet", B2out, t5 - t4, Nxz_B2_fixed),
        ]:
            rows.append({
                "mode": "shallow_sweep",
                "method": method,
                "delta": delta,
                "resolution": res,
                "Fy_num": out["Fy"],
                "Fy_exact": Fy_exact,
                "rel_err_Fy": abs(out["Fy"] - Fy_exact) / abs(Fy_exact),
                "sym_resid": np.sqrt(out["Fx"]**2 + out["Fz"]**2 + out["Mx"]**2 + out["Mz"]**2),
                "time_sec": dt,
            })

    # Convergence at a shallow case
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
        out = version_B_analytic_sheet(
            R=R, delta=delta_conv, k=k,
            Na=Na, Nt=Nt
        )
        t1 = time.perf_counter()
        rows.append({
            "mode": "convergence",
            "method": "B_analytic_sheet",
            "delta": delta_conv,
            "resolution": Na,
            "Fy_num": out["Fy"],
            "Fy_exact": Fy_exact_conv,
            "rel_err_Fy": abs(out["Fy"] - Fy_exact_conv) / abs(Fy_exact_conv),
            "sym_resid": np.sqrt(out["Fx"]**2 + out["Fz"]**2 + out["Mx"]**2 + out["Mz"]**2),
            "time_sec": t1 - t0,
        })

    extracted_demo = None
    for Nxz in [12, 16, 24, 32, 48, 64, 96]:
        t0 = time.perf_counter()
        out = version_B_sdf_extracted_sheet(
            R=R, delta=delta_conv, k=k,
            cube_size=cube_size, cube_height=cube_height,
            Nxz=Nxz, bisection_steps=30
        )
        t1 = time.perf_counter()
        rows.append({
            "mode": "convergence",
            "method": "B_sdf_extracted_sheet",
            "delta": delta_conv,
            "resolution": Nxz,
            "Fy_num": out["Fy"],
            "Fy_exact": Fy_exact_conv,
            "rel_err_Fy": abs(out["Fy"] - Fy_exact_conv) / abs(Fy_exact_conv),
            "sym_resid": np.sqrt(out["Fx"]**2 + out["Fz"]**2 + out["Mx"]**2 + out["Mz"]**2),
            "time_sec": t1 - t0,
        })
        if Nxz == 24:
            extracted_demo = out["elems"]

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "hemisphere_cube_Bsdf_results.csv", index=False)

    # Plot 1: shallow force compare
    sweep = df[df["mode"] == "shallow_sweep"].copy()
    plt.figure(figsize=(7.0, 4.6))
    for method, grp in sweep.groupby("method"):
        grp = grp.sort_values("delta")
        plt.plot(grp["delta"], grp["Fy_num"], marker="o", label=method)
    exact_curve = sweep.drop_duplicates("delta").sort_values("delta")
    plt.plot(exact_curve["delta"], exact_curve["Fy_exact"], marker="x", label="exact")
    plt.xlabel("penetration depth δ")
    plt.ylabel("vertical force Fy")
    plt.title("Shallow indentation: A vs B_analytic vs B_sdf_extracted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hemisphere_cube_Bsdf_shallow_force_compare.png", dpi=180)
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
    plt.savefig(figures_dir / "hemisphere_cube_Bsdf_convergence.png", dpi=180)
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
    plt.title("Accuracy vs runtime at shallow indentation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hemisphere_cube_Bsdf_error_vs_runtime.png", dpi=180)
    plt.close()

    # Plot 4: symmetry residual
    plt.figure(figsize=(7.0, 4.6))
    for method, grp in conv.groupby("method"):
        grp = grp.sort_values("resolution")
        plt.plot(grp["resolution"], grp["sym_resid"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("resolution parameter")
    plt.ylabel("symmetry residual")
    plt.title("Centered axisymmetric case residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hemisphere_cube_Bsdf_symmetry_residual.png", dpi=180)
    plt.close()

    # Plot 5: geometry demo
    if extracted_demo is not None:
        make_geometry_demo(figures_dir, elems=extracted_demo, R=R, delta=delta_conv)

    print("Saved:")
    print(data_dir / "hemisphere_cube_Bsdf_results.csv")
    print(figures_dir / "hemisphere_cube_Bsdf_shallow_force_compare.png")
    print(figures_dir / "hemisphere_cube_Bsdf_convergence.png")
    print(figures_dir / "hemisphere_cube_Bsdf_error_vs_runtime.png")
    print(figures_dir / "hemisphere_cube_Bsdf_symmetry_residual.png")
    print(figures_dir / "hemisphere_cube_Bsdf_geometry_demo.png")


if __name__ == "__main__":
    main()

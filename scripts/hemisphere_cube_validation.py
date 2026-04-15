#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D validation example:
    rigid hemisphere pressing downward into the top face of a compliant cube

Model:
    - Rigid hemisphere / spherical lower surface of radius R
    - Compliant cube top behaves like a hydrostatic / linear pressure field:
          p(y) = k * max(0, -y)
      inside the cube (top face at y=0)
    - Traction on the rigid spherical surface:
          t = -p * n_out
    - Numerical force from 3D SDF narrow-band volume integral:
          F ≈ ∫_{cube} -p(y) ∇phi_sphere(x) δ_eta(phi_sphere(x)) dV
    - Analytic reference for centered indentation:
          Fy_exact = k * V_cap
      where V_cap is the submerged spherical-cap volume:
          V_cap = π δ^2 (R - δ/3)

Why this is a useful 3D validation:
    1) The geometry is truly 3D and curved.
    2) The force direction comes from the 3D surface normal, not a fixed global axis.
    3) There is an analytic reference for the net vertical force.

Outputs:
    - hemisphere_cube_validation_results.csv
    - hemisphere_cube_force_vs_penetration.png
    - hemisphere_cube_convergence.png
    - hemisphere_cube_geometry_demo.png

Usage:
    python hemisphere_cube_validation.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def delta_cosine(s: np.ndarray, eta: float) -> np.ndarray:
    a = np.abs(s)
    out = np.zeros_like(s, dtype=float)
    mask = a <= eta
    out[mask] = 0.5 / eta * (1.0 + np.cos(np.pi * s[mask] / eta))
    return out


def sphere_sdf_and_grad(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: tuple[float, float, float], R: float):
    cx, cy, cz = center
    DX = X - cx
    DY = Y - cy
    DZ = Z - cz
    RR = np.sqrt(DX * DX + DY * DY + DZ * DZ)
    phi = RR - R

    # outward unit normal of the sphere
    eps = 1e-15
    nx = DX / np.maximum(RR, eps)
    ny = DY / np.maximum(RR, eps)
    nz = DZ / np.maximum(RR, eps)
    return phi, nx, ny, nz


def exact_cap_volume(R: float, delta: float) -> float:
    return np.pi * delta * delta * (R - delta / 3.0)


def exact_cap_surface_area(R: float, delta: float) -> float:
    # wetted spherical surface area
    return 2.0 * np.pi * R * delta


def exact_contact_radius(R: float, delta: float) -> float:
    return np.sqrt(max(0.0, 2.0 * R * delta - delta * delta))


def numerical_force_band(
    *,
    R: float,
    delta: float,
    k: float,
    cube_size: float,
    cube_height: float,
    N: int,
    eta_factor: float = 1.5,
):
    """
    3D narrow-band integral over the cube volume.

    The cube occupies:
        x,z in [-cube_size/2, cube_size/2]
        y   in [-cube_height, 0]

    The hemisphere is centered on the vertical axis:
        center = (0, R - delta, 0)
    For delta < R, the submerged curved part is the same as the lower spherical cap.
    """
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

    # cube pressure field: p = k * depth below top face
    p = k * np.clip(-Y, 0.0, None)

    # narrow band width in signed-distance space
    h = max(dx, dy, dz)
    eta = eta_factor * h
    band = delta_cosine(phi, eta)

    # traction on rigid body is inward normal = -n_out
    tx = -p * nx
    ty = -p * ny
    tz = -p * nz

    Fx = np.sum(tx * band) * dV
    Fy = np.sum(ty * band) * dV
    Fz = np.sum(tz * band) * dV

    # use the same band to estimate the wetted spherical surface area
    area_num = np.sum(band) * dV

    # moments about origin
    Mx = np.sum((Y * tz - Z * ty) * band) * dV
    My = np.sum((Z * tx - X * tz) * band) * dV
    Mz = np.sum((X * ty - Y * tx) * band) * dV

    return {
        "dx": dx, "dy": dy, "dz": dz, "eta": eta,
        "Fx": Fx, "Fy": Fy, "Fz": Fz,
        "Mx": Mx, "My": My, "Mz": Mz,
        "area_num": area_num,
    }


def make_geometry_demo(out_dir: Path, *, R: float, delta: float, cube_size: float):
    """
    3D wireframe demo of the hemisphere/cube-top geometry and the exact circular contact boundary.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    cy = R - delta

    # lower hemisphere wireframe
    theta = np.linspace(0.0, 2.0 * np.pi, 80)
    phi = np.linspace(np.pi / 2.0, np.pi, 30)  # lower hemisphere only
    TT, PP = np.meshgrid(theta, phi)
    X = R * np.sin(PP) * np.cos(TT)
    Y = cy + R * np.cos(PP)
    Z = R * np.sin(PP) * np.sin(TT)
    ax.plot_wireframe(X, Y, Z, rstride=2, cstride=4, linewidth=0.6)

    # cube top face as a wireframe grid
    gx = np.linspace(-cube_size / 2.0, cube_size / 2.0, 12)
    gz = np.linspace(-cube_size / 2.0, cube_size / 2.0, 12)
    GX, GZ = np.meshgrid(gx, gz)
    GY = np.zeros_like(GX)
    ax.plot_wireframe(GX, GY, GZ, rstride=1, cstride=1, linewidth=0.6)

    # exact circular contact boundary on y=0
    a = exact_contact_radius(R, delta)
    t = np.linspace(0.0, 2.0 * np.pi, 300)
    ax.plot(a * np.cos(t), np.zeros_like(t), a * np.sin(t), linewidth=2.0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D geometry: hemisphere pressing the cube top face")
    ax.set_box_aspect((1, 0.7, 1))
    plt.tight_layout()
    plt.savefig(out_dir / "hemisphere_cube_geometry_demo.png", dpi=180)
    plt.close()


def main():
    out_dir = Path.cwd()
    figures_dir = out_dir / "result" / "figures"
    data_dir = out_dir / "result" / "data"

    # Geometry / material parameters
    R = 1.0
    k = 10.0
    cube_size = 2.8
    cube_height = 1.2

    # Sweep in penetration depth
    deltas = np.linspace(0.05, 0.45, 9)
    N_sweep = 84

    rows = []
    for delta in deltas:
        num = numerical_force_band(
            R=R, delta=delta, k=k,
            cube_size=cube_size, cube_height=cube_height,
            N=N_sweep, eta_factor=1.5
        )

        V_cap = exact_cap_volume(R, delta)
        A_cap = exact_cap_surface_area(R, delta)
        Fy_exact = k * V_cap
        a_exact = exact_contact_radius(R, delta)

        rows.append({
            "mode": "penetration_sweep",
            "N": N_sweep,
            "delta": delta,
            "Fy_num": num["Fy"],
            "Fy_exact": Fy_exact,
            "rel_err_Fy": abs(num["Fy"] - Fy_exact) / abs(Fy_exact),
            "area_num": num["area_num"],
            "area_exact": A_cap,
            "rel_err_area": abs(num["area_num"] - A_cap) / abs(A_cap),
            "Fx_num": num["Fx"],
            "Fz_num": num["Fz"],
            "Mx_num": num["Mx"],
            "Mz_num": num["Mz"],
            "a_exact": a_exact,
        })

    # Convergence study at one moderately curved case
    delta_conv = 0.30
    for N in [36, 48, 60, 72, 84, 96, 120]:
        num = numerical_force_band(
            R=R, delta=delta_conv, k=k,
            cube_size=cube_size, cube_height=cube_height,
            N=N, eta_factor=1.5
        )
        V_cap = exact_cap_volume(R, delta_conv)
        A_cap = exact_cap_surface_area(R, delta_conv)
        Fy_exact = k * V_cap

        rows.append({
            "mode": "convergence",
            "N": N,
            "delta": delta_conv,
            "Fy_num": num["Fy"],
            "Fy_exact": Fy_exact,
            "rel_err_Fy": abs(num["Fy"] - Fy_exact) / abs(Fy_exact),
            "area_num": num["area_num"],
            "area_exact": A_cap,
            "rel_err_area": abs(num["area_num"] - A_cap) / abs(A_cap),
            "Fx_num": num["Fx"],
            "Fz_num": num["Fz"],
            "Mx_num": num["Mx"],
            "Mz_num": num["Mz"],
            "a_exact": exact_contact_radius(R, delta_conv),
        })

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "hemisphere_cube_validation_results.csv", index=False)

    # Plot 1: force vs penetration
    sweep = df[df["mode"] == "penetration_sweep"].sort_values("delta")
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(sweep["delta"], sweep["Fy_num"], marker="o", label="numerical band integral")
    plt.plot(sweep["delta"], sweep["Fy_exact"], marker="x", label="analytic reference")
    plt.xlabel("penetration depth δ")
    plt.ylabel("vertical force Fy")
    plt.title("Hemisphere pressing cube: force vs penetration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hemisphere_cube_force_vs_penetration.png", dpi=180)
    plt.close()

    # Plot 2: convergence
    conv = df[df["mode"] == "convergence"].sort_values("N")
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(conv["N"], conv["rel_err_Fy"], marker="o", label="force error")
    plt.plot(conv["N"], np.abs(conv["Fx_num"]), marker="s", label="|Fx| symmetry residual")
    plt.plot(conv["N"], np.abs(conv["Fz_num"]), marker="^", label="|Fz| symmetry residual")
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("error / residual")
    plt.title("3D convergence for the hemisphere-cube benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "hemisphere_cube_convergence.png", dpi=180)
    plt.close()

    # Plot 3: geometry demo
    make_geometry_demo(figures_dir, R=R, delta=delta_conv, cube_size=cube_size)

    print("Saved:")
    print(data_dir / "hemisphere_cube_validation_results.csv")
    print(figures_dir / "hemisphere_cube_force_vs_penetration.png")
    print(figures_dir / "hemisphere_cube_convergence.png")
    print(figures_dir / "hemisphere_cube_geometry_demo.png")


if __name__ == "__main__":
    main()

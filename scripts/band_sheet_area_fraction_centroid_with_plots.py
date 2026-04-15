#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-contained comparison of:
    1) cell-center patch
    2) area-fraction patch
    3) area-fraction + local centroid patch

embedded into the full band-integral pipeline.

This script saves:
    - band_sheet_area_fraction_centroid_metrics.csv
    - band_sheet_area_fraction_centroid_demo.png
    - band_sheet_area_error_comparison.png
    - band_sheet_force_error_comparison.png
    - band_sheet_moment_error_comparison.png

Usage:
    python band_sheet_area_fraction_centroid_with_plots.py

Notes:
    1) Uses matplotlib Agg backend, so it works on headless servers.
    2) Images are saved to the current working directory.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def delta_cosine(s):
    raise RuntimeError("This helper should not be called directly.")


def delta_cosine(s: np.ndarray, eta: float) -> np.ndarray:
    a = np.abs(s)
    out = np.zeros_like(s, dtype=float)
    mask = a <= eta
    out[mask] = 0.5 / eta * (1.0 + np.cos(np.pi * s[mask] / eta))
    return out


def analytic_planar_contact(kA: float, kB: float, delta: float, patch_area: float, xbar: float, zbar: float = 0.0) -> dict:
    y_star = -kB * delta / (kA + kB)
    p_star = (kA * kB / (kA + kB)) * delta
    Fy = patch_area * p_star
    Mx = -zbar * Fy
    Mz = xbar * Fy
    return {
        "y_star": y_star,
        "p_star": p_star,
        "area": patch_area,
        "xbar": xbar,
        "zbar": zbar,
        "Fy": Fy,
        "Mx": Mx,
        "Mz": Mz,
    }


def rect_vertices(center: tuple[float, float], size: tuple[float, float], angle_rad: float) -> np.ndarray:
    cx, cz = center
    lx, lz = size
    hx, hz = lx / 2.0, lz / 2.0
    local = np.array([[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz]])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([cx, cz])


def point_in_rotated_rect(points: np.ndarray, center: tuple[float, float], size: tuple[float, float], angle_rad: float) -> np.ndarray:
    cx, cz = center
    lx, lz = size
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    Rt = np.array([[c, s], [-s, c]])  # inverse rotation
    local = (points - np.array([cx, cz])) @ Rt.T
    return (np.abs(local[:, 0]) <= lx / 2.0) & (np.abs(local[:, 1]) <= lz / 2.0)


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return a[0] * b[1] - a[1] * b[0]


def line_intersection_with_clip_edge(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    r = p2 - p1
    s = b - a
    denom = cross2(r, s)
    if abs(denom) < eps:
        return p2.copy()
    t = cross2(a - p1, s) / denom
    return p1 + t * r


def clip_polygon_against_edge(poly: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if len(poly) == 0:
        return poly
    output = []
    prev = poly[-1]
    prev_inside = cross2(b - a, prev - a) >= -eps
    for curr in poly:
        curr_inside = cross2(b - a, curr - a) >= -eps
        if curr_inside:
            if not prev_inside:
                output.append(line_intersection_with_clip_edge(prev, curr, a, b))
            output.append(curr)
        elif prev_inside:
            output.append(line_intersection_with_clip_edge(prev, curr, a, b))
        prev = curr
        prev_inside = curr_inside
    return np.array(output, dtype=float)


def clip_square_by_convex_polygon(square_poly: np.ndarray, clip_poly: np.ndarray) -> np.ndarray:
    poly = square_poly.copy()
    for i in range(len(clip_poly)):
        a = clip_poly[i]
        b = clip_poly[(i + 1) % len(clip_poly)]
        poly = clip_polygon_against_edge(poly, a, b)
        if len(poly) == 0:
            break
    return poly


def polygon_area_centroid(poly: np.ndarray) -> tuple[float, np.ndarray]:
    if len(poly) < 3:
        return 0.0, np.array([np.nan, np.nan])
    x = poly[:, 0]
    y = poly[:, 1]
    xp = np.roll(x, -1)
    yp = np.roll(y, -1)
    cross = x * yp - xp * y
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-14:
        return 0.0, np.array([np.nan, np.nan])
    Cx = np.sum((x + xp) * cross) / (6.0 * A)
    Cy = np.sum((y + yp) * cross) / (6.0 * A)
    return abs(A), np.array([Cx, Cy])


def make_grid(xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float, Nx: int, Ny: int, Nz: int):
    x_edges = np.linspace(xmin, xmax, Nx + 1)
    y_edges = np.linspace(ymin, ymax, Ny + 1)
    z_edges = np.linspace(zmin, zmax, Nz + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    dz = z_edges[1] - z_edges[0]
    return x_edges, y_edges, z_edges, x_centers, y_centers, z_centers, dx, dy, dz


def clipped_column_geometry(x_edges: np.ndarray, z_edges: np.ndarray, rect_poly: np.ndarray):
    Nx = len(x_edges) - 1
    Nz = len(z_edges) - 1
    frac = np.zeros((Nx, Nz), dtype=float)
    cx = np.zeros((Nx, Nz), dtype=float)
    cz = np.zeros((Nx, Nz), dtype=float)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    Xc, Zc = np.meshgrid(x_centers, z_centers, indexing="ij")
    cx[:, :] = Xc
    cz[:, :] = Zc

    dx = x_edges[1] - x_edges[0]
    dz = z_edges[1] - z_edges[0]
    cell_area = dx * dz

    for i in range(Nx):
        for j in range(Nz):
            square = np.array([
                [x_edges[i],     z_edges[j]],
                [x_edges[i + 1], z_edges[j]],
                [x_edges[i + 1], z_edges[j + 1]],
                [x_edges[i],     z_edges[j + 1]],
            ], dtype=float)
            clipped = clip_square_by_convex_polygon(square, rect_poly)
            A, C = polygon_area_centroid(clipped)
            frac[i, j] = A / cell_area
            if A > 0:
                cx[i, j] = C[0]
                cz[i, j] = C[1]
    return frac, cx, cz


def band_integral_planar_patch(
    Nx: int,
    Ny: int,
    Nz: int,
    *,
    kA: float,
    kB: float,
    delta: float,
    xmin: float,
    xmax: float,
    zmin: float,
    zmax: float,
    pad_y: float,
    patch_center: tuple[float, float],
    patch_size: tuple[float, float],
    angle_rad: float,
    method: str = "cell_center",
    eta_factor: float = 2.0,
) -> dict:
    ymin = -delta - pad_y
    ymax = pad_y

    x_edges, y_edges, z_edges, x_centers, y_centers, z_centers, dx, dy, dz = make_grid(
        xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz
    )
    dV = dx * dy * dz

    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

    # Opposing half-spaces
    phi_A = Y
    phi_B = -(Y + delta)
    dA = np.clip(-phi_A, 0.0, None)
    dB = np.clip(-phi_B, 0.0, None)
    pA = kA * dA
    pB = kB * dB

    h = pA - pB
    grad_h_norm = kA + kB
    eta_h = grad_h_norm * eta_factor * dy

    overlap = (phi_A <= 0.0) & (phi_B <= 0.0)
    band = delta_cosine(h, eta_h) * grad_h_norm * overlap
    p_bar = 0.5 * (pA + pB)

    Xc2, Zc2 = np.meshgrid(x_centers, z_centers, indexing="ij")
    pts2 = np.column_stack([Xc2.ravel(), Zc2.ravel()])
    center_mask = point_in_rotated_rect(pts2, patch_center, patch_size, angle_rad).reshape(Nx, Nz).astype(float)

    rect_poly = rect_vertices(patch_center, patch_size, angle_rad)
    frac, cx_local, cz_local = clipped_column_geometry(x_edges, z_edges, rect_poly)

    if method == "cell_center":
        patch_weight = center_mask[:, None, :]
        Xmom = X
        Zmom = Z
    elif method == "area_fraction":
        patch_weight = frac[:, None, :]
        Xmom = X
        Zmom = Z
    elif method == "area_fraction_centroid":
        patch_weight = frac[:, None, :]
        Xmom = np.broadcast_to(cx_local[:, None, :], (Nx, Ny, Nz))
        Zmom = np.broadcast_to(cz_local[:, None, :], (Nx, Ny, Nz))
    else:
        raise ValueError(f"Unknown method: {method}")

    weight = band * patch_weight

    Fy = np.sum(p_bar * weight) * dV
    Mx = np.sum((-Zmom * p_bar) * weight) * dV
    Mz = np.sum((Xmom * p_bar) * weight) * dV

    area_est = np.sum(weight) * dV
    y_star_est = np.sum(Y * weight) * dV / area_est if area_est > 0 else np.nan
    xbar_est = np.sum(Xmom * weight) * dV / area_est if area_est > 0 else np.nan
    zbar_est = np.sum(Zmom * weight) * dV / area_est if area_est > 0 else np.nan

    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "Fy": Fy,
        "Mx": Mx,
        "Mz": Mz,
        "area_est": area_est,
        "y_star_est": y_star_est,
        "xbar_est": xbar_est,
        "zbar_est": zbar_est,
        "frac": frac,
        "center_mask": center_mask,
        "cx_local": cx_local,
        "cz_local": cz_local,
        "rect_poly": rect_poly,
        "x_edges": x_edges,
        "z_edges": z_edges,
    }


def make_plots(df: pd.DataFrame, demo: dict, out_dir: Path) -> None:
    # 1) Local area fraction / centroid demo
    plt.figure(figsize=(7.2, 5.8))
    Xe, Ze = np.meshgrid(demo["x_edges"], demo["z_edges"], indexing="ij")
    plt.pcolormesh(Xe, Ze, demo["frac"], shading="flat")
    poly = np.vstack([demo["rect_poly"], demo["rect_poly"][0]])
    plt.plot(poly[:, 0], poly[:, 1], linewidth=2.0)
    mask = demo["frac"] > 1e-12
    plt.scatter(demo["cx_local"][mask], demo["cz_local"][mask], s=14)
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Local area fraction and local centroid for each x-z column")
    plt.colorbar(label="column area fraction inside exact patch")
    plt.tight_layout()
    plt.savefig(out_dir / "band_sheet_area_fraction_centroid_demo.png", dpi=180)
    plt.close()

    # 2) Area error
    plt.figure(figsize=(7.0, 4.5))
    for method, grp in df.groupby("method"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["rel_err_area"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("relative area error")
    plt.title("Area error: cell-center vs area-fraction variants")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "band_sheet_area_error_comparison.png", dpi=180)
    plt.close()

    # 3) Force error
    plt.figure(figsize=(7.0, 4.5))
    for method, grp in df.groupby("method"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["rel_err_Fy"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("relative force error")
    plt.title("Force error in the full band integral")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "band_sheet_force_error_comparison.png", dpi=180)
    plt.close()

    # 4) Moment error
    plt.figure(figsize=(7.0, 4.5))
    for method, grp in df.groupby("method"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["rel_err_Mz"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("relative moment error")
    plt.title("Moment error: local centroid matters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "band_sheet_moment_error_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    out_dir = Path.cwd()
    figures_dir = out_dir / "result" / "figures"
    data_dir = out_dir / "result" / "data"

    kA, kB, delta = 30.0, 90.0, 0.10
    patch_center = (0.22, -0.10)
    patch_size = (1.20, 0.55)
    angle_deg = 28.0
    angle_rad = np.deg2rad(angle_deg)

    xmin, xmax = -0.70, 1.20
    zmin, zmax = -0.75, 0.55
    pad_y = 0.16

    patch_area_exact = patch_size[0] * patch_size[1]
    analytic = analytic_planar_contact(kA, kB, delta, patch_area_exact, patch_center[0], patch_center[1])

    # Demo grid for the local area-fraction / centroid picture
    demo = band_integral_planar_patch(
        18, 18, 18,
        kA=kA, kB=kB, delta=delta,
        xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, pad_y=pad_y,
        patch_center=patch_center, patch_size=patch_size, angle_rad=angle_rad,
        method="area_fraction_centroid"
    )

    rows = []
    for N in [12, 18, 24, 36, 48, 72, 96]:
        for method in ["cell_center", "area_fraction", "area_fraction_centroid"]:
            out = band_integral_planar_patch(
                N, N, N,
                kA=kA, kB=kB, delta=delta,
                xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, pad_y=pad_y,
                patch_center=patch_center, patch_size=patch_size, angle_rad=angle_rad,
                method=method
            )
            rows.append({
                "N": N,
                "method": method,
                "area_num": out["area_est"],
                "area_exact": analytic["area"],
                "rel_err_area": abs(out["area_est"] - analytic["area"]) / analytic["area"],
                "xbar_num": out["xbar_est"],
                "xbar_exact": analytic["xbar"],
                "abs_err_xbar": abs(out["xbar_est"] - analytic["xbar"]),
                "y_star_num": out["y_star_est"],
                "y_star_exact": analytic["y_star"],
                "abs_err_y_star": abs(out["y_star_est"] - analytic["y_star"]),
                "Fy_num": out["Fy"],
                "Fy_exact": analytic["Fy"],
                "rel_err_Fy": abs(out["Fy"] - analytic["Fy"]) / analytic["Fy"],
                "Mz_num": out["Mz"],
                "Mz_exact": analytic["Mz"],
                "rel_err_Mz": abs(out["Mz"] - analytic["Mz"]) / abs(analytic["Mz"]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "band_sheet_area_fraction_centroid_metrics.csv", index=False)
    make_plots(df, demo, figures_dir)

    print("解析真值:")
    print(analytic)
    print("\n明细表已保存到:")
    print(data_dir / "band_sheet_area_fraction_centroid_metrics.csv")
    print("\n图片已保存到:")
    print(figures_dir / "band_sheet_area_fraction_centroid_demo.png")
    print(figures_dir / "band_sheet_area_error_comparison.png")
    print(figures_dir / "band_sheet_force_error_comparison.png")
    print(figures_dir / "band_sheet_moment_error_comparison.png")


if __name__ == "__main__":
    main()

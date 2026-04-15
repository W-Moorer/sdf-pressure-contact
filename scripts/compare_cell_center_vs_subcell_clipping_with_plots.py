#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare cell-center patching vs subcell clipping, with built-in plotting.

This script generates:
    - cell_center_vs_subcell_clipping_comparison.png
    - cell_center_vs_subcell_clipping_area_error.png
    - cell_center_vs_subcell_clipping_centroid_error.png
    - cell_center_vs_subcell_clipping_moment_error.png
    - cell_center_vs_subcell_clipping_metrics.csv

Usage:
    python compare_cell_center_vs_subcell_clipping.py

It uses matplotlib's Agg backend, so it works on headless servers.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def rect_vertices(center, size, angle_rad):
    cx, cz = center
    lx, lz = size
    hx, hz = lx / 2.0, lz / 2.0
    local = np.array([
        [-hx, -hz],
        [ hx, -hz],
        [ hx,  hz],
        [-hx,  hz],
    ])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([cx, cz])


def point_in_rotated_rect(points, center, size, angle_rad):
    cx, cz = center
    lx, lz = size
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    Rt = np.array([[c, s], [-s, c]])  # inverse rotation
    local = (points - np.array([cx, cz])) @ Rt.T
    return (np.abs(local[:, 0]) <= lx / 2.0) & (np.abs(local[:, 1]) <= lz / 2.0)


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def line_intersection_with_clip_edge(p1, p2, a, b, eps=1e-14):
    r = p2 - p1
    s = b - a
    denom = cross2(r, s)
    if abs(denom) < eps:
        return p2.copy()
    t = cross2(a - p1, s) / denom
    return p1 + t * r


def clip_polygon_against_edge(poly, a, b, eps=1e-12):
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


def clip_square_by_convex_polygon(square_poly, clip_poly):
    poly = square_poly.copy()
    for i in range(len(clip_poly)):
        a = clip_poly[i]
        b = clip_poly[(i + 1) % len(clip_poly)]
        poly = clip_polygon_against_edge(poly, a, b)
        if len(poly) == 0:
            break
    return poly


def polygon_area_centroid(poly):
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


def make_grid(xmin, xmax, zmin, zmax, Nx, Nz):
    x_edges = np.linspace(xmin, xmax, Nx + 1)
    z_edges = np.linspace(zmin, zmax, Nz + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dx = x_edges[1] - x_edges[0]
    dz = z_edges[1] - z_edges[0]
    return x_edges, z_edges, x_centers, z_centers, dx, dz


def cell_center_patch_metrics(x_edges, z_edges, center, size, angle_rad, p_star):
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dx = x_edges[1] - x_edges[0]
    dz = z_edges[1] - z_edges[0]
    cell_area = dx * dz
    Xc, Zc = np.meshgrid(x_centers, z_centers, indexing="ij")
    pts = np.column_stack([Xc.ravel(), Zc.ravel()])
    inside = point_in_rotated_rect(pts, center, size, angle_rad).reshape(len(x_centers), len(z_centers)).astype(float)

    area = inside.sum() * cell_area
    if area > 0:
        xbar = (Xc * inside).sum() * cell_area / area
        zbar = (Zc * inside).sum() * cell_area / area
    else:
        xbar = np.nan
        zbar = np.nan
    Fy = p_star * area
    Mz = p_star * area * xbar if area > 0 else np.nan
    return inside, {"area": area, "xbar": xbar, "zbar": zbar, "Fy": Fy, "Mz": Mz}


def clipped_patch_metrics(x_edges, z_edges, rect_poly, p_star):
    Nx = len(x_edges) - 1
    Nz = len(z_edges) - 1
    frac = np.zeros((Nx, Nz), dtype=float)
    area_sum = 0.0
    first_moment_x = 0.0
    first_moment_z = 0.0
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
                area_sum += A
                first_moment_x += A * C[0]
                first_moment_z += A * C[1]

    xbar = first_moment_x / area_sum
    zbar = first_moment_z / area_sum
    Fy = p_star * area_sum
    Mz = p_star * area_sum * xbar
    return frac, {"area": area_sum, "xbar": xbar, "zbar": zbar, "Fy": Fy, "Mz": Mz}


def make_plots(df, x_edges, z_edges, cc_frac, clip_frac, rect_poly, cc_metrics, clip_metrics, exact, out_dir):
    # 1) Side-by-side geometry comparison in a single image
    comparison_png = out_dir / "cell_center_vs_subcell_clipping_comparison.png"
    xmin, xmax = x_edges[0], x_edges[-1]
    zmin, zmax = z_edges[0], z_edges[-1]
    gap = 0.35
    shift = (xmax - xmin) + gap

    plt.figure(figsize=(12, 5.2))
    ax = plt.gca()

    Xe1, Ze1 = np.meshgrid(x_edges, z_edges, indexing="ij")
    mesh = ax.pcolormesh(Xe1, Ze1, cc_frac, shading="flat", alpha=0.95)

    Xe2, Ze2 = np.meshgrid(x_edges + shift, z_edges, indexing="ij")
    ax.pcolormesh(Xe2, Ze2, clip_frac, shading="flat", alpha=0.95)

    poly_closed = np.vstack([rect_poly, rect_poly[0]])
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], linewidth=2.0)
    ax.plot(poly_closed[:, 0] + shift, poly_closed[:, 1], linewidth=2.0)

    ax.plot([exact["xbar"]], [exact["zbar"]], marker="x", markersize=10)
    ax.plot([exact["xbar"] + shift], [exact["zbar"]], marker="x", markersize=10)
    ax.plot([cc_metrics["xbar"]], [cc_metrics["zbar"]], marker="o", markersize=6)
    ax.plot([clip_metrics["xbar"] + shift], [clip_metrics["zbar"]], marker="o", markersize=6)

    ax.text(0.5 * (xmin + xmax), zmax + 0.06, "cell-center patch", ha="center", va="bottom", fontsize=12)
    ax.text(0.5 * (xmin + xmax) + shift, zmax + 0.06, "subcell quadrature / clipping", ha="center", va="bottom", fontsize=12)
    ax.axvline(xmax + 0.5 * gap, linestyle="--", linewidth=1.0)

    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax + shift)
    ax.set_ylim(zmin, zmax + 0.12)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Same grid, same exact patch, different patch integration")
    plt.colorbar(mesh, ax=ax, shrink=0.88, label="cell area fraction inside exact patch")
    plt.tight_layout()
    plt.savefig(comparison_png, dpi=180)
    plt.close()

    # 2) Area error
    plt.figure(figsize=(7.0, 4.5))
    for method, grp in df.groupby("method"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["rel_err_area"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("relative area error")
    plt.title("Area error: cell-center vs subcell clipping")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cell_center_vs_subcell_clipping_area_error.png", dpi=180)
    plt.close()

    # 3) Centroid error
    plt.figure(figsize=(7.0, 4.5))
    for method, grp in df.groupby("method"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["abs_err_xbar"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("absolute centroid error in x")
    plt.title("Centroid error: cell-center vs subcell clipping")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cell_center_vs_subcell_clipping_centroid_error.png", dpi=180)
    plt.close()

    # 4) Moment error
    plt.figure(figsize=(7.0, 4.5))
    for method, grp in df.groupby("method"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["rel_err_Mz"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("relative moment error")
    plt.title("Moment error: cell-center vs subcell clipping")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cell_center_vs_subcell_clipping_moment_error.png", dpi=180)
    plt.close()


def main():
    out_dir = Path.cwd()
    figures_dir = out_dir / "result" / "figures"
    data_dir = out_dir / "result" / "data"

    # Exact benchmark
    kA, kB, delta = 30.0, 90.0, 0.10
    p_star = (kA * kB / (kA + kB)) * delta

    center = (0.22, -0.10)
    size = (1.20, 0.55)
    angle_deg = 28.0
    angle_rad = np.deg2rad(angle_deg)

    rect_poly = rect_vertices(center, size, angle_rad)
    exact = {
        "area": size[0] * size[1],
        "xbar": center[0],
        "zbar": center[1],
    }
    exact["Fy"] = p_star * exact["area"]
    exact["Mz"] = p_star * exact["area"] * exact["xbar"]

    # Coarse visualization grid
    xmin, xmax = -0.70, 1.20
    zmin, zmax = -0.75, 0.55
    Nx_vis, Nz_vis = 18, 18
    x_edges, z_edges, _, _, _, _ = make_grid(xmin, xmax, zmin, zmax, Nx_vis, Nz_vis)
    cc_frac, cc_metrics = cell_center_patch_metrics(x_edges, z_edges, center, size, angle_rad, p_star)
    clip_frac, clip_metrics = clipped_patch_metrics(x_edges, z_edges, rect_poly, p_star)

    # Convergence table
    rows = []
    for N in [12, 18, 24, 36, 48, 72, 96]:
        x_edges_N, z_edges_N, _, _, _, _ = make_grid(xmin, xmax, zmin, zmax, N, N)
        _, cc = cell_center_patch_metrics(x_edges_N, z_edges_N, center, size, angle_rad, p_star)
        _, cl = clipped_patch_metrics(x_edges_N, z_edges_N, rect_poly, p_star)
        for method, m in [("cell_center", cc), ("subcell_clipping", cl)]:
            rows.append({
                "N": N,
                "method": method,
                "area_num": m["area"],
                "area_exact": exact["area"],
                "rel_err_area": abs(m["area"] - exact["area"]) / exact["area"],
                "xbar_num": m["xbar"],
                "xbar_exact": exact["xbar"],
                "abs_err_xbar": abs(m["xbar"] - exact["xbar"]),
                "Fy_num": m["Fy"],
                "Fy_exact": exact["Fy"],
                "rel_err_Fy": abs(m["Fy"] - exact["Fy"]) / exact["Fy"],
                "Mz_num": m["Mz"],
                "Mz_exact": exact["Mz"],
                "rel_err_Mz": abs(m["Mz"] - exact["Mz"]) / abs(exact["Mz"]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "cell_center_vs_subcell_clipping_metrics.csv", index=False)
    make_plots(df, x_edges, z_edges, cc_frac, clip_frac, rect_poly, cc_metrics, clip_metrics, exact, figures_dir)

    print("粗网格可视化（18x18）:")
    print(f"  cell-center:   area={cc_metrics['area']:.6f}, xbar={cc_metrics['xbar']:.6f}, Fy={cc_metrics['Fy']:.6f}, Mz={cc_metrics['Mz']:.6f}")
    print(f"  subcell-clip:  area={clip_metrics['area']:.6f}, xbar={clip_metrics['xbar']:.6f}, Fy={clip_metrics['Fy']:.6f}, Mz={clip_metrics['Mz']:.6f}")
    print(f"  exact:         area={exact['area']:.6f}, xbar={exact['xbar']:.6f}, Fy={exact['Fy']:.6f}, Mz={exact['Mz']:.6f}")
    print("\n已保存:")
    print(data_dir / "cell_center_vs_subcell_clipping_metrics.csv")
    print(figures_dir / "cell_center_vs_subcell_clipping_comparison.png")
    print(figures_dir / "cell_center_vs_subcell_clipping_area_error.png")
    print(figures_dir / "cell_center_vs_subcell_clipping_centroid_error.png")
    print(figures_dir / "cell_center_vs_subcell_clipping_moment_error.png")


if __name__ == "__main__":
    main()

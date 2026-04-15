#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upgrade the band-integral prototype by replacing y-direction single-point sampling
with:
    1) 1D subcell Gauss quadrature
    2) analytic y-cell band integration

This script assumes the x-z patch geometry is already handled with:
    area fraction + local centroid

So the remaining force error is dominated by y-direction band integration.

Outputs:
    - y_band_integration_metrics.csv
    - y_band_force_error_comparison.png
    - y_band_moment_error_comparison.png
    - y_band_sheet_error_comparison.png
    - y_band_integrand_profile_demo.png

Usage:
    python y_band_integration_with_plots.py
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


def clipped_patch_geometry(x_edges: np.ndarray, z_edges: np.ndarray, rect_poly: np.ndarray):
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

    area_exact = float(np.sum(frac) * cell_area)
    xbar_exact = float(np.sum(frac * cx) * cell_area / area_exact)
    zbar_exact = float(np.sum(frac * cz) * cell_area / area_exact)
    return frac, cx, cz, area_exact, xbar_exact, zbar_exact


def band_fields(y: np.ndarray, *, kA: float, kB: float, delta: float):
    pA = kA * np.clip(-y, 0.0, None)
    pB = kB * np.clip(y + delta, 0.0, None)
    pbar = 0.5 * (pA + pB)
    h = pA - pB
    return pA, pB, pbar, h


def integrate_y_cell_center(yc: float, dy: float, *, kA: float, kB: float, delta: float, eta_h: float):
    pA, pB, pbar, h = band_fields(np.array([yc]), kA=kA, kB=kB, delta=delta)
    val_force = pbar[0] * delta_cosine(h, eta_h)[0] * (kA + kB) * dy
    val_area  = delta_cosine(h, eta_h)[0] * (kA + kB) * dy
    val_yarea = yc * delta_cosine(h, eta_h)[0] * (kA + kB) * dy
    return val_force, val_area, val_yarea


def integrate_y_gauss(yl: float, yr: float, *, kA: float, kB: float, delta: float, eta_h: float, order: int = 4):
    xi, wi = np.polynomial.legendre.leggauss(order)
    y = 0.5 * (yr - yl) * xi + 0.5 * (yr + yl)
    jac = 0.5 * (yr - yl)
    pA, pB, pbar, h = band_fields(y, kA=kA, kB=kB, delta=delta)
    delta_vals = delta_cosine(h, eta_h)
    band_weight = delta_vals * (kA + kB)
    val_force = np.sum(wi * pbar * band_weight) * jac
    val_area  = np.sum(wi * band_weight) * jac
    val_yarea = np.sum(wi * y * band_weight) * jac
    return val_force, val_area, val_yarea


def _clip_interval(a: float, b: float, lo: float, hi: float):
    aa = max(a, lo)
    bb = min(b, hi)
    if bb <= aa:
        return None
    return aa, bb


def _antiderivative_delta(s: float, eta: float):
    beta = np.pi / eta
    return 0.5 / eta * (s + np.sin(beta * s) / beta)


def _antiderivative_s_delta(s: float, eta: float):
    beta = np.pi / eta
    return 0.5 / eta * (0.5 * s * s + s * np.sin(beta * s) / beta + np.cos(beta * s) / (beta * beta))


def _antiderivative_force_integrand_s(s: float, eta: float, p_star: float, alpha: float):
    beta = np.pi / eta
    return 0.5 / eta * (
        p_star * s
        - 0.5 * alpha * s * s
        + p_star * np.sin(beta * s) / beta
        - alpha * (s * np.sin(beta * s) / beta + np.cos(beta * s) / (beta * beta))
    )


def integrate_y_analytic(yl: float, yr: float, *, kA: float, kB: float, delta: float, eta_h: float):
    K = kA + kB
    c = kB * delta
    p_star = (kA * kB / K) * delta
    alpha = (kB - kA) / (2.0 * K)

    dom = _clip_interval(yl, yr, -delta, 0.0)
    if dom is None:
        return 0.0, 0.0, 0.0
    yl2, yr2 = dom

    s_lo = -K * yr2 - c
    s_hi = -K * yl2 - c
    dom_s = _clip_interval(s_lo, s_hi, -eta_h, eta_h)
    if dom_s is None:
        return 0.0, 0.0, 0.0
    a, b = dom_s

    force = _antiderivative_force_integrand_s(b, eta_h, p_star, alpha) - _antiderivative_force_integrand_s(a, eta_h, p_star, alpha)
    area = _antiderivative_delta(b, eta_h) - _antiderivative_delta(a, eta_h)
    yarea = -(1.0 / K) * (
        (_antiderivative_s_delta(b, eta_h) - _antiderivative_s_delta(a, eta_h))
        + c * (_antiderivative_delta(b, eta_h) - _antiderivative_delta(a, eta_h))
    )
    return force, area, yarea


def run_band_integral(
    N: int,
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
    y_method: str = "cell_center",
    eta_factor: float = 2.0,
    gauss_order: int = 4,
):
    x_edges = np.linspace(xmin, xmax, N + 1)
    z_edges = np.linspace(zmin, zmax, N + 1)
    ymin = -delta - pad_y
    ymax = pad_y
    y_edges = np.linspace(ymin, ymax, N + 1)

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    dz = z_edges[1] - z_edges[0]
    cell_area = dx * dz

    rect_poly = rect_vertices(patch_center, patch_size, angle_rad)
    frac, cx, cz, area_exact_patch, xbar_exact_patch, zbar_exact_patch = clipped_patch_geometry(x_edges, z_edges, rect_poly)

    K = kA + kB
    eta_h = K * eta_factor * dy

    force_y = np.zeros(N, dtype=float)
    area_y = np.zeros(N, dtype=float)
    yarea_y = np.zeros(N, dtype=float)
    for m in range(N):
        yl, yr = y_edges[m], y_edges[m + 1]
        yc = 0.5 * (yl + yr)
        if y_method == "cell_center":
            f, a, ya = integrate_y_cell_center(yc, dy, kA=kA, kB=kB, delta=delta, eta_h=eta_h)
        elif y_method == "gauss4":
            f, a, ya = integrate_y_gauss(yl, yr, kA=kA, kB=kB, delta=delta, eta_h=eta_h, order=gauss_order)
        elif y_method == "analytic":
            f, a, ya = integrate_y_analytic(yl, yr, kA=kA, kB=kB, delta=delta, eta_h=eta_h)
        else:
            raise ValueError(f"Unknown y_method: {y_method}")
        force_y[m] = f
        area_y[m] = a
        yarea_y[m] = ya

    Acol = frac * cell_area
    Fy = np.sum(Acol) * np.sum(force_y)
    A_band = np.sum(Acol) * np.sum(area_y)
    Y_band = np.sum(Acol) * np.sum(yarea_y)

    col_force_scalar = np.sum(force_y) * Acol
    Mz = np.sum(cx * col_force_scalar)
    Mx = np.sum(-cz * col_force_scalar)

    xbar = np.sum(cx * Acol) / np.sum(Acol)
    zbar = np.sum(cz * Acol) / np.sum(Acol)
    ybar = Y_band / A_band if A_band > 0 else np.nan

    return {
        "N": N,
        "dy": dy,
        "eta_h": eta_h,
        "Fy": Fy,
        "Mx": Mx,
        "Mz": Mz,
        "area_band": A_band,
        "xbar": xbar,
        "zbar": zbar,
        "ybar": ybar,
    }


def make_demo_integrand_plot(out_dir: Path, *, kA: float, kB: float, delta: float, N_demo: int, pad_y: float, eta_factor: float = 2.0):
    ymin = -delta - pad_y
    ymax = pad_y
    y_edges = np.linspace(ymin, ymax, N_demo + 1)
    dy = y_edges[1] - y_edges[0]
    K = kA + kB
    eta_h = K * eta_factor * dy

    ys = np.linspace(ymin, ymax, 2000)
    _, _, pbar, h = band_fields(ys, kA=kA, kB=kB, delta=delta)
    exact_integrand = pbar * delta_cosine(h, eta_h) * K

    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    center_vals = []
    gauss_vals = []
    analytic_vals = []
    for m in range(N_demo):
        yl, yr = y_edges[m], y_edges[m + 1]
        yc = 0.5 * (yl + yr)
        f0, _, _ = integrate_y_cell_center(yc, dy, kA=kA, kB=kB, delta=delta, eta_h=eta_h)
        f1, _, _ = integrate_y_gauss(yl, yr, kA=kA, kB=kB, delta=delta, eta_h=eta_h, order=4)
        f2, _, _ = integrate_y_analytic(yl, yr, kA=kA, kB=kB, delta=delta, eta_h=eta_h)
        center_vals.append(f0 / dy)
        gauss_vals.append(f1 / dy)
        analytic_vals.append(f2 / dy)

    plt.figure(figsize=(7.3, 4.8))
    plt.plot(ys, exact_integrand, linewidth=2.0, label="continuous integrand")
    plt.step(y_centers, center_vals, where="mid", label="cell-center / dy")
    plt.step(y_centers, gauss_vals, where="mid", label="gauss4 / dy")
    plt.step(y_centers, analytic_vals, where="mid", label="analytic / dy")
    plt.axvline(-kB * delta / (kA + kB), linestyle="--", linewidth=1.0, label="exact sheet y*")
    plt.xlabel("y")
    plt.ylabel("band force density in y")
    plt.title("Why y-direction single-point sampling leaves residual Fy error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "y_band_integrand_profile_demo.png", dpi=180)
    plt.close()


def main():
    out_dir = Path.cwd()
    figures_dir = out_dir / "result" / "figures"
    data_dir = out_dir / "result" / "data"

    kA, kB, delta = 30.0, 90.0, 0.10
    patch_center = (0.22, -0.10)
    patch_size = (1.20, 0.55)
    angle_rad = np.deg2rad(28.0)

    xmin, xmax = -0.70, 1.20
    zmin, zmax = -0.75, 0.55
    pad_y = 0.16

    x_edges_ref = np.linspace(xmin, xmax, 401)
    z_edges_ref = np.linspace(zmin, zmax, 401)
    rect_poly = rect_vertices(patch_center, patch_size, angle_rad)
    _, _, _, patch_area_exact, xbar_exact, zbar_exact = clipped_patch_geometry(x_edges_ref, z_edges_ref, rect_poly)
    exact = analytic_planar_contact(kA, kB, delta, patch_area_exact, xbar_exact, zbar_exact)

    rows = []
    for N in [12, 18, 24, 36, 48, 72, 96]:
        for method in ["cell_center", "gauss4", "analytic"]:
            out = run_band_integral(
                N,
                kA=kA, kB=kB, delta=delta,
                xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, pad_y=pad_y,
                patch_center=patch_center, patch_size=patch_size, angle_rad=angle_rad,
                y_method=method, eta_factor=2.0, gauss_order=4
            )
            rows.append({
                "N": N,
                "y_method": method,
                "Fy_num": out["Fy"],
                "Fy_exact": exact["Fy"],
                "rel_err_Fy": abs(out["Fy"] - exact["Fy"]) / exact["Fy"],
                "Mz_num": out["Mz"],
                "Mz_exact": exact["Mz"],
                "rel_err_Mz": abs(out["Mz"] - exact["Mz"]) / abs(exact["Mz"]),
                "ybar_num": out["ybar"],
                "ybar_exact": exact["y_star"],
                "abs_err_ybar": abs(out["ybar"] - exact["y_star"]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "y_band_integration_metrics.csv", index=False)

    plt.figure(figsize=(7.0, 4.5))
    for method, grp in df.groupby("y_method"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["rel_err_Fy"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("relative Fy error")
    plt.title("Upgrading y-direction integration reduces residual force error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "y_band_force_error_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    for method, grp in df.groupby("y_method"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["rel_err_Mz"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("relative Mz error")
    plt.title("Moment error follows force once x-z geometry is exact")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "y_band_moment_error_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    for method, grp in df.groupby("y_method"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["abs_err_ybar"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("absolute y* error")
    plt.title("Recovered sheet location vs y-direction integration method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "y_band_sheet_error_comparison.png", dpi=180)
    plt.close()

    make_demo_integrand_plot(figures_dir, kA=kA, kB=kB, delta=delta, N_demo=18, pad_y=pad_y, eta_factor=2.0)

    print("Exact reference:")
    print(exact)
    print("\nSaved:")
    print(data_dir / "y_band_integration_metrics.csv")
    print(figures_dir / "y_band_force_error_comparison.png")
    print(figures_dir / "y_band_moment_error_comparison.png")
    print(figures_dir / "y_band_sheet_error_comparison.png")
    print(figures_dir / "y_band_integrand_profile_demo.png")


if __name__ == "__main__":
    main()

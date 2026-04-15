#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-column force accumulator for SDF pressure-field contact.

Goal:
    Move the previous "analytic y-cell integration" into an engineering-style
    per-column / per-voxel accumulator.

Main idea:
    1) x-z patch geometry is represented per column by:
         alpha_ij   : area fraction of the column inside the exact patch
         c_ij=(cx,cz): local centroid of that clipped area
    2) y-direction band integration is preintegrated once per y-cell:
         I_force[m] = \int_{y_m^-}^{y_m^+} pbar(y) delta_eta(h(y)) |dh/dy| dy
         I_area[m]  = \int_{y_m^-}^{y_m^+} delta_eta(h(y)) |dh/dy| dy
         I_yarea[m] = \int_{y_m^-}^{y_m^+} y delta_eta(h(y)) |dh/dy| dy
    3) each column accumulates:
         Fcol_ij = alpha_ij * dx * dz * sum_m I_force[m]
    4) total wrench:
         Fy = sum_ij Fcol_ij
         Mx = -sum_ij cz_ij * Fcol_ij
         Mz =  sum_ij cx_ij * Fcol_ij

This script saves:
    - per_column_force_accumulator_demo.py
    - per_column_force_accumulator_metrics.csv
    - per_column_force_accumulator_force_map.png
    - per_column_force_accumulator_error_comparison.png
    - per_column_force_accumulator_timing_comparison.png

Usage:
    python per_column_force_accumulator_demo.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time


# -----------------------------
# Common math
# -----------------------------
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


def band_fields(y: np.ndarray, *, kA: float, kB: float, delta: float):
    pA = kA * np.clip(-y, 0.0, None)
    pB = kB * np.clip(y + delta, 0.0, None)
    pbar = 0.5 * (pA + pB)
    h = pA - pB
    return pA, pB, pbar, h


# -----------------------------
# x-z clipping / column geometry
# -----------------------------
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

    return frac, cx, cz


# -----------------------------
# Analytic y-cell integration
# -----------------------------
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


def integrate_y_cell_analytic(yl: float, yr: float, *, kA: float, kB: float, delta: float, eta_h: float):
    """
    Exact y-cell band integral for the planar benchmark.
    """
    K = kA + kB
    c = kB * delta
    p_star = (kA * kB / K) * delta
    alpha = (kB - kA) / (2.0 * K)

    dom = _clip_interval(yl, yr, -delta, 0.0)
    if dom is None:
        return 0.0, 0.0, 0.0
    yl2, yr2 = dom

    # s(y) = -K*y - c
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


def precompute_y_integrals(y_edges: np.ndarray, *, kA: float, kB: float, delta: float, eta_factor: float = 2.0):
    dy = y_edges[1] - y_edges[0]
    eta_h = (kA + kB) * eta_factor * dy
    N = len(y_edges) - 1
    I_force = np.zeros(N, dtype=float)
    I_area = np.zeros(N, dtype=float)
    I_yarea = np.zeros(N, dtype=float)
    for m in range(N):
        yl, yr = y_edges[m], y_edges[m + 1]
        I_force[m], I_area[m], I_yarea[m] = integrate_y_cell_analytic(
            yl, yr, kA=kA, kB=kB, delta=delta, eta_h=eta_h
        )
    return {
        "eta_h": eta_h,
        "I_force": I_force,
        "I_area": I_area,
        "I_yarea": I_yarea,
    }


# -----------------------------
# Accumulators
# -----------------------------
def accumulate_wrench_from_columns(
    frac: np.ndarray,
    cx: np.ndarray,
    cz: np.ndarray,
    dx: float,
    dz: float,
    y_pre: dict,
):
    """
    Engineering-style per-column accumulator.

    Each x-z column uses:
        column_area = alpha_ij * dx * dz
        column_force = column_area * sum_m I_force[m]
    """
    Acol = frac * dx * dz
    force_per_area = float(np.sum(y_pre["I_force"]))
    area_measure_per_area = float(np.sum(y_pre["I_area"]))
    yarea_per_area = float(np.sum(y_pre["I_yarea"]))

    Fcol = Acol * force_per_area
    Fy = float(np.sum(Fcol))
    Mx = float(np.sum(-cz * Fcol))
    Mz = float(np.sum(cx * Fcol))

    A_band = float(np.sum(Acol) * area_measure_per_area)
    Y_band = float(np.sum(Acol) * yarea_per_area)
    ybar = Y_band / A_band if A_band > 0 else np.nan

    xbar = float(np.sum(cx * Acol) / np.sum(Acol))
    zbar = float(np.sum(cz * Acol) / np.sum(Acol))

    return {
        "Fy": Fy,
        "Mx": Mx,
        "Mz": Mz,
        "ybar": ybar,
        "xbar": xbar,
        "zbar": zbar,
        "A_band": A_band,
        "Fcol": Fcol,
    }


def accumulate_wrench_naive_voxel(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    frac: np.ndarray,
    cx: np.ndarray,
    cz: np.ndarray,
    *,
    kA: float,
    kB: float,
    delta: float,
    eta_factor: float = 2.0,
):
    """
    Reference-style implementation:
        explicit triple sum over (i,j,m)
    but still uses exact x-z area fraction and exact local centroid.
    """
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    dz = z_edges[1] - z_edges[0]
    cell_area = dx * dz
    eta_h = (kA + kB) * eta_factor * dy

    N = len(y_edges) - 1
    Fy = 0.0
    Mx = 0.0
    Mz = 0.0
    A_band = 0.0
    Y_band = 0.0

    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    for m in range(N):
        yc = y_centers[m]
        pA, pB, pbar, h = band_fields(np.array([yc]), kA=kA, kB=kB, delta=delta)
        w = delta_cosine(h, eta_h)[0] * (kA + kB) * dy
        dF_per_area = pbar[0] * w
        dA_per_area = w
        dYA_per_area = yc * w

        dA_cols = frac * cell_area
        dF_cols = dA_cols * dF_per_area

        Fy += np.sum(dF_cols)
        Mx += np.sum(-cz * dF_cols)
        Mz += np.sum(cx * dF_cols)
        A_band += np.sum(dA_cols) * dA_per_area
        Y_band += np.sum(dA_cols) * dYA_per_area

    ybar = Y_band / A_band if A_band > 0 else np.nan
    xbar = float(np.sum(cx * frac * cell_area) / np.sum(frac * cell_area))
    zbar = float(np.sum(cz * frac * cell_area) / np.sum(frac * cell_area))

    return {
        "Fy": float(Fy),
        "Mx": float(Mx),
        "Mz": float(Mz),
        "ybar": float(ybar),
        "xbar": xbar,
        "zbar": zbar,
        "A_band": float(A_band),
    }


# -----------------------------
# Demo / benchmark
# -----------------------------
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

    # High-resolution geometric reference for the exact patch moments
    x_edges_ref = np.linspace(xmin, xmax, 401)
    z_edges_ref = np.linspace(zmin, zmax, 401)
    rect_poly_ref = rect_vertices(patch_center, patch_size, angle_rad)
    frac_ref, cx_ref, cz_ref = clipped_patch_geometry(x_edges_ref, z_edges_ref, rect_poly_ref)
    dx_ref = x_edges_ref[1] - x_edges_ref[0]
    dz_ref = z_edges_ref[1] - z_edges_ref[0]
    A_ref = np.sum(frac_ref) * dx_ref * dz_ref
    xbar_ref = np.sum(cx_ref * frac_ref) * dx_ref * dz_ref / A_ref
    zbar_ref = np.sum(cz_ref * frac_ref) * dx_ref * dz_ref / A_ref
    exact = analytic_planar_contact(kA, kB, delta, A_ref, xbar_ref, zbar_ref)

    rows = []

    # One force map demo at moderate resolution
    N_demo = 36
    x_edges = np.linspace(xmin, xmax, N_demo + 1)
    z_edges = np.linspace(zmin, zmax, N_demo + 1)
    y_edges = np.linspace(-delta - pad_y, pad_y, N_demo + 1)
    frac, cx, cz = clipped_patch_geometry(x_edges, z_edges, rect_vertices(patch_center, patch_size, angle_rad))
    dx = x_edges[1] - x_edges[0]
    dz = z_edges[1] - z_edges[0]
    y_pre = precompute_y_integrals(y_edges, kA=kA, kB=kB, delta=delta, eta_factor=2.0)
    demo = accumulate_wrench_from_columns(frac, cx, cz, dx, dz, y_pre)

    # Save force map figure
    plt.figure(figsize=(7.2, 5.8))
    Xe, Ze = np.meshgrid(x_edges, z_edges, indexing="ij")
    plt.pcolormesh(Xe, Ze, demo["Fcol"], shading="flat")
    poly = rect_vertices(patch_center, patch_size, angle_rad)
    poly = np.vstack([poly, poly[0]])
    plt.plot(poly[:, 0], poly[:, 1], linewidth=2.0)
    plt.scatter([exact["xbar"]], [exact["zbar"]], marker="x", s=80, label="exact centroid")
    plt.scatter([demo["xbar"]], [demo["zbar"]], marker="o", s=35, label="accumulator centroid")
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Per-column force contribution map")
    plt.colorbar(label="column force contribution to Fy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "per_column_force_accumulator_force_map.png", dpi=180)
    plt.close()

    # Accuracy + timing benchmark
    for N in [12, 18, 24, 36, 48, 72, 96]:
        x_edges = np.linspace(xmin, xmax, N + 1)
        z_edges = np.linspace(zmin, zmax, N + 1)
        y_edges = np.linspace(-delta - pad_y, pad_y, N + 1)
        dx = x_edges[1] - x_edges[0]
        dz = z_edges[1] - z_edges[0]

        rect_poly = rect_vertices(patch_center, patch_size, angle_rad)
        frac, cx, cz = clipped_patch_geometry(x_edges, z_edges, rect_poly)

        # per-column accumulator
        t0 = time.perf_counter()
        y_pre = precompute_y_integrals(y_edges, kA=kA, kB=kB, delta=delta, eta_factor=2.0)
        out_col = accumulate_wrench_from_columns(frac, cx, cz, dx, dz, y_pre)
        t1 = time.perf_counter()

        # naive triple-loop style reference
        t2 = time.perf_counter()
        out_naive = accumulate_wrench_naive_voxel(
            x_edges, y_edges, z_edges, frac, cx, cz,
            kA=kA, kB=kB, delta=delta, eta_factor=2.0
        )
        t3 = time.perf_counter()

        for method, out, dt in [
            ("per_column_accumulator", out_col, t1 - t0),
            ("naive_voxel_sum", out_naive, t3 - t2),
        ]:
            rows.append({
                "N": N,
                "method": method,
                "time_sec": dt,
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
    df.to_csv(data_dir / "per_column_force_accumulator_metrics.csv", index=False)

    # Error comparison
    plt.figure(figsize=(7.0, 4.5))
    for method, grp in df.groupby("method"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["rel_err_Fy"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("relative Fy error")
    plt.title("Per-column accumulator vs naive voxel sum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "per_column_force_accumulator_error_comparison.png", dpi=180)
    plt.close()

    # Timing comparison
    plt.figure(figsize=(7.0, 4.5))
    for method, grp in df.groupby("method"):
        grp = grp.sort_values("N")
        plt.plot(grp["N"], grp["time_sec"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("grid resolution N")
    plt.ylabel("runtime [s]")
    plt.title("Why preintegrated column accumulation is attractive")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "per_column_force_accumulator_timing_comparison.png", dpi=180)
    plt.close()

    print("Exact reference:")
    print(exact)
    print("\nDemo accumulator at N=36:")
    print({k: demo[k] for k in ["Fy", "Mx", "Mz", "xbar", "zbar", "ybar"]})
    print("\nSaved:")
    print(data_dir / "per_column_force_accumulator_metrics.csv")
    print(figures_dir / "per_column_force_accumulator_force_map.png")
    print(figures_dir / "per_column_force_accumulator_error_comparison.png")
    print(figures_dir / "per_column_force_accumulator_timing_comparison.png")


if __name__ == "__main__":
    main()

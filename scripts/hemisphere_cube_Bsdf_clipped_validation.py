#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized implementation of:
    SDF top-slice contour extraction
    -> x-z cell-fraction / contour clipping
    -> local footprint centroid
    -> local-normal root solve
    -> local-normal 3D accumulator

Benchmark:
    rigid hemisphere pressing shallowly into the top face of a compliant cube

Methods compared:
    A_direct_band
    B_analytic_sheet
    B_sdf_column_center
    B_sdf_clipped_sheet   (optimized)

Outputs:
    - hemisphere_cube_Bsdf_clipped_results.csv
    - hemisphere_cube_Bsdf_clipped_shallow_force_compare.png
    - hemisphere_cube_Bsdf_clipped_convergence.png
    - hemisphere_cube_Bsdf_clipped_error_vs_runtime.png
    - hemisphere_cube_Bsdf_clipped_symmetry_residual.png
    - hemisphere_cube_Bsdf_clipped_geometry_demo.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time


# -----------------------------
# Shared benchmark helpers
# -----------------------------
def delta_cosine(s: np.ndarray, eta: float) -> np.ndarray:
    a = np.abs(s)
    out = np.zeros_like(s, dtype=float)
    mask = a <= eta
    out[mask] = 0.5 / eta * (1.0 + np.cos(np.pi * s[mask] / eta))
    return out


def exact_cap_volume(R: float, delta: float) -> float:
    return np.pi * delta * delta * (R - delta / 3.0)


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
# Method A: direct voxel band
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
# Polygon utilities
# -----------------------------
def polygon_signed_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    xp = np.roll(x, -1)
    yp = np.roll(y, -1)
    return 0.5 * np.sum(x * yp - xp * y)


def polygon_area_centroid(poly: np.ndarray):
    if len(poly) < 3:
        return 0.0, np.array([np.nan, np.nan])
    x = poly[:, 0]
    y = poly[:, 1]
    xp = np.roll(x, -1)
    yp = np.roll(y, -1)
    cross = x * yp - xp * y
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-15:
        return 0.0, np.array([np.nan, np.nan])
    Cx = np.sum((x + xp) * cross) / (6.0 * A)
    Cy = np.sum((y + yp) * cross) / (6.0 * A)
    return abs(A), np.array([Cx, Cy])


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return a[0] * b[1] - a[1] * b[0]


def line_intersection_with_clip_edge(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-14):
    r = p2 - p1
    s = b - a
    denom = cross2(r, s)
    if abs(denom) < eps:
        return p2.copy()
    t = cross2(a - p1, s) / denom
    return p1 + t * r


def clip_polygon_against_edge(poly: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-12):
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


def clip_square_by_convex_polygon(square_poly: np.ndarray, clip_poly: np.ndarray):
    poly = square_poly.copy()
    for i in range(len(clip_poly)):
        a = clip_poly[i]
        b = clip_poly[(i + 1) % len(clip_poly)]
        poly = clip_polygon_against_edge(poly, a, b)
        if len(poly) == 0:
            break
    return poly


def point_in_convex_polygon(pt: np.ndarray, poly: np.ndarray, eps: float = 1e-12) -> bool:
    # poly is CCW
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        if cross2(b - a, pt - a) < -eps:
            return False
    return True


def resample_closed_polygon(poly: np.ndarray, M: int) -> np.ndarray:
    if len(poly) < 3:
        return poly
    closed = np.vstack([poly, poly[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0.0:
        return poly
    targets = np.linspace(0.0, total, M + 1)[:-1]
    out = []
    j = 0
    for t in targets:
        while j + 1 < len(s) and s[j + 1] < t:
            j += 1
        a = closed[j]
        b = closed[j + 1]
        ds = s[j + 1] - s[j]
        w = 0.0 if ds <= 0 else (t - s[j]) / ds
        out.append((1 - w) * a + w * b)
    out = np.array(out)
    if polygon_signed_area(out) < 0.0:
        out = out[::-1]
    return out


# -----------------------------
# Automatic top-slice contour extraction from SDF
# -----------------------------
def extract_top_slice_contour_polygon(
    *,
    R: float,
    delta: float,
    cube_size: float,
    Ncontour: int = 400,
    Mpoly: int = 96,
):
    x_min, x_max = -cube_size / 2.0, cube_size / 2.0
    z_min, z_max = -cube_size / 2.0, cube_size / 2.0
    xs = np.linspace(x_min, x_max, Ncontour)
    zs = np.linspace(z_min, z_max, Ncontour)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    phi = sphere_phi_xyz(X, 0.0, Z, R=R, delta=delta)

    fig, ax = plt.subplots()
    cs = ax.contour(xs, zs, phi, levels=[0.0])
    segs = cs.allsegs[0]
    plt.close(fig)

    if not segs:
        return np.zeros((0, 2), dtype=float)

    poly = max(segs, key=lambda arr: arr.shape[0]).copy()
    if np.linalg.norm(poly[0] - poly[-1]) > 1e-12:
        poly = np.vstack([poly, poly[0]])
    poly = poly[:-1]
    if polygon_signed_area(poly) < 0.0:
        poly = poly[::-1]
    poly = resample_closed_polygon(poly, Mpoly)
    return poly


# -----------------------------
# Method B2: old column-center SDF extraction
# -----------------------------
def version_B_sdf_column_center(
    *,
    R: float,
    delta: float,
    k: float,
    cube_size: float,
    cube_height: float,
    Nxz: int,
    bisection_steps: int = 30,
):
    x_min, x_max = -cube_size / 2.0, cube_size / 2.0
    z_min, z_max = -cube_size / 2.0, cube_size / 2.0

    xs = np.linspace(x_min, x_max, Nxz, endpoint=False) + (x_max - x_min) / Nxz / 2.0
    zs = np.linspace(z_min, z_max, Nxz, endpoint=False) + (z_max - z_min) / Nxz / 2.0
    dx = (x_max - x_min) / Nxz
    dz = (z_max - z_min) / Nxz

    Fx = Fy = Fz = 0.0
    Mx = My = Mz = 0.0

    num_elems = 0
    for x in xs:
        for z in zs:
            phi_top = sphere_phi_xyz(x, 0.0, z, R=R, delta=delta)
            phi_bot = sphere_phi_xyz(x, -cube_height, z, R=R, delta=delta)
            if not (phi_top <= 0.0 and phi_bot >= 0.0):
                continue

            yl = -cube_height
            yr = 0.0
            for _ in range(bisection_steps):
                ym = 0.5 * (yl + yr)
                fm = sphere_phi_xyz(x, ym, z, R=R, delta=delta)
                if fm > 0.0:
                    yl = ym
                else:
                    yr = ym
            y_sigma = 0.5 * (yl + yr)

            n = sphere_normal_xyz(x, y_sigma, z, R=R, delta=delta)
            if abs(n[1]) < 1e-12:
                continue

            dA = dx * dz / abs(n[1])
            p = k * max(0.0, -y_sigma)

            dFx = -p * n[0] * dA
            dFy = -p * n[1] * dA
            dFz = -p * n[2] * dA

            Fx += dFx
            Fy += dFy
            Fz += dFz
            Mx += y_sigma * dFz - z * dFy
            My += z * dFx - x * dFz
            Mz += x * dFy - y_sigma * dFx
            num_elems += 1

    return {"Fx": Fx, "Fy": Fy, "Fz": Fz, "Mx": Mx, "My": My, "Mz": Mz, "num_elems": num_elems}


# -----------------------------
# Method B3: optimized clipped footprint + local centroid + root solve
# -----------------------------
def version_B_sdf_clipped_sheet(
    *,
    R: float,
    delta: float,
    k: float,
    cube_size: float,
    cube_height: float,
    Nxz: int,
    Ncontour: int = 400,
    Mpoly: int = 96,
    bisection_steps: int = 30,
):
    poly = extract_top_slice_contour_polygon(
        R=R, delta=delta, cube_size=cube_size, Ncontour=Ncontour, Mpoly=Mpoly
    )
    if len(poly) == 0:
        return {
            "Fx": 0.0, "Fy": 0.0, "Fz": 0.0,
            "Mx": 0.0, "My": 0.0, "Mz": 0.0,
            "num_elems": 0, "poly": poly, "elems": []
        }

    x_min, x_max = -cube_size / 2.0, cube_size / 2.0
    z_min, z_max = -cube_size / 2.0, cube_size / 2.0
    x_edges = np.linspace(x_min, x_max, Nxz + 1)
    z_edges = np.linspace(z_min, z_max, Nxz + 1)
    dx = x_edges[1] - x_edges[0]
    dz = z_edges[1] - z_edges[0]
    cell_area = dx * dz

    poly_xmin = np.min(poly[:, 0])
    poly_xmax = np.max(poly[:, 0])
    poly_zmin = np.min(poly[:, 1])
    poly_zmax = np.max(poly[:, 1])

    i_min = max(0, int(np.floor((poly_xmin - x_min) / dx)) - 1)
    i_max = min(Nxz - 1, int(np.floor((poly_xmax - x_min) / dx)) + 1)
    j_min = max(0, int(np.floor((poly_zmin - z_min) / dz)) - 1)
    j_max = min(Nxz - 1, int(np.floor((poly_zmax - z_min) / dz)) + 1)

    Fx = Fy = Fz = 0.0
    Mx = My = Mz = 0.0
    elems = []

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            square = np.array([
                [x_edges[i],   z_edges[j]],
                [x_edges[i+1], z_edges[j]],
                [x_edges[i+1], z_edges[j+1]],
                [x_edges[i],   z_edges[j+1]],
            ], dtype=float)

            corners = square
            inside = [point_in_convex_polygon(c, poly) for c in corners]

            if all(inside):
                Aproj = cell_area
                C = np.array([0.5 * (x_edges[i] + x_edges[i+1]), 0.5 * (z_edges[j] + z_edges[j+1])], dtype=float)
            else:
                # quick bbox reject already done globally; only boundary candidates reach here
                clipped = clip_square_by_convex_polygon(square, poly)
                Aproj, C = polygon_area_centroid(clipped)
                if Aproj <= 1e-15:
                    continue

            x_c, z_c = float(C[0]), float(C[1])

            phi_top = sphere_phi_xyz(x_c, 0.0, z_c, R=R, delta=delta)
            phi_bot = sphere_phi_xyz(x_c, -cube_height, z_c, R=R, delta=delta)
            if not (phi_top <= 0.0 and phi_bot >= 0.0):
                continue

            yl = -cube_height
            yr = 0.0
            for _ in range(bisection_steps):
                ym = 0.5 * (yl + yr)
                fm = sphere_phi_xyz(x_c, ym, z_c, R=R, delta=delta)
                if fm > 0.0:
                    yl = ym
                else:
                    yr = ym
            y_sigma = 0.5 * (yl + yr)

            n = sphere_normal_xyz(x_c, y_sigma, z_c, R=R, delta=delta)
            if abs(n[1]) < 1e-12:
                continue

            dA = Aproj / abs(n[1])
            p = k * max(0.0, -y_sigma)

            dFx = -p * n[0] * dA
            dFy = -p * n[1] * dA
            dFz = -p * n[2] * dA

            Fx += dFx
            Fy += dFy
            Fz += dFz
            Mx += y_sigma * dFz - z_c * dFy
            My += z_c * dFx - x_c * dFz
            Mz += x_c * dFy - y_sigma * dFx

            elems.append({
                "x": x_c, "y": y_sigma, "z": z_c,
                "nx": n[0], "ny": n[1], "nz": n[2],
                "Aproj": Aproj, "dA": dA
            })

    return {
        "Fx": Fx, "Fy": Fy, "Fz": Fz,
        "Mx": Mx, "My": My, "Mz": Mz,
        "num_elems": len(elems), "poly": poly, "elems": elems
    }


# -----------------------------
# Geometry demo
# -----------------------------
def make_geometry_demo(out_dir: Path, *, poly: np.ndarray, elems: list[dict], cube_size: float):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12.0, 5.0))

    ax1 = fig.add_subplot(1, 2, 1)
    if len(poly) > 0:
        poly_closed = np.vstack([poly, poly[0]])
        ax1.plot(poly_closed[:, 0], poly_closed[:, 1], linewidth=2.0, label="footprint contour from SDF")
    if elems:
        X = np.array([e["x"] for e in elems])
        Z = np.array([e["z"] for e in elems])
        ax1.scatter(X, Z, s=10, label="clipped-cell centroids")
    ax1.set_aspect("equal")
    ax1.set_xlim(-cube_size / 2, cube_size / 2)
    ax1.set_ylim(-cube_size / 2, cube_size / 2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax1.set_title("Top-slice contour clipping in x-z")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    if elems:
        X = np.array([e["x"] for e in elems])
        Y = np.array([e["y"] for e in elems])
        Z = np.array([e["z"] for e in elems])
        NX = np.array([e["nx"] for e in elems])
        NY = np.array([e["ny"] for e in elems])
        NZ = np.array([e["nz"] for e in elems])
        ax2.scatter(X, Y, Z, s=8)
        stride = max(1, len(elems) // 120)
        ax2.quiver(X[::stride], Y[::stride], Z[::stride],
                   NX[::stride], NY[::stride], NZ[::stride],
                   length=0.08, normalize=True)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("Recovered sheet elements and local normals")
    ax2.set_box_aspect((1, 0.6, 1))

    plt.tight_layout()
    plt.savefig(out_dir / "hemisphere_cube_Bsdf_clipped_geometry_demo.png", dpi=180)
    plt.close()


# -----------------------------
# Main experiment
# -----------------------------
def main():
    out_dir = Path(__file__).resolve().parent

    R = 1.0
    k = 10.0
    cube_size = 1.6
    cube_height = 0.25
    shallow_deltas = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12])

    rows = []

    N_A_fixed = 96
    Na_B1_fixed = 24
    Nt_B1_fixed = 96
    Nxz_B2_fixed = 64
    Nxz_B3_fixed = 64

    geom_demo = None

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
        B1out = version_B_analytic_sheet(
            R=R, delta=delta, k=k,
            Na=Na_B1_fixed, Nt=Nt_B1_fixed
        )
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        B2out = version_B_sdf_column_center(
            R=R, delta=delta, k=k,
            cube_size=cube_size, cube_height=cube_height,
            Nxz=Nxz_B2_fixed, bisection_steps=30
        )
        t5 = time.perf_counter()

        t6 = time.perf_counter()
        B3out = version_B_sdf_clipped_sheet(
            R=R, delta=delta, k=k,
            cube_size=cube_size, cube_height=cube_height,
            Nxz=Nxz_B3_fixed, Ncontour=400, Mpoly=96, bisection_steps=30
        )
        t7 = time.perf_counter()

        if abs(delta - 0.03) < 1e-12:
            geom_demo = (B3out["poly"], B3out["elems"])

        for method, out, dt, res in [
            ("A_direct_band", Aout, t1 - t0, N_A_fixed),
            ("B_analytic_sheet", B1out, t3 - t2, Na_B1_fixed),
            ("B_sdf_column_center", B2out, t5 - t4, Nxz_B2_fixed),
            ("B_sdf_clipped_sheet", B3out, t7 - t6, Nxz_B3_fixed),
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

    for Nxz in [12, 16, 24, 32, 48, 64, 96]:
        t0 = time.perf_counter()
        out = version_B_sdf_column_center(
            R=R, delta=delta_conv, k=k,
            cube_size=cube_size, cube_height=cube_height,
            Nxz=Nxz, bisection_steps=30
        )
        t1 = time.perf_counter()
        rows.append({
            "mode": "convergence",
            "method": "B_sdf_column_center",
            "delta": delta_conv,
            "resolution": Nxz,
            "Fy_num": out["Fy"],
            "Fy_exact": Fy_exact_conv,
            "rel_err_Fy": abs(out["Fy"] - Fy_exact_conv) / abs(Fy_exact_conv),
            "sym_resid": np.sqrt(out["Fx"]**2 + out["Fz"]**2 + out["Mx"]**2 + out["Mz"]**2),
            "time_sec": t1 - t0,
        })

    for Nxz in [12, 16, 24, 32, 48, 64, 96]:
        t0 = time.perf_counter()
        out = version_B_sdf_clipped_sheet(
            R=R, delta=delta_conv, k=k,
            cube_size=cube_size, cube_height=cube_height,
            Nxz=Nxz, Ncontour=400, Mpoly=96, bisection_steps=30
        )
        t1 = time.perf_counter()
        rows.append({
            "mode": "convergence",
            "method": "B_sdf_clipped_sheet",
            "delta": delta_conv,
            "resolution": Nxz,
            "Fy_num": out["Fy"],
            "Fy_exact": Fy_exact_conv,
            "rel_err_Fy": abs(out["Fy"] - Fy_exact_conv) / abs(Fy_exact_conv),
            "sym_resid": np.sqrt(out["Fx"]**2 + out["Fz"]**2 + out["Mx"]**2 + out["Mz"]**2),
            "time_sec": t1 - t0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hemisphere_cube_Bsdf_clipped_results.csv", index=False)

    sweep = df[df["mode"] == "shallow_sweep"].copy()
    plt.figure(figsize=(7.2, 4.8))
    for method, grp in sweep.groupby("method"):
        grp = grp.sort_values("delta")
        plt.plot(grp["delta"], grp["Fy_num"], marker="o", label=method)
    exact_curve = sweep.drop_duplicates("delta").sort_values("delta")
    plt.plot(exact_curve["delta"], exact_curve["Fy_exact"], marker="x", label="exact")
    plt.xlabel("penetration depth δ")
    plt.ylabel("vertical force Fy")
    plt.title("Shallow indentation: contour-clipping upgrade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hemisphere_cube_Bsdf_clipped_shallow_force_compare.png", dpi=180)
    plt.close()

    conv = df[df["mode"] == "convergence"].copy()
    plt.figure(figsize=(7.2, 4.8))
    for method, grp in conv.groupby("method"):
        grp = grp.sort_values("resolution")
        plt.plot(grp["resolution"], grp["rel_err_Fy"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("resolution parameter")
    plt.ylabel("relative Fy error")
    plt.title("Shallow case convergence at δ=0.03")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hemisphere_cube_Bsdf_clipped_convergence.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
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
    plt.savefig(out_dir / "hemisphere_cube_Bsdf_clipped_error_vs_runtime.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for method, grp in conv.groupby("method"):
        grp = grp.sort_values("resolution")
        plt.plot(grp["resolution"], grp["sym_resid"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("resolution parameter")
    plt.ylabel("symmetry residual")
    plt.title("Centered axisymmetric case residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hemisphere_cube_Bsdf_clipped_symmetry_residual.png", dpi=180)
    plt.close()

    if geom_demo is not None:
        make_geometry_demo(out_dir, poly=geom_demo[0], elems=geom_demo[1], cube_size=cube_size)

    print("Saved:")
    print(out_dir / "hemisphere_cube_Bsdf_clipped_results.csv")
    print(out_dir / "hemisphere_cube_Bsdf_clipped_shallow_force_compare.png")
    print(out_dir / "hemisphere_cube_Bsdf_clipped_convergence.png")
    print(out_dir / "hemisphere_cube_Bsdf_clipped_error_vs_runtime.png")
    print(out_dir / "hemisphere_cube_Bsdf_clipped_symmetry_residual.png")
    print(out_dir / "hemisphere_cube_Bsdf_clipped_geometry_demo.png")


if __name__ == "__main__":
    main()

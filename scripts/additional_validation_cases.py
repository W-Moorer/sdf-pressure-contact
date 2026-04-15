import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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

def exact_force_moment(R: float, delta: float, k: float, x0: float = 0.0, z0: float = 0.0):
    Fy = k * exact_cap_volume(R, delta)
    return {
        'Fx': 0.0, 'Fy': Fy, 'Fz': 0.0,
        'Mx': -z0 * Fy, 'My': 0.0, 'Mz': x0 * Fy,
    }

def sphere_phi_xyz(x, y, z, *, R: float, delta: float, x0: float = 0.0, z0: float = 0.0):
    cx = x0; cy = R - delta; cz = z0
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - R

def sphere_normal_xyz(x, y, z, *, R: float, delta: float, x0: float = 0.0, z0: float = 0.0):
    cx = x0; cy = R - delta; cz = z0
    dx = x - cx; dy = y - cy; dz = z - cz
    rr = np.sqrt(dx * dx + dy * dy + dz * dz)
    eps = 1e-15
    return np.array([dx, dy, dz]) / max(rr, eps)

# -----------------------------
# Method A: direct voxel band
# -----------------------------
def version_A_direct_band(*, R, delta, k, cube_size, cube_height, N, x0=0.0, z0=0.0, eta_factor=1.5):
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

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    cx = x0; cy = R - delta; cz = z0
    RR = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    phi = RR - R
    eps = 1e-15
    nx = (X - cx) / np.maximum(RR, eps)
    ny = (Y - cy) / np.maximum(RR, eps)
    nz = (Z - cz) / np.maximum(RR, eps)
    p = k * np.clip(-Y, 0.0, None)
    h = max(dx, dy, dz)
    eta = eta_factor * h
    band = delta_cosine(phi, eta)
    tx = -p * nx; ty = -p * ny; tz = -p * nz
    Fx = np.sum(tx * band) * dV
    Fy = np.sum(ty * band) * dV
    Fz = np.sum(tz * band) * dV
    Mx = np.sum((Y * tz - Z * ty) * band) * dV
    My = np.sum((Z * tx - X * tz) * band) * dV
    Mz = np.sum((X * ty - Y * tx) * band) * dV
    return {'Fx': Fx, 'Fy': Fy, 'Fz': Fz, 'Mx': Mx, 'My': My, 'Mz': Mz}

# -----------------------------
# Polygon utilities
# -----------------------------
def polygon_signed_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]; y = poly[:, 1]
    xp = np.roll(x, -1); yp = np.roll(y, -1)
    return 0.5 * np.sum(x * yp - xp * y)

def polygon_area_centroid(poly: np.ndarray):
    if len(poly) < 3:
        return 0.0, np.array([np.nan, np.nan])
    x = poly[:, 0]; y = poly[:, 1]
    xp = np.roll(x, -1); yp = np.roll(y, -1)
    cross = x * yp - xp * y
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-15:
        return 0.0, np.array([np.nan, np.nan])
    Cx = np.sum((x + xp) * cross) / (6.0 * A)
    Cy = np.sum((y + yp) * cross) / (6.0 * A)
    return abs(A), np.array([Cx, Cy])

def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return a[0] * b[1] - a[1] * b[0]

def triangle_area(tri: np.ndarray) -> float:
    return 0.5 * abs(cross2(tri[1] - tri[0], tri[2] - tri[0]))

def line_intersection_with_clip_edge(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-14):
    r = p2 - p1; s = b - a
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
        prev = curr; prev_inside = curr_inside
    return np.array(output, dtype=float)

def clip_square_by_convex_polygon(square_poly: np.ndarray, clip_poly: np.ndarray):
    poly = square_poly.copy()
    for i in range(len(clip_poly)):
        a = clip_poly[i]; b = clip_poly[(i + 1) % len(clip_poly)]
        poly = clip_polygon_against_edge(poly, a, b)
        if len(poly) == 0:
            break
    return poly

def point_in_convex_polygon(pt: np.ndarray, poly: np.ndarray, eps: float = 1e-12) -> bool:
    for i in range(len(poly)):
        a = poly[i]; b = poly[(i + 1) % len(poly)]
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
        a = closed[j]; b = closed[j + 1]
        ds = s[j + 1] - s[j]
        w = 0.0 if ds <= 0 else (t - s[j]) / ds
        out.append((1 - w) * a + w * b)
    out = np.array(out)
    if polygon_signed_area(out) < 0.0:
        out = out[::-1]
    return out

def triangulate_convex_polygon(poly: np.ndarray):
    if len(poly) < 3:
        return []
    _, c = polygon_area_centroid(poly)
    tris = []
    for i in range(len(poly)):
        tri = np.vstack([c, poly[i], poly[(i + 1) % len(poly)]])
        A = triangle_area(tri)
        if A > 1e-15:
            tris.append(tri)
    return tris

def triangle_quadrature_points_3pt(tri: np.ndarray):
    A = triangle_area(tri)
    v0, v1, v2 = tri
    bary = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]], dtype=float)
    out = []; w = A / 3.0
    for lam in bary:
        p = lam[0] * v0 + lam[1] * v1 + lam[2] * v2
        out.append((p, w))
    return out

# -----------------------------
# Automatic top-slice contour extraction from SDF
# -----------------------------
def extract_top_slice_contour_polygon(*, R, delta, cube_size, x0=0.0, z0=0.0, Ncontour=300, Mpoly=72):
    x_min, x_max = -cube_size / 2.0, cube_size / 2.0
    z_min, z_max = -cube_size / 2.0, cube_size / 2.0
    xs = np.linspace(x_min, x_max, Ncontour)
    zs = np.linspace(z_min, z_max, Ncontour)
    X, Z = np.meshgrid(xs, zs, indexing='xy')
    phi = sphere_phi_xyz(X, 0.0, Z, R=R, delta=delta, x0=x0, z0=z0)
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
# Method B2: column-center extraction
# -----------------------------
def version_B_sdf_column_center(*, R, delta, k, cube_size, cube_height, Nxz, x0=0.0, z0=0.0, bisection_steps=30):
    x_min, x_max = -cube_size / 2.0, cube_size / 2.0
    z_min, z_max = -cube_size / 2.0, cube_size / 2.0
    xs = np.linspace(x_min, x_max, Nxz, endpoint=False) + (x_max - x_min) / Nxz / 2.0
    zs = np.linspace(z_min, z_max, Nxz, endpoint=False) + (z_max - z_min) / Nxz / 2.0
    dx = (x_max - x_min) / Nxz; dz = (z_max - z_min) / Nxz
    Fx = Fy = Fz = 0.0; Mx = My = Mz = 0.0
    for x in xs:
        for z in zs:
            phi_top = sphere_phi_xyz(x, 0.0, z, R=R, delta=delta, x0=x0, z0=z0)
            phi_bot = sphere_phi_xyz(x, -cube_height, z, R=R, delta=delta, x0=x0, z0=z0)
            if not (phi_top <= 0.0 and phi_bot >= 0.0):
                continue
            yl = -cube_height; yr = 0.0
            for _ in range(bisection_steps):
                ym = 0.5 * (yl + yr)
                fm = sphere_phi_xyz(x, ym, z, R=R, delta=delta, x0=x0, z0=z0)
                if fm > 0.0:
                    yl = ym
                else:
                    yr = ym
            y_sigma = 0.5 * (yl + yr)
            n = sphere_normal_xyz(x, y_sigma, z, R=R, delta=delta, x0=x0, z0=z0)
            if abs(n[1]) < 1e-12:
                continue
            dA = dx * dz / abs(n[1]); p = k * max(0.0, -y_sigma)
            dFx = -p * n[0] * dA; dFy = -p * n[1] * dA; dFz = -p * n[2] * dA
            Fx += dFx; Fy += dFy; Fz += dFz
            Mx += y_sigma * dFz - z * dFy
            My += z * dFx - x * dFz
            Mz += x * dFy - y_sigma * dFx
    return {'Fx': Fx, 'Fy': Fy, 'Fz': Fz, 'Mx': Mx, 'My': My, 'Mz': Mz}

# -----------------------------
# Method B4: clipped polygon + triangle quadrature
# -----------------------------
def version_B_sdf_quad_sheet(*, R, delta, k, cube_size, cube_height, Nxz, x0=0.0, z0=0.0, Ncontour=300, Mpoly=72, bisection_steps=30):
    poly = extract_top_slice_contour_polygon(R=R, delta=delta, cube_size=cube_size, x0=x0, z0=z0, Ncontour=Ncontour, Mpoly=Mpoly)
    if len(poly) == 0:
        return {'Fx': 0.0, 'Fy': 0.0, 'Fz': 0.0, 'Mx': 0.0, 'My': 0.0, 'Mz': 0.0, 'poly': poly, 'qps': []}
    x_min, x_max = -cube_size / 2.0, cube_size / 2.0
    z_min, z_max = -cube_size / 2.0, cube_size / 2.0
    x_edges = np.linspace(x_min, x_max, Nxz + 1)
    z_edges = np.linspace(z_min, z_max, Nxz + 1)
    dx = x_edges[1] - x_edges[0]; dz = z_edges[1] - z_edges[0]
    poly_xmin = np.min(poly[:, 0]); poly_xmax = np.max(poly[:, 0])
    poly_zmin = np.min(poly[:, 1]); poly_zmax = np.max(poly[:, 1])
    i_min = max(0, int(np.floor((poly_xmin - x_min) / dx)) - 1)
    i_max = min(Nxz - 1, int(np.floor((poly_xmax - x_min) / dx)) + 1)
    j_min = max(0, int(np.floor((poly_zmin - z_min) / dz)) - 1)
    j_max = min(Nxz - 1, int(np.floor((poly_zmax - z_min) / dz)) + 1)
    Fx = Fy = Fz = 0.0; Mx = My = Mz = 0.0; qps = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            square = np.array([[x_edges[i], z_edges[j]], [x_edges[i + 1], z_edges[j]], [x_edges[i + 1], z_edges[j + 1]], [x_edges[i], z_edges[j + 1]]], dtype=float)
            corners = square
            inside = [point_in_convex_polygon(c, poly) for c in corners]
            if all(inside):
                clipped = square
            else:
                clipped = clip_square_by_convex_polygon(square, poly)
                if len(clipped) < 3:
                    continue
            tris = triangulate_convex_polygon(clipped)
            for tri in tris:
                for p2d, w_proj in triangle_quadrature_points_3pt(tri):
                    x_q, z_q = float(p2d[0]), float(p2d[1])
                    phi_top = sphere_phi_xyz(x_q, 0.0, z_q, R=R, delta=delta, x0=x0, z0=z0)
                    phi_bot = sphere_phi_xyz(x_q, -cube_height, z_q, R=R, delta=delta, x0=x0, z0=z0)
                    if not (phi_top <= 0.0 and phi_bot >= 0.0):
                        continue
                    yl = -cube_height; yr = 0.0
                    for _ in range(bisection_steps):
                        ym = 0.5 * (yl + yr)
                        fm = sphere_phi_xyz(x_q, ym, z_q, R=R, delta=delta, x0=x0, z0=z0)
                        if fm > 0.0:
                            yl = ym
                        else:
                            yr = ym
                    y_sigma = 0.5 * (yl + yr)
                    n = sphere_normal_xyz(x_q, y_sigma, z_q, R=R, delta=delta, x0=x0, z0=z0)
                    if abs(n[1]) < 1e-12:
                        continue
                    dA = w_proj / abs(n[1]); p = k * max(0.0, -y_sigma)
                    dFx = -p * n[0] * dA; dFy = -p * n[1] * dA; dFz = -p * n[2] * dA
                    Fx += dFx; Fy += dFy; Fz += dFz
                    Mx += y_sigma * dFz - z_q * dFy
                    My += z_q * dFx - x_q * dFz
                    Mz += x_q * dFy - y_sigma * dFx
                    qps.append({'x': x_q, 'y': y_sigma, 'z': z_q, 'nx': n[0], 'ny': n[1], 'nz': n[2], 'w_proj': w_proj, 'dA': dA})
    return {'Fx': Fx, 'Fy': Fy, 'Fz': Fz, 'Mx': Mx, 'My': My, 'Mz': Mz, 'poly': poly, 'qps': qps}

def make_offset_geometry_demo(out_dir: Path, *, poly: np.ndarray, qps: list[dict], cube_size: float):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(12.0, 5.0))
    ax1 = fig.add_subplot(1, 2, 1)
    if len(poly) > 0:
        poly_closed = np.vstack([poly, poly[0]])
        ax1.plot(poly_closed[:, 0], poly_closed[:, 1], linewidth=2.0, label='footprint contour from SDF')
    if qps:
        X = np.array([e['x'] for e in qps]); Z = np.array([e['z'] for e in qps])
        ax1.scatter(X, Z, s=8, label='quadrature points')
    ax1.set_aspect('equal'); ax1.set_xlim(-cube_size / 2, cube_size / 2); ax1.set_ylim(-cube_size / 2, cube_size / 2)
    ax1.set_xlabel('x'); ax1.set_ylabel('z'); ax1.set_title('Offset footprint clipping + quadrature'); ax1.legend()
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if qps:
        X = np.array([e['x'] for e in qps]); Y = np.array([e['y'] for e in qps]); Z = np.array([e['z'] for e in qps])
        NX = np.array([e['nx'] for e in qps]); NY = np.array([e['ny'] for e in qps]); NZ = np.array([e['nz'] for e in qps])
        ax2.scatter(X, Y, Z, s=8)
        stride = max(1, len(qps) // 120)
        ax2.quiver(X[::stride], Y[::stride], Z[::stride], NX[::stride], NY[::stride], NZ[::stride], length=0.08, normalize=True)
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
    ax2.set_title('Recovered local-normal quadrature points'); ax2.set_box_aspect((1, 0.6, 1))
    plt.tight_layout(); plt.savefig(out_dir / 'additional_validation_offset_geometry.png', dpi=180); plt.close()

def main(out_dir: Path):
    cube_size = 1.9; cube_height = 0.28; rows = []
    # Case 1 offset sweep
    R = 1.0; k = 10.0; x0 = 0.18; z0 = -0.11
    deltas = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12])
    geometry_demo_data = None
    for delta in deltas:
        exact = exact_force_moment(R, delta, k, x0=x0, z0=z0)
        t0 = time.perf_counter(); A = version_A_direct_band(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, N=96, x0=x0, z0=z0); t1 = time.perf_counter()
        t2 = time.perf_counter(); Bc = version_B_sdf_column_center(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, Nxz=64, x0=x0, z0=z0); t3 = time.perf_counter()
        t4 = time.perf_counter(); Bq = version_B_sdf_quad_sheet(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, Nxz=48, x0=x0, z0=z0); t5 = time.perf_counter()
        if abs(delta - 0.03) < 1e-12:
            geometry_demo_data = (Bq['poly'], Bq['qps'])
        for method, out, dt, res in [('A_direct_band', A, t1 - t0, 96), ('B_sdf_column_center', Bc, t3 - t2, 64), ('B_sdf_quad_sheet', Bq, t5 - t4, 48)]:
            rows.append({'case_family':'offset_sweep','method':method,'R':R,'k':k,'delta':delta,'x0':x0,'z0':z0,'resolution':res,
                         'Fy_num':out['Fy'],'Fy_exact':exact['Fy'],'rel_err_Fy':abs(out['Fy']-exact['Fy'])/abs(exact['Fy']),
                         'Mx_num':out['Mx'],'Mx_exact':exact['Mx'],'rel_err_Mx':abs(out['Mx']-exact['Mx'])/(abs(exact['Mx'])+1e-30),
                         'Mz_num':out['Mz'],'Mz_exact':exact['Mz'],'rel_err_Mz':abs(out['Mz']-exact['Mz'])/(abs(exact['Mz'])+1e-30),
                         'sym_resid':np.sqrt(out['Fx']**2+out['Fz']**2+out['My']**2),'time_sec':dt})
    # Case 2 curvature sweep
    delta = 0.03; k = 10.0; x0 = 0.0; z0 = 0.0; Rs = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    for R in Rs:
        exact = exact_force_moment(R, delta, k)
        t0=time.perf_counter(); A=version_A_direct_band(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, N=96); t1=time.perf_counter()
        t2=time.perf_counter(); Bc=version_B_sdf_column_center(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, Nxz=64); t3=time.perf_counter()
        t4=time.perf_counter(); Bq=version_B_sdf_quad_sheet(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, Nxz=48); t5=time.perf_counter()
        for method, out, dt, res in [('A_direct_band', A, t1 - t0, 96), ('B_sdf_column_center', Bc, t3 - t2, 64), ('B_sdf_quad_sheet', Bq, t5 - t4, 48)]:
            rows.append({'case_family':'curvature_sweep','method':method,'R':R,'k':k,'delta':delta,'x0':x0,'z0':z0,'resolution':res,
                         'Fy_num':out['Fy'],'Fy_exact':exact['Fy'],'rel_err_Fy':abs(out['Fy']-exact['Fy'])/abs(exact['Fy']),
                         'Mx_num':out['Mx'],'Mx_exact':exact['Mx'],'rel_err_Mx':0.0,'Mz_num':out['Mz'],'Mz_exact':exact['Mz'],'rel_err_Mz':0.0,
                         'sym_resid':np.sqrt(out['Fx']**2+out['Fz']**2+out['Mx']**2+out['Mz']**2),'time_sec':dt})
    # Case 3 stiffness sweep
    R = 1.0; delta = 0.03; ks = np.array([2.0, 5.0, 10.0, 20.0, 40.0])
    for k in ks:
        exact = exact_force_moment(R, delta, k)
        t0=time.perf_counter(); A=version_A_direct_band(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, N=96); t1=time.perf_counter()
        t2=time.perf_counter(); Bc=version_B_sdf_column_center(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, Nxz=64); t3=time.perf_counter()
        t4=time.perf_counter(); Bq=version_B_sdf_quad_sheet(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, Nxz=48); t5=time.perf_counter()
        for method, out, dt, res in [('A_direct_band', A, t1 - t0, 96), ('B_sdf_column_center', Bc, t3 - t2, 64), ('B_sdf_quad_sheet', Bq, t5 - t4, 48)]:
            rows.append({'case_family':'stiffness_sweep','method':method,'R':R,'k':k,'delta':delta,'x0':0.0,'z0':0.0,'resolution':res,
                         'Fy_num':out['Fy'],'Fy_exact':exact['Fy'],'rel_err_Fy':abs(out['Fy']-exact['Fy'])/abs(exact['Fy']),
                         'Mx_num':out['Mx'],'Mx_exact':exact['Mx'],'rel_err_Mx':0.0,'Mz_num':out['Mz'],'Mz_exact':exact['Mz'],'rel_err_Mz':0.0,
                         'sym_resid':np.sqrt(out['Fx']**2+out['Fz']**2+out['Mx']**2+out['Mz']**2),'time_sec':dt})
    # Case 4 offset convergence
    R = 1.0; k = 10.0; delta = 0.03; x0 = 0.18; z0 = -0.11; exact = exact_force_moment(R, delta, k, x0=x0, z0=z0)
    for N in [48, 64, 80, 96, 120]:
        t0=time.perf_counter(); out=version_A_direct_band(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, N=N, x0=x0, z0=z0); t1=time.perf_counter()
        rows.append({'case_family':'offset_convergence','method':'A_direct_band','R':R,'k':k,'delta':delta,'x0':x0,'z0':z0,'resolution':N,
                     'Fy_num':out['Fy'],'Fy_exact':exact['Fy'],'rel_err_Fy':abs(out['Fy']-exact['Fy'])/abs(exact['Fy']),
                     'Mx_num':out['Mx'],'Mx_exact':exact['Mx'],'rel_err_Mx':abs(out['Mx']-exact['Mx'])/abs(exact['Mx']),
                     'Mz_num':out['Mz'],'Mz_exact':exact['Mz'],'rel_err_Mz':abs(out['Mz']-exact['Mz'])/abs(exact['Mz']),
                     'sym_resid':np.sqrt(out['Fx']**2+out['Fz']**2+out['My']**2),'time_sec':t1-t0})
    for Nxz in [16, 24, 32, 48, 64]:
        t0=time.perf_counter(); out=version_B_sdf_column_center(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, Nxz=Nxz, x0=x0, z0=z0); t1=time.perf_counter()
        rows.append({'case_family':'offset_convergence','method':'B_sdf_column_center','R':R,'k':k,'delta':delta,'x0':x0,'z0':z0,'resolution':Nxz,
                     'Fy_num':out['Fy'],'Fy_exact':exact['Fy'],'rel_err_Fy':abs(out['Fy']-exact['Fy'])/abs(exact['Fy']),
                     'Mx_num':out['Mx'],'Mx_exact':exact['Mx'],'rel_err_Mx':abs(out['Mx']-exact['Mx'])/abs(exact['Mx']),
                     'Mz_num':out['Mz'],'Mz_exact':exact['Mz'],'rel_err_Mz':abs(out['Mz']-exact['Mz'])/abs(exact['Mz']),
                     'sym_resid':np.sqrt(out['Fx']**2+out['Fz']**2+out['My']**2),'time_sec':t1-t0})
    for Nxz in [12, 16, 24, 32, 48]:
        t0=time.perf_counter(); out=version_B_sdf_quad_sheet(R=R, delta=delta, k=k, cube_size=cube_size, cube_height=cube_height, Nxz=Nxz, x0=x0, z0=z0); t1=time.perf_counter()
        rows.append({'case_family':'offset_convergence','method':'B_sdf_quad_sheet','R':R,'k':k,'delta':delta,'x0':x0,'z0':z0,'resolution':Nxz,
                     'Fy_num':out['Fy'],'Fy_exact':exact['Fy'],'rel_err_Fy':abs(out['Fy']-exact['Fy'])/abs(exact['Fy']),
                     'Mx_num':out['Mx'],'Mx_exact':exact['Mx'],'rel_err_Mx':abs(out['Mx']-exact['Mx'])/abs(exact['Mx']),
                     'Mz_num':out['Mz'],'Mz_exact':exact['Mz'],'rel_err_Mz':abs(out['Mz']-exact['Mz'])/abs(exact['Mz']),
                     'sym_resid':np.sqrt(out['Fx']**2+out['Fz']**2+out['My']**2),'time_sec':t1-t0})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'additional_validation_cases_results.csv', index=False)

    if geometry_demo_data is not None:
        make_offset_geometry_demo(out_dir, poly=geometry_demo_data[0], qps=geometry_demo_data[1], cube_size=cube_size)

    off = df[df['case_family'] == 'offset_sweep'].copy()
    plt.figure(figsize=(7.2, 4.8))
    for method, grp in off.groupby('method'):
        grp = grp.sort_values('delta')
        plt.plot(grp['delta'], grp['Fy_num'], marker='o', label=f'{method}: Fy')
    exact_curve = off.drop_duplicates('delta').sort_values('delta')
    plt.plot(exact_curve['delta'], exact_curve['Fy_exact'], marker='x', label='exact Fy')
    plt.xlabel('penetration depth δ'); plt.ylabel('vertical force Fy'); plt.title('Offset case: force accuracy')
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'additional_validation_offset_force.png', dpi=180); plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for method, grp in off.groupby('method'):
        grp = grp.sort_values('delta')
        plt.plot(grp['delta'], grp['Mz_num'], marker='o', label=f'{method}: Mz')
        plt.plot(grp['delta'], grp['Mx_num'], marker='s', label=f'{method}: Mx')
    plt.plot(exact_curve['delta'], exact_curve['Mz_exact'], marker='x', label='exact Mz')
    plt.plot(exact_curve['delta'], exact_curve['Mx_exact'], marker='^', label='exact Mx')
    plt.xlabel('penetration depth δ'); plt.ylabel('moment about origin'); plt.title('Offset case: moment accuracy')
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'additional_validation_offset_moment.png', dpi=180); plt.close()

    curv = df[df['case_family'] == 'curvature_sweep'].copy()
    plt.figure(figsize=(7.2, 4.8))
    for method, grp in curv.groupby('method'):
        grp = grp.sort_values('R')
        plt.plot(grp['R'], grp['rel_err_Fy'], marker='o', label=method)
    plt.yscale('log'); plt.xlabel('sphere radius R'); plt.ylabel('relative Fy error'); plt.title('Curvature sweep at fixed shallow indentation')
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'additional_validation_curvature_error.png', dpi=180); plt.close()

    stiff = df[df['case_family'] == 'stiffness_sweep'].copy()
    plt.figure(figsize=(7.2, 4.8))
    for method, grp in stiff.groupby('method'):
        grp = grp.sort_values('k')
        plt.plot(grp['k'], grp['rel_err_Fy'], marker='o', label=method)
    plt.yscale('log'); plt.xlabel('stiffness k'); plt.ylabel('relative Fy error'); plt.title('Stiffness sweep at fixed shallow indentation')
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'additional_validation_stiffness_error.png', dpi=180); plt.close()

    conv = df[df['case_family'] == 'offset_convergence'].copy()
    plt.figure(figsize=(7.2, 4.8))
    for method, grp in conv.groupby('method'):
        grp = grp.sort_values('resolution')
        plt.plot(grp['resolution'], grp['rel_err_Fy'], marker='o', label=f'{method}: Fy')
        plt.plot(grp['resolution'], grp['rel_err_Mz'], marker='s', label=f'{method}: Mz')
    plt.yscale('log'); plt.xlabel('resolution parameter'); plt.ylabel('relative error'); plt.title('Offset shallow convergence: force and moment')
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / 'additional_validation_convergence_error.png', dpi=180); plt.close()
    return df

if __name__ == '__main__':
    out = Path('/mnt/data')
    main(out)

import numpy as np
import pandas as pd

def delta_cosine(s, eta):
    a = np.abs(s)
    out = np.zeros_like(s, dtype=float)
    mask = a <= eta
    out[mask] = 0.5 / eta * (1.0 + np.cos(np.pi * s[mask] / eta))
    return out

def analytic_planar_contact(kA, kB, delta, patch_area, xbar, zbar=0.0):
    y_star = -kB * delta / (kA + kB)
    p_star = (kA * kB / (kA + kB)) * delta
    Fy = patch_area * p_star
    Mx = -zbar * Fy
    Mz = xbar * Fy
    return {"y_star": y_star, "p_star": p_star, "area": patch_area, "xbar": xbar, "zbar": zbar, "Fy": Fy, "Mx": Mx, "Mz": Mz}

def rect_vertices(center, size, angle_rad):
    cx, cz = center
    lx, lz = size
    hx, hz = lx / 2.0, lz / 2.0
    local = np.array([[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz]])
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([cx, cz])

def point_in_rotated_rect(points, center, size, angle_rad):
    cx, cz = center
    lx, lz = size
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    Rt = np.array([[c, s], [-s, c]])
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

def clipped_column_geometry(x_edges, z_edges, rect_poly):
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

def band_integral_planar_patch(Nx, Ny, Nz, *, kA, kB, delta, xmin, xmax, zmin, zmax, pad_y, patch_center, patch_size, angle_rad, method="cell_center", eta_factor=2.0):
    ymin = -delta - pad_y
    ymax = pad_y
    x_edges = np.linspace(xmin, xmax, Nx + 1)
    y_edges = np.linspace(ymin, ymax, Ny + 1)
    z_edges = np.linspace(zmin, zmax, Nz + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    dz = z_edges[1] - z_edges[0]
    dV = dx * dy * dz

    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
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
        Xmom, Zmom = X, Z
    elif method == "area_fraction":
        patch_weight = frac[:, None, :]
        Xmom, Zmom = X, Z
    elif method == "area_fraction_centroid":
        patch_weight = frac[:, None, :]
        Xmom = np.broadcast_to(cx_local[:, None, :], (Nx, Ny, Nz))
        Zmom = np.broadcast_to(cz_local[:, None, :], (Nx, Ny, Nz))
    else:
        raise ValueError("unknown method")

    weight = band * patch_weight
    Fy = np.sum(p_bar * weight) * dV
    Mz = np.sum(Xmom * p_bar * weight) * dV
    area_est = np.sum(weight) * dV
    xbar_est = np.sum(Xmom * weight) * dV / area_est
    return {"area_est": area_est, "xbar_est": xbar_est, "Fy": Fy, "Mz": Mz}

if __name__ == "__main__":
    kA, kB, delta = 30.0, 90.0, 0.10
    patch_center = (0.22, -0.10)
    patch_size = (1.20, 0.55)
    angle_rad = np.deg2rad(28.0)
    xmin, xmax = -0.70, 1.20
    zmin, zmax = -0.75, 0.55
    pad_y = 0.16
    for method in ["cell_center", "area_fraction", "area_fraction_centroid"]:
        out = band_integral_planar_patch(36, 36, 36, kA=kA, kB=kB, delta=delta, xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, pad_y=pad_y, patch_center=patch_center, patch_size=patch_size, angle_rad=angle_rad, method=method)
        print(method, out)

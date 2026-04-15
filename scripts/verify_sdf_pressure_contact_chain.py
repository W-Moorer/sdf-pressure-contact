import numpy as np

def delta_cosine(s, eta):
    a = np.abs(s)
    out = np.zeros_like(s, dtype=float)
    mask = a <= eta
    out[mask] = 0.5 / eta * (1.0 + np.cos(np.pi * s[mask] / eta))
    return out

def analytic_planar_contact(kA, kB, delta, Lx, Lz, x0=0.0, z0=0.0):
    area = Lx * Lz
    y_star = -kB * delta / (kA + kB)
    p_star = (kA * kB / (kA + kB)) * delta
    Fy = area * p_star
    Mz = x0 * Fy
    return {"area": area, "y_star": y_star, "p_star": p_star, "Fy": Fy, "Mz": Mz}

def volume_integral_planar_contact(
    Nx, Ny, Nz, *,
    kA, kB, delta, Lx, Lz, x0=0.0, z0=0.0,
    pad_x=0.2, pad_z=0.2, pad_y=0.15, eta_factor=2.0
):
    x_min = x0 - 0.5 * Lx - pad_x
    x_max = x0 + 0.5 * Lx + pad_x
    z_min = z0 - 0.5 * Lz - pad_z
    z_max = z0 + 0.5 * Lz + pad_z
    y_min = -delta - pad_y
    y_max = pad_y

    xs = np.linspace(x_min, x_max, Nx, endpoint=False) + (x_max - x_min) / Nx / 2
    ys = np.linspace(y_min, y_max, Ny, endpoint=False) + (y_max - y_min) / Ny / 2
    zs = np.linspace(z_min, z_max, Nz, endpoint=False) + (z_max - z_min) / Nz / 2

    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    dz = (z_max - z_min) / Nz
    dV = dx * dy * dz

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    phi_A = Y
    phi_B = -(Y + delta)

    dA = np.clip(-phi_A, 0.0, None)
    dB = np.clip(-phi_B, 0.0, None)

    pA = kA * dA
    pB = kB * dB

    h = pA - pB
    grad_h_norm = (kA + kB)

    overlap = (phi_A <= 0.0) & (phi_B <= 0.0)
    patch = (np.abs(X - x0) <= 0.5 * Lx) & (np.abs(Z - z0) <= 0.5 * Lz)

    eta_y = eta_factor * dy
    eta_h = grad_h_norm * eta_y
    weight = delta_cosine(h, eta_h) * grad_h_norm * overlap * patch

    p_bar = 0.5 * (pA + pB)
    Fy = np.sum(p_bar * weight) * dV
    Mz = np.sum(X * p_bar * weight) * dV
    area_est = np.sum(weight) * dV
    y_star_est = np.sum(Y * weight) * dV / area_est

    return {"Fy": Fy, "Mz": Mz, "area_est": area_est, "y_star_est": y_star_est, "eta_y": eta_y, "dy": dy}

if __name__ == "__main__":
    params = dict(kA=30.0, kB=90.0, delta=0.10, Lx=1.20, Lz=0.80, x0=0.35, z0=0.0)
    out = volume_integral_planar_contact(128, 128, 128, **params)
    ana = analytic_planar_contact(**params)
    print("Numerical:", out)
    print("Analytic:", ana)

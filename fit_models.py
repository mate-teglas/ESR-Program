"""Automatic ESR fitting for amplitude-modulated and derivative-detected spectra.

Public API kept compatible with the existing GUI:
    fit_curve(B, y, npeaks=4, model='mixed') -> popt, pcov
    multi_peak_derivative(B, *params)
    multi_peak_mixed(B, *params)
    extract_peak_table(popt, model)
    fit_spectrum_series(...)

For amplitude-modulated ESR, ``model='mixed'`` uses a NON-DERIVATIVE
absorption/dispersion mixture for every physical resonance:

    A_abs * L_abs(B) + A_disp * L_disp(B)

The nonlinear optimizer varies only B0 and gamma. Absorptive/dispersive
coefficients and polynomial baseline are solved exactly by linear least squares
at every optimizer step (variable projection).
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.signal import savgol_filter

MU_B = 9.2740100783e-24
H = 6.62607015e-34
DEFAULT_GAMMA_BOUNDS = (0.002, 0.35)
MAX_FIT_POINTS = 900
BATCH_FIT_POINTS = 420
BASELINE_DEGREE = 3
PUBLIC_BASELINE_TERMS = BASELINE_DEGREE + 1


def lorentzian(B, A, B0, gamma):
    B = np.asarray(B, float)
    g = abs(float(gamma))
    return A * g * g / ((B - B0) ** 2 + g * g)


def dispersive_lorentzian(B, A, B0, gamma):
    B = np.asarray(B, float)
    g = abs(float(gamma))
    return A * g * (B - B0) / ((B - B0) ** 2 + g * g)


def derivative_lorentzian(B, A, B0, gamma):
    B = np.asarray(B, float)
    g = abs(float(gamma))
    den = ((B - B0) ** 2 + g * g) ** 2
    return A * (-2.0 * g * g * (B - B0)) / den


def mixed_derivative_peak(B, A, theta, B0, gamma):
    """Legacy name retained for GUI compatibility.

    Despite the historical name, this is now the correct AM-ESR line shape:
    a non-derivative absorptive/dispersive Lorentzian mixture.
    """
    return A * (
        np.cos(theta) * lorentzian(B, 1.0, B0, gamma)
        + np.sin(theta) * dispersive_lorentzian(B, 1.0, B0, gamma)
    )


def _split_blocks_and_baseline(p, block_size):
    """Accept new cubic-baseline params and legacy linear-baseline params."""
    p = np.asarray(p, float)
    if len(p) >= block_size + PUBLIC_BASELINE_TERMS and (len(p) - PUBLIC_BASELINE_TERMS) % block_size == 0:
        return p[:-PUBLIC_BASELINE_TERMS], p[-PUBLIC_BASELINE_TERMS:]
    if len(p) >= block_size + 2 and (len(p) - 2) % block_size == 0:
        return p[:-2], p[-2:]
    raise ValueError("Invalid ESR parameter-vector length")


def _evaluate_baseline(B, coeffs):
    B = np.asarray(B, float)
    if len(coeffs) == 2:  # legacy c0+c1(B-Bmid)
        mid = 0.5 * (B.min() + B.max())
        return coeffs[0] + coeffs[1] * (B - mid)
    mid = 0.5 * (B.min() + B.max())
    half = max(0.5 * (B.max() - B.min()), 1e-12)
    t = (B - mid) / half
    out = np.zeros_like(B)
    for k, c in enumerate(coeffs):
        out += c * t**k
    return out


def multi_peak_derivative(B, *params):
    B = np.asarray(B, float)
    blocks, baseline = _split_blocks_and_baseline(params, 3)
    out = np.zeros_like(B)
    for A, B0, g in blocks.reshape(-1, 3):
        out += derivative_lorentzian(B, A, B0, g)
    return out + _evaluate_baseline(B, baseline)


def multi_peak_mixed(B, *params):
    """Sum of AM-ESR absorption/dispersion mixtures plus cubic baseline."""
    B = np.asarray(B, float)
    blocks, baseline = _split_blocks_and_baseline(params, 4)
    out = np.zeros_like(B)
    for A, th, B0, g in blocks.reshape(-1, 4):
        out += mixed_derivative_peak(B, A, th, B0, g)
    return out + _evaluate_baseline(B, baseline)

def _xy(B, y):
    B = np.asarray(B, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(B) & np.isfinite(y)
    B, y = B[m], y[m]
    order = np.argsort(B)
    B, y = B[order], y[order]
    keep = np.r_[True, np.diff(B) > 0]
    B, y = B[keep], y[keep]
    if len(B) < 30 or B[-1] <= B[0]:
        raise ValueError("Need at least 30 valid magnetic-field points.")
    return B, y


def estimate_noise(y):
    y = np.asarray(y, float)
    d = np.diff(y)
    if len(d) < 2:
        return max(float(np.ptp(y)) * 1e-5, 1e-15)
    sigma = np.median(np.abs(d - np.median(d))) / (0.67448975 * np.sqrt(2.0))
    return float(max(sigma, np.ptp(y) * 1e-6, 1e-15))


def _downsample(B, y, maxn=MAX_FIT_POINTS):
    if len(B) <= maxn:
        return B, y
    idx = np.unique(np.linspace(0, len(B) - 1, maxn).astype(int))
    return B[idx], y[idx]


def _odd_window(n, desired):
    w = min(int(desired), n if n % 2 else n - 1)
    if w % 2 == 0:
        w -= 1
    return max(5, w)


def _smooth(y, desired=51):
    w = _odd_window(len(y), desired)
    if w < 5:
        return np.asarray(y, float).copy()
    return savgol_filter(y, w, min(3, w - 2), mode="interp")


def _baseline_columns(B, degree=BASELINE_DEGREE):
    mid = 0.5 * (B.min() + B.max())
    half = max(0.5 * (B.max() - B.min()), 1e-12)
    t = (B - mid) / half
    return [t ** k for k in range(degree + 1)]


def _linear_design(B, centers, gammas, model, baseline_degree=BASELINE_DEGREE):
    cols = []
    if model == "mixed":
        for c, g in zip(centers, gammas):
            cols.extend([
                lorentzian(B, 1.0, c, g),
                dispersive_lorentzian(B, 1.0, c, g),
            ])
    else:
        for c, g in zip(centers, gammas):
            cols.append(derivative_lorentzian(B, 1.0, c, g))
    cols.extend(_baseline_columns(B, baseline_degree))
    return np.column_stack(cols)


def _solve_linear(B, y, centers, gammas, model, baseline_degree=BASELINE_DEGREE):
    X = _linear_design(B, centers, gammas, model, baseline_degree)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coef, X @ coef


def _encode_params(centers, gammas, coef, model):
    blocks = []
    if model == "mixed":
        for i, (c, g) in enumerate(zip(centers, gammas)):
            a_abs, a_disp = coef[2 * i:2 * i + 2]
            A = float(np.hypot(a_abs, a_disp))
            theta = float(np.arctan2(a_disp, a_abs))
            blocks.append([A, theta, float(c), float(g)])
        base = coef[2 * len(centers):]
    else:
        for i, (c, g) in enumerate(zip(centers, gammas)):
            blocks.append([float(coef[i]), float(c), float(g)])
        base = coef[len(centers):]

    center_col = 2 if model == "mixed" else 1
    order = np.argsort([b[center_col] for b in blocks])
    blocks = np.asarray(blocks, float)[order]

    # Public models historically expose only c0,c1. Fold the higher-order
    # baseline into fitting internally but return its linear approximation.
    # Re-solve c0,c1 later on the full curve in _public_params_from_internal.
    return blocks, np.asarray(base, float)


def _public_params_from_internal(B, y, centers, gammas, coef, model):
    blocks, base = _encode_params(centers, gammas, coef, model)
    # Keep all four polynomial baseline terms so the curve displayed by the GUI
    # is exactly the curve optimized by the fitter.
    return np.r_[blocks.ravel(), base]

def _decode_seed(p, model):
    p = np.asarray(p, float)
    block_size = 4 if model == "mixed" else 3
    blocks, _ = _split_blocks_and_baseline(p, block_size)
    b = blocks.reshape(-1, block_size)
    if model == "mixed":
        return b[:, 2], np.abs(b[:, 3])
    return b[:, 1], np.abs(b[:, 2])

def _broad_detrend(B, y):
    n = len(y)
    desired = min(1001, max(101, (n // 3) | 1))
    w = _odd_window(n, desired)
    baseline = savgol_filter(y, w, 3, mode="interp")
    return y - baseline, baseline


def _mixed_matched_candidates(B, y, max_candidates=20):
    """Find AM-ESR candidates using a 2D absorption/dispersion matched filter."""
    B, y = _xy(B, y)
    span = float(B[-1] - B[0])
    step = float(np.median(np.diff(B)))
    z, _ = _broad_detrend(B, y)
    z = _smooth(z, 31)
    # Candidate search is only initialization, so a moderate downsample is safe
    # and makes batch fitting much faster.
    Bs, zs = _downsample(B, z, maxn=520)
    zn = np.linalg.norm(zs) + 1e-30

    grid = np.linspace(B[0] + 0.008 * span, B[-1] - 0.008 * span, 280)
    gmin = max(DEFAULT_GAMMA_BOUNDS[0], 3.0 * step)
    gmax = min(DEFAULT_GAMMA_BOUNDS[1], 0.12 * span, 0.20)
    widths = np.geomspace(gmin, gmax, 10)

    scored = []
    for c in grid:
        best_score = -np.inf
        best_g = None
        for g in widths:
            qa = lorentzian(Bs, 1.0, c, g)
            qd = dispersive_lorentzian(Bs, 1.0, c, g)
            # Fast exact projection onto span{qa,qd} using a 2x2 normal matrix.
            aa = float(qa @ qa); dd = float(qd @ qd); ad = float(qa @ qd)
            az = float(qa @ zs); dz = float(qd @ zs)
            det = aa * dd - ad * ad
            if det <= 1e-30:
                continue
            ca = (az * dd - dz * ad) / det
            cd = (dz * aa - az * ad) / det
            projection = ca * qa + cd * qd
            score = np.linalg.norm(projection) / zn
            if score > best_score:
                best_score, best_g = score, g
        scored.append((float(c), float(best_score), float(best_g)))

    scored.sort(key=lambda q: q[1], reverse=True)
    selected = []
    for q in scored:
        # Do not allow both sides/tails of one resonance to become separate seeds.
        sep = max(10.0 * step, 0.035 * span,
                  0.9 * (q[2] + (selected[0][2] if selected else q[2])))
        if all(abs(q[0] - r[0]) > max(10.0 * step, 0.035 * span,
                                      0.9 * (q[2] + r[2])) for r in selected):
            selected.append(q)
        if len(selected) >= max_candidates:
            break
    return selected


def _derivative_candidates(B, y, max_candidates=20):
    # Kept for legacy derivative mode.
    B, y = _xy(B, y)
    z, _ = _broad_detrend(B, y)
    z = _smooth(z, 31)
    span = float(B[-1] - B[0])
    step = float(np.median(np.diff(B)))
    grid = np.linspace(B[0] + .01 * span, B[-1] - .01 * span,
                       min(500, max(250, len(B) // 5)))
    widths = np.geomspace(max(DEFAULT_GAMMA_BOUNDS[0], 3 * step),
                          min(.20, .12 * span), 12)
    zn = np.linalg.norm(z) + 1e-30
    scored = []
    for c in grid:
        best = (-np.inf, None)
        for g in widths:
            q = derivative_lorentzian(B, 1.0, c, g)
            score = abs(np.dot(z, q)) / (np.linalg.norm(q) * zn + 1e-30)
            if score > best[0]:
                best = (score, g)
        scored.append((float(c), float(best[0]), float(best[1])))
    scored.sort(key=lambda q: q[1], reverse=True)
    selected = []
    for q in scored:
        if all(abs(q[0] - r[0]) > max(10 * step, .035 * span,
                                      .9 * (q[2] + r[2])) for r in selected):
            selected.append(q)
        if len(selected) >= max_candidates:
            break
    return selected


def _candidates(B, y, model, max_candidates=20):
    return (_mixed_matched_candidates(B, y, max_candidates)
            if model == "mixed"
            else _derivative_candidates(B, y, max_candidates))


def auto_find_peaks(B, y, npeaks=2, model="mixed"):
    B, y = _xy(B, y)
    return sorted(q[0] for q in _candidates(B, y, model, max(2, npeaks))[:npeaks])


def build_initial_guess(B, y, npeaks, mixed=False, centers=None):
    B, y = _xy(B, y)
    model = "mixed" if mixed else "derivative"
    cand = _candidates(B, y, model, max(10, npeaks + 4))
    if centers is None:
        centers = [q[0] for q in cand[:npeaks]]
    gammas = []
    for c in centers:
        near = min(cand, key=lambda q: abs(q[0] - c)) if cand else (c, 0, .04)
        gammas.append(float(near[2] or .04))
    coef, _ = _solve_linear(B, y, np.asarray(centers), np.asarray(gammas), model)
    return _public_params_from_internal(B, y, np.asarray(centers), np.asarray(gammas), coef, model)


def _fit_with_seed(B, y, centers0, gammas0, model, localize=True, max_nfev=2500,
                   max_fit_points=MAX_FIT_POINTS, center_bounds=None):
    B, y = _xy(B, y)
    Bd, yd = _downsample(B, y, maxn=max_fit_points)
    centers0 = np.asarray(centers0, float)
    gammas0 = np.asarray(gammas0, float)
    n = len(centers0)
    span = float(Bd[-1] - Bd[0])
    step = float(np.median(np.diff(Bd)))
    gl = max(DEFAULT_GAMMA_BOUNDS[0], 1.5 * step)
    gu = min(DEFAULT_GAMMA_BOUNDS[1], 0.25 * span)

    order = np.argsort(centers0)
    centers0 = centers0[order]
    gammas0 = np.clip(gammas0[order], gl, gu)

    if center_bounds is not None:
        # Manual-center mode: each selected center receives an explicit hard
        # optimization interval.  This prevents several components from
        # collapsing onto the strongest resonance.
        cb = np.asarray(center_bounds, dtype=float)
        if cb.shape != (n, 2):
            raise ValueError("center_bounds must have shape (n_components, 2).")
        cb = cb[order]
        c_lo = np.maximum(Bd[0], cb[:, 0])
        c_hi = np.minimum(Bd[-1], cb[:, 1])
        # Preserve the order of nearby manual picks by splitting overlapping
        # intervals at the midpoint between the selected centers.
        for i in range(n - 1):
            midpoint = 0.5 * (centers0[i] + centers0[i + 1])
            c_hi[i] = min(c_hi[i], midpoint - 0.5 * step)
            c_lo[i + 1] = max(c_lo[i + 1], midpoint + 0.5 * step)
    elif localize:
        radii = np.clip(np.maximum(0.10 * span, 4.0 * gammas0),
                        0.08 * span, 0.22 * span)
        c_lo = np.maximum(Bd[0], centers0 - radii)
        c_hi = np.minimum(Bd[-1], centers0 + radii)
        for i in range(n - 1):
            midpoint = 0.5 * (centers0[i] + centers0[i + 1])
            c_hi[i] = min(c_hi[i], midpoint - step)
            c_lo[i + 1] = max(c_lo[i + 1], midpoint + step)
    else:
        c_lo = np.full(n, Bd[0])
        c_hi = np.full(n, Bd[-1])

    for i in range(n):
        if c_hi[i] <= c_lo[i] + 2 * step:
            c_lo[i] = max(Bd[0], centers0[i] - 3 * step)
            c_hi[i] = min(Bd[-1], centers0[i] + 3 * step)

    sigma = estimate_noise(yd)
    v0 = np.r_[np.clip(centers0, c_lo + 1e-10, c_hi - 1e-10),
               np.log(gammas0)]
    lb = np.r_[c_lo, np.full(n, np.log(gl))]
    ub = np.r_[c_hi, np.full(n, np.log(gu))]

    def residual(v):
        c = v[:n]
        g = np.exp(v[n:])
        _, yf = _solve_linear(Bd, yd, c, g, model)
        return (yf - yd) / sigma

    result = least_squares(
        residual,
        np.clip(v0, lb + 1e-10, ub - 1e-10),
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_nfev,
        x_scale="jac",
    )
    if not result.success:
        raise RuntimeError("Nonlinear fit did not converge.")

    centers = result.x[:n]
    gammas = np.exp(result.x[n:])
    coef, fitted_internal = _solve_linear(B, y, centers, gammas, model)
    popt = _public_params_from_internal(B, y, centers, gammas, coef, model)
    public_func = multi_peak_mixed if model == "mixed" else multi_peak_derivative
    residual_public = y - public_func(B, *popt)
    rss = float(residual_public @ residual_public)
    return rss, popt, result


def _candidate_not_near_existing(candidate, existing_centers, existing_gammas, span, step):
    c = float(candidate[0])
    for ec, eg in zip(existing_centers, existing_gammas):
        exclusion = max(10.0 * step, 0.035 * span, 1.25 * float(eg))
        if abs(c - float(ec)) < exclusion:
            return False
    return True


def _sequential_seed(B, y, n, model, initial_seed=None):
    B, y = _xy(B, y)
    span = float(B[-1] - B[0])
    step = float(np.median(np.diff(B)))

    seed_centers = None
    seed_gammas = None
    if initial_seed is not None:
        try:
            c0, g0 = _decode_seed(initial_seed, model)
            if len(c0) == n:
                return np.asarray(c0, float), np.asarray(g0, float)
            if 0 < len(c0) < n:
                seed_centers = list(map(float, c0))
                seed_gammas = list(map(float, g0))
        except Exception:
            pass

    candidates = _candidates(B, y, model, 24)
    if not candidates:
        centers = np.linspace(B[0] + .2 * span, B[-1] - .2 * span, n)
        return centers, np.full(n, min(.04, .04 * span))

    if seed_centers is not None:
        centers = seed_centers
        gammas = seed_gammas
    else:
        centers = [float(candidates[0][0])]
        gammas = [float(candidates[0][2] or .04)]

    while len(centers) < n:
        try:
            _, partial_p, _ = _fit_with_seed(
                B, y, centers, gammas, model, localize=True, max_nfev=1600
            )
            centers_arr, gammas_arr = _decode_seed(partial_p, model)
            centers = list(map(float, centers_arr))
            gammas = list(map(float, gammas_arr))
            func = multi_peak_mixed if model == "mixed" else multi_peak_derivative
            residual = y - func(B, *partial_p)
        except Exception:
            residual = y.copy()

        residual_candidates = _candidates(B, residual, model, 30)
        chosen = None
        for q in residual_candidates:
            if _candidate_not_near_existing(q, centers, gammas, span, step):
                chosen = q
                break
        if chosen is None:
            for q in candidates:
                if _candidate_not_near_existing(q, centers, gammas, span, step):
                    chosen = q
                    break
        if chosen is None:
            edges = [B[0]] + sorted(centers) + [B[-1]]
            gaps = np.diff(edges)
            j = int(np.argmax(gaps))
            chosen = (0.5 * (edges[j] + edges[j + 1]), 0.0, min(.04, .04 * span))

        centers.append(float(chosen[0]))
        gammas.append(float(chosen[2] or .04))

    order = np.argsort(centers)
    return np.asarray(centers)[order], np.asarray(gammas)[order]


def _fit_n(B, y, n, model, seed=None, residual_centers=None, fast=False, center_bounds=None):
    B, y = _xy(B, y)
    span = float(B[-1] - B[0])
    step = float(np.median(np.diff(B)))

    c_seq, g_seq = _sequential_seed(B, y, n, model, initial_seed=seed)
    starts = [(c_seq, g_seq)]

    # Single-spectrum mode tries many alternative initializations. Batch mode
    # relies mainly on the previous-angle warm start and is much faster.
    if not fast and center_bounds is None:
        cand = _candidates(B, y, model, 18)
        diverse = []
        for q in cand:
            if all(abs(q[0] - r[0]) > max(10 * step, .035 * span,
                                          1.0 * (q[2] + r[2])) for r in diverse):
                diverse.append(q)
        max_combinations = 28
        for comb in list(combinations(diverse[:9], n))[:max_combinations]:
            starts.append((
                np.asarray([q[0] for q in comb], float),
                np.asarray([q[2] for q in comb], float),
            ))

    best = None
    for c0, g0 in starts:
        try:
            rss, popt, result = _fit_with_seed(
                B, y, c0, g0, model, True,
                700 if fast else 2200,
                BATCH_FIT_POINTS if fast else MAX_FIT_POINTS,
                center_bounds=center_bounds,
            )
            rows = extract_peak_table(popt, model)
            centers = np.asarray([r["B0"] for r in rows])
            gammas = np.asarray([r["gamma"] for r in rows])
            distinct = True
            for i in range(len(rows) - 1):
                min_sep = max(8 * step, .018 * span,
                              .40 * (gammas[i] + gammas[i + 1]))
                if centers[i + 1] - centers[i] < min_sep:
                    distinct = False
                    break
            if not distinct:
                continue
            # Reject a solution where one component becomes a very broad
            # background surrogate while the other components are narrow ESR lines.
            if len(gammas) >= 3:
                med_g = float(np.median(gammas))
                if np.max(gammas) > max(0.085, 2.5 * med_g):
                    continue
            if best is None or rss < best[0]:
                best = (rss, popt, result)
        except Exception:
            continue

    if best is None:
        best = _fit_with_seed(
            B, y, c_seq, g_seq, model, True,
            1000 if fast else 3000,
            BATCH_FIT_POINTS if fast else MAX_FIT_POINTS,
            center_bounds=center_bounds,
        )
    return best[1], np.full((len(best[1]), len(best[1])), np.nan)


def _metrics(B, y, p, model):
    func = multi_peak_mixed if model == "mixed" else multi_peak_derivative
    residual = y - func(B, *p)
    rss = float(residual @ residual)
    n = len(y)
    k = len(p)
    bic = n * np.log(max(rss / n, 1e-300)) + k * np.log(n)
    aic = n * np.log(max(rss / n, 1e-300)) + 2 * k
    aicc = aic + 2 * k * (k + 1) / max(1, n - k - 1)
    return dict(
        residual=residual,
        rss=rss,
        rmse=float(np.sqrt(rss / n)),
        bic=float(bic),
        aicc=float(aicc),
        noise_sigma=estimate_noise(residual),
    )


def extract_peak_table(popt, model="derivative"):
    p = np.asarray(popt, float)
    rows = []
    if model == "mixed":
        blocks, _ = _split_blocks_and_baseline(p, 4)
        blocks = blocks.reshape(-1, 4)
        blocks = blocks[np.argsort(blocks[:, 2])]
        for i, (A, th, B0, g) in enumerate(blocks, 1):
            A_abs = float(A * np.cos(th))
            A_disp = float(A * np.sin(th))
            # Absorptive integrated area. Signed and absolute forms are exported.
            area_abs = float(np.pi * abs(g) * A_abs)
            rows.append(dict(
                peak=i,
                A=float(A),
                theta=float(th),
                A_abs=A_abs,
                A_disp=A_disp,
                B0=float(B0),
                gamma=float(abs(g)),
                intensity=float(abs(area_abs)),
                signed_absorption_area=area_abs,
            ))
    else:
        blocks, _ = _split_blocks_and_baseline(p, 3)
        blocks = blocks.reshape(-1, 3)
        blocks = blocks[np.argsort(blocks[:, 1])]
        for i, (A, B0, g) in enumerate(blocks, 1):
            rows.append(dict(
                peak=i, A=float(A), B0=float(B0), gamma=float(abs(g)),
                intensity=float(abs(A) * np.pi),
            ))
    total = sum(r["intensity"] for r in rows)
    for r in rows:
        r["intensity_fraction"] = r["intensity"] / total if total else np.nan
    return rows


def _valid(B, y, p, model):
    rows = extract_peak_table(p, model)
    noise = estimate_noise(y)
    span = float(B[-1] - B[0])
    step = float(np.median(np.diff(B)))
    centers = []
    for r in rows:
        if not (B[0] < r["B0"] < B[-1]):
            return False
        if not (max(DEFAULT_GAMMA_BOUNDS[0], step) < r["gamma"] < min(DEFAULT_GAMMA_BOUNDS[1], .25 * span) * .995):
            return False
        if model == "mixed":
            comp = mixed_derivative_peak(B, r["A"], r["theta"], r["B0"], r["gamma"])
        else:
            comp = derivative_lorentzian(B, r["A"], r["B0"], r["gamma"])
        if np.ptp(comp) < 3.0 * noise:
            return False
        centers.append(r["B0"])
    return not any(np.diff(sorted(centers)) < max(8 * step, .012 * span))


def _extra_component_is_credible(previous_trial, new_trial, model):
    """Reject extra mathematical components that merely absorb broad baseline.

    For the AM-ESR data used to validate this fitter, real weak domains are
    narrow, coherent resonances. Spurious extra components tended to be broad.
    The threshold scales with the widths of the already accepted main lines.
    """
    old_rows = extract_peak_table(previous_trial["p"], model)
    new_rows = extract_peak_table(new_trial["p"], model)
    old_centers = np.asarray([r["B0"] for r in old_rows])
    added = max(new_rows, key=lambda r: np.min(np.abs(old_centers - r["B0"])))
    main_median = float(np.median([r["gamma"] for r in old_rows]))
    width_limit = max(0.055, 2.2 * main_median)
    if added["gamma"] > width_limit:
        return False
    if added["intensity_fraction"] < 0.012:
        return False
    return True


def _added_component_local_evidence(B, y, old_trial, new_trial, model, sigma_threshold=4.0):
    """Measure whether the newly added line removes a real local residual feature."""
    old_rows = extract_peak_table(old_trial["p"], model)
    new_rows = extract_peak_table(new_trial["p"], model)
    old_centers = np.asarray([r["B0"] for r in old_rows], float)
    added = max(new_rows, key=lambda r: np.min(np.abs(old_centers - r["B0"])))
    func = multi_peak_mixed if model == "mixed" else multi_peak_derivative
    rold = y - func(B, *old_trial["p"])
    rnew = y - func(B, *new_trial["p"])
    half = max(5.0 * added["gamma"], 0.025 * (B[-1] - B[0]))
    m = np.abs(B - added["B0"]) <= half
    if np.sum(m) < 12:
        return False, 0.0, 0.0
    local_old = float(np.sqrt(np.mean(rold[m] ** 2)))
    local_new = float(np.sqrt(np.mean(rnew[m] ** 2)))
    improvement = (local_old - local_new) / max(local_old, 1e-30)
    noise = estimate_noise(rnew)
    signal_sigma = float(np.ptp(rold[m]) / max(noise, 1e-30))
    credible = signal_sigma >= sigma_threshold and improvement >= 0.22
    return credible, improvement, signal_sigma


def fit_curve(B, y, npeaks=4, model="mixed", p0=None, return_details=False,
              min_components=None, fast=False, weak_sigma=4.0,
              manual_centers=None, center_tolerance=0.08):
    """Fit one spectrum.

    ``npeaks`` is a maximum, not a forced count. For max >= 2 the fitter begins
    with two physical resonances and only accepts 3/4 when the added resonance is
    narrow, significant, distinct, and substantially improves the residual.
    """
    B, y = _xy(B, y)
    model = model.lower()
    if model not in ("derivative", "mixed"):
        raise ValueError("model must be 'derivative' or 'mixed'")
    maxc = int(np.clip(npeaks, 1, 4))
    minc = int(min_components if min_components is not None else (1 if maxc == 1 else 2))
    minc = min(minc, maxc)

    manual_bounds = None
    if manual_centers is not None:
        mc = np.sort(np.asarray(manual_centers, dtype=float))
        if len(mc) != minc or minc != maxc:
            raise ValueError("Manual centers require a forced component count matching the number of picks.")
        tol = float(center_tolerance)
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("Center tolerance must be a positive number in tesla.")
        manual_bounds = np.column_stack((mc - tol, mc + tol))

    trials = []
    for n in range(minc, maxc + 1):
        seed = p0 if (n == minc and p0 is not None) else None
        try:
            p, cov = _fit_n(B, y, n, model, seed=seed, fast=fast,
                            center_bounds=manual_bounds if n == minc else None)
            m = _metrics(B, y, p, model)
            trials.append(dict(n=n, p=p, cov=cov, m=m, valid=_valid(B, y, p, model)))
        except Exception:
            continue
    if not trials:
        raise RuntimeError("Automatic ESR fit failed.")

    pool = [t for t in trials if t["valid"]] or trials
    chosen = pool[0]
    selection_log = []
    for trial in pool[1:]:
        global_ratio = chosen["m"]["rmse"] / max(trial["m"]["rmse"], 1e-30)
        local_ok, local_improvement, local_sigma = _added_component_local_evidence(
            B, y, chosen, trial, model, sigma_threshold=weak_sigma
        )
        physical_ok = _extra_component_is_credible(chosen, trial, model)
        bic_ok = trial["m"]["bic"] < chosen["m"]["bic"] - 20
        accept = physical_ok and local_ok and (bic_ok or global_ratio > 1.05)
        selection_log.append(dict(
            from_n=chosen["n"], to_n=trial["n"], accepted=bool(accept),
            local_improvement=local_improvement, local_sigma=local_sigma,
            global_rmse_ratio=global_ratio, bic_delta=trial["m"]["bic"]-chosen["m"]["bic"],
        ))
        if accept:
            chosen = trial

    raw_noise = estimate_noise(y)
    ratio = chosen["m"]["rmse"] / raw_noise
    quality = "excellent" if ratio <= 2.5 else ("acceptable" if ratio <= 7 else "needs_review")
    details = dict(
        n_components=chosen["n"], model=model, success=True, quality=quality,
        **chosen["m"],
        tested_models=[dict(
            n_components=t["n"], bic=t["m"]["bic"], aicc=t["m"]["aicc"],
            rmse=t["m"]["rmse"], physically_valid=t["valid"]
        ) for t in trials],
        selection_log=selection_log, fast=bool(fast), weak_sigma=float(weak_sigma),
        manual_centers=(None if manual_centers is None else list(map(float, manual_centers))),
        center_tolerance=(None if manual_centers is None else float(center_tolerance)),
    )
    fit_curve.last_details = details
    if return_details:
        return chosen["p"], chosen["cov"], details
    return chosen["p"], chosen["cov"]


fit_curve.last_details = None



def _predict_track_B0(history, angle):
    """Linear prediction from the last two observations of one domain."""
    if not history:
        return np.nan
    if len(history) == 1:
        return float(history[-1][1])
    a1, b1 = history[-2][0], history[-2][1]
    a2, b2 = history[-1][0], history[-1][1]
    da = float(a2 - a1)
    if abs(da) < 1e-12:
        return float(b2)
    slope = float(b2 - b1) / da
    # Avoid wild extrapolation after a bad/missing point.
    slope = float(np.clip(slope, -0.08, 0.08))
    return float(b2 + slope * (angle - a2))


def track_domain_branches(series_fits, max_domains=4, max_gap=3):
    """Assign persistent physical domain IDs across an angular series.

    Tracking is performed independently for each sweep direction.  A domain is
    predicted from its previous one or two B0 values and matched to the new
    fitted components with a Hungarian assignment.  Missing domains create
    gaps instead of causing a new color/ID.  Only IDs 1..``max_domains`` are
    ever used.

    The function mutates and returns ``series_fits``.
    """
    groups = {}
    for s in series_fits:
        groups.setdefault(str(s.get("direction", "unknown")).lower(), []).append(s)

    for _direction, group in groups.items():
        ordered = sorted(group, key=lambda q: (
            float(q.get("angle", np.nan)),
            str(q.get("name", q.get("filename", ""))),
        ))
        tracks = {i: {"history": [], "gamma": None, "intensity": None,
                      "last_index": -10**9} for i in range(1, max_domains + 1)}

        for k, s in enumerate(ordered):
            if (not s.get("success")) or str(s.get("review_status", "")).lower() == "excluded":
                continue
            angle = float(s.get("angle", k))
            rows = sorted(s.get("peaks") or extract_peak_table(s["popt"], s["details"]["model"]),
                          key=lambda r: float(r["B0"]))[:max_domains]
            if not rows:
                s["peaks"] = []
                continue

            # Candidate active tracks: keep domains alive across a few missing angles.
            active_ids = [did for did, tr in tracks.items()
                          if tr["history"] and (k - tr["last_index"] <= max_gap + 1)]
            assigned = {}
            used_rows = set()

            if active_ids:
                pred = np.array([_predict_track_B0(tracks[did]["history"], angle)
                                 for did in active_ids], float)
                C = np.array([float(r["B0"]) for r in rows], float)
                G = np.array([max(float(r.get("gamma", .02)), 1e-6) for r in rows], float)
                I = np.array([max(float(r.get("intensity", 1e-12)), 1e-12) for r in rows], float)
                span = max(float(np.ptp(C)) if len(C) > 1 else .5, .3)
                cost = np.empty((len(active_ids), len(rows)), float)
                for ii, did in enumerate(active_ids):
                    tr = tracks[did]
                    pg = max(float(tr["gamma"] or np.median(G)), 1e-6)
                    pi = max(float(tr["intensity"] or np.median(I)), 1e-12)
                    # B0 continuity dominates. Width and intensity only break ties.
                    cost[ii] = ((C - pred[ii]) / max(.035, .12 * span)) ** 2
                    cost[ii] += .08 * ((np.log(G) - np.log(pg)) / 1.0) ** 2
                    cost[ii] += .025 * ((np.log(I) - np.log(pi)) / 1.5) ** 2
                ri, ci = linear_sum_assignment(cost)
                for ii, jj in zip(ri, ci):
                    did = active_ids[ii]
                    # Gate implausible jumps; leave track missing instead of swapping.
                    max_jump = max(.12, 5.0 * float(rows[jj].get("gamma", .02)))
                    if abs(float(rows[jj]["B0"]) - pred[ii]) <= max_jump:
                        assigned[jj] = did
                        used_rows.add(jj)

            # Assign newly visible components to unused IDs.  Preserve field order
            # relative to already assigned tracks as much as possible.
            free_ids = [did for did in range(1, max_domains + 1)
                        if did not in assigned.values()]
            unassigned_rows = [j for j in range(len(rows)) if j not in used_rows]
            if unassigned_rows and free_ids:
                # Estimate each free ID's expected field from any older history.
                with_history = [did for did in free_ids if tracks[did]["history"]]
                no_history = [did for did in free_ids if not tracks[did]["history"]]
                for j in list(unassigned_rows):
                    if not with_history:
                        break
                    c = float(rows[j]["B0"])
                    did = min(with_history,
                              key=lambda d: abs(c - _predict_track_B0(tracks[d]["history"], angle)))
                    if abs(c - _predict_track_B0(tracks[did]["history"], angle)) <= .20:
                        assigned[j] = did
                        with_history.remove(did)
                        free_ids.remove(did)
                        unassigned_rows.remove(j)
                # Truly new tracks receive remaining IDs in B0 order.
                for j, did in zip(unassigned_rows, sorted(free_ids)):
                    assigned[j] = did

            for j, row in enumerate(rows):
                did = int(assigned.get(j, min(j + 1, max_domains)))
                row["domain_id"] = did
                row["peak"] = did
                tr = tracks[did]
                tr["history"].append((angle, float(row["B0"])))
                tr["history"] = tr["history"][-4:]
                tr["gamma"] = float(row.get("gamma", tr["gamma"] or .02))
                tr["intensity"] = float(row.get("intensity", tr["intensity"] or 1e-12))
                tr["last_index"] = k
            s["peaks"] = rows

        # Automatic quality screening after branch assignment.  Manual
        # accepted/excluded labels are preserved; otherwise suspicious fits
        # are marked needs_review with human-readable reasons.
        previous_by_domain = {}
        for s in ordered:
            if not s.get("success") or str(s.get("review_status", "")).lower() == "excluded":
                continue
            reasons = []
            details = s.get("details", {})
            if str(details.get("quality", "")).lower() == "needs_review":
                reasons.append("optimizer quality flag")
            rmse = float(details.get("rmse", np.nan))
            noise = float(details.get("noise_sigma", details.get("noise", np.nan)))
            if np.isfinite(rmse) and np.isfinite(noise) and noise > 0 and rmse > 3.0 * noise:
                reasons.append(f"RMSE is {rmse/noise:.1f}× noise")
            peaks = s.get("peaks", [])
            centers = sorted(float(q["B0"]) for q in peaks)
            if len(centers) > 1 and np.min(np.diff(centers)) < 0.012:
                reasons.append("two fitted centers nearly collapsed")
            for q in peaks:
                gamma = float(q.get("gamma", np.nan))
                if np.isfinite(gamma) and (gamma < 0.0035 or gamma > 0.40):
                    reasons.append(f"Domain {q.get('domain_id', '?')} linewidth near limit")
                did = int(q.get("domain_id", q.get("peak", 0)))
                if did in previous_by_domain:
                    jump = abs(float(q["B0"]) - previous_by_domain[did])
                    if jump > 0.16:
                        reasons.append(f"Domain {did} B0 jump {jump:.3f} T")
                previous_by_domain[did] = float(q["B0"])
            # Deduplicate while preserving order.
            reasons = list(dict.fromkeys(reasons))
            s["review_reasons"] = reasons
            current = str(s.get("review_status", "unreviewed")).lower()
            if current not in ("accepted", "excluded"):
                s["review_status"] = "needs_review" if reasons else "unreviewed"
    return series_fits


def fit_spectrum_series(spectra, npeaks=4, model="mixed", min_components=None,
                        weak_sigma=4.0, fast_batch=True):
    norm = []
    for i, s in enumerate(spectra):
        if isinstance(s, dict):
            norm.append({**s, "_i": i})
        else:
            a, B, y = s
            norm.append(dict(angle=a, B=B, y=y, _i=i))
    norm.sort(key=lambda s: (
        _series_metadata_key(s),
        float(s.get("angle", s["_i"])),
        str(s.get("name", s.get("filename", ""))),
    ))

    outputs = []
    prev_p = None
    previous_group = None
    for series_index, s in enumerate(norm):
        current_group = _series_metadata_key(s)
        if current_group != previous_group:
            # Never warm-start one temperature/frequency/direction series from
            # another experimental condition.
            prev_p = None
        previous_group = current_group
        try:
            p, cov, details = fit_curve(
                s["B"], s["y"], npeaks=npeaks, model=model, p0=prev_p,
                return_details=True, min_components=min_components,
                fast=(fast_batch and series_index > 0), weak_sigma=weak_sigma,
            )
            rows = sorted(extract_peak_table(p, model), key=lambda r: r["B0"])[:4]
            outputs.append({**s, "success": True, "popt": p, "pcov": cov,
                            "details": details, "peaks": rows})
            prev_p = p
        except Exception as exc:
            outputs.append({**s, "success": False, "error": str(exc), "peaks": []})
            prev_p = None

    # Persistent branch IDs, not left-to-right IDs at every angle.
    track_domain_branches(outputs, max_domains=4)
    outputs.sort(key=lambda s: s["_i"])
    return outputs


def rebuild_fit_rows(series_fits, frequency_GHz):
    """Create export/plot rows from the current detailed series fits."""
    rows = []
    for fit in series_fits:
        if not fit.get("success") or str(fit.get("review_status", "")).lower() == "excluded":
            continue
        info = fit.get("details", {})
        direction = str(fit.get("direction", "unknown")).lower()
        fit_frequency = fit.get("frequency_GHz", frequency_GHz)
        if fit_frequency is None or not np.isfinite(float(fit_frequency)):
            fit_frequency = frequency_GHz
        for peak in fit.get("peaks", []):
            row = dict(
                angle=float(fit["angle"]), direction=direction,
                measurement_id=str(fit.get("measurement_id", fit.get("name", fit.get("filename", "")))),
                filename=str(fit.get("name", fit.get("filename", ""))),
                temperature_K=fit.get("temperature_K", np.nan),
                frequency_GHz=float(fit_frequency),
                peak=int(peak["domain_id"]), domain_id=int(peak["domain_id"]),
                B0=float(peak["B0"]), gamma=float(peak["gamma"]),
                A=float(peak.get("A", np.nan)),
                intensity=float(peak.get("intensity", np.nan)),
                intensity_fraction=float(peak.get("intensity_fraction", np.nan)),
                g=g_factor(float(fit_frequency), float(peak["B0"])),
                model=info.get("model", "mixed"),
                npeaks=int(info.get("n_components", len(fit.get("peaks", [])))),
                rmse=info.get("rmse", np.nan), bic=info.get("bic", np.nan),
                fit_quality=info.get("quality", "unknown"),
            )
            if "theta" in peak:
                row["theta"] = float(peak["theta"])
            rows.append(row)
    return rows

def g_factor(frequency_GHz, B0_T):
    return H * (frequency_GHz * 1e9) / (MU_B * B0_T)


def fit_all_angles(angle_curves, frequency_GHz, npeaks=4, model="mixed",
                   min_components=None, weak_sigma=4.0, fast_batch=True):
    series = []
    for curve in angle_curves:
        if isinstance(curve, dict):
            series.append(curve)
        else:
            angle, B, y = curve
            series.append(dict(angle=angle, B=B, y=y))
    fitted = fit_spectrum_series(
        series, npeaks=npeaks, model=model, min_components=min_components,
        weak_sigma=weak_sigma, fast_batch=fast_batch,
    )
    return rebuild_fit_rows(fitted, frequency_GHz)

def auto_fit_window(B, y, margin_T=0.25, sigma_threshold=4.0,
                    min_width_T=1.0, fallback=(2.5, 4.6)):
    """Estimate a useful fit interval from broad-baseline-subtracted signal.

    The detector is intentionally conservative: it finds all points where the
    detrended signal exceeds ``sigma_threshold`` times the noise, then expands
    the interval by ``margin_T``. If detection is unreliable, it uses the
    supplied fallback clipped to the available field range.
    """
    B, y = _xy(B, y)
    z, _ = _broad_detrend(B, y)
    zs = _smooth(z, 31)
    sigma = estimate_noise(zs)
    active = np.abs(zs) >= float(sigma_threshold) * sigma
    if np.any(active):
        idx = np.flatnonzero(active)
        lo = float(B[idx[0]] - margin_T)
        hi = float(B[idx[-1]] + margin_T)
        if hi - lo >= min_width_T:
            return max(float(B[0]), lo), min(float(B[-1]), hi)
    lo, hi = fallback
    lo = max(float(B[0]), float(lo))
    hi = min(float(B[-1]), float(hi))
    if hi <= lo:
        return float(B[0]), float(B[-1])
    return lo, hi


def evaluate_fit_curve(B, popt, model="mixed"):
    """Evaluate a public fit parameter vector."""
    func = multi_peak_mixed if model == "mixed" else multi_peak_derivative
    return func(np.asarray(B, float), *np.asarray(popt, float))

# ================================================================
# Reference-anchored domain tracking (overrides earlier definition)
# ================================================================

def _tracking_rows(s, max_domains=4):
    rows = s.get("peaks") or extract_peak_table(s["popt"], s["details"]["model"])
    return sorted(rows, key=lambda r: float(r["B0"]))[:max_domains]


def _series_metadata_key(s):
    """Condition key used by fitting and tracking.

    Rounded values avoid floating-point header noise while still ensuring that
    different temperatures, frequencies, and sweep directions are never
    treated as one angular series.
    """
    def rounded(value):
        try:
            value = float(value)
            return round(value, 3) if np.isfinite(value) else None
        except Exception:
            return None
    return (
        rounded(s.get("temperature_K")),
        rounded(s.get("frequency_GHz")),
        str(s.get("direction", "unknown")).lower(),
    )


def _assign_reference_rows(rows, max_domains=4):
    """Validate/preserve a manually named 1--4-domain reference spectrum."""
    if not 1 <= len(rows) <= max_domains:
        raise ValueError(f"Reference must contain between 1 and {max_domains} fitted domains.")
    used = set()
    for row in rows:
        did = int(row.get("domain_id", 0) or 0)
        if did < 1 or did > max_domains or did in used:
            raise ValueError("Reference domain IDs must be unique values from 1 to 4.")
        used.add(did)
        row["peak"] = did
        row["domain_locked"] = True
    return rows


def _track_one_direction_from_reference(ordered, ref_index, max_domains=4, max_gap=3):
    """Track both forward and backward from one manually chosen reference."""
    ref = ordered[ref_index]
    ref_rows = _tracking_rows(ref, max_domains=max_domains)
    _assign_reference_rows(ref_rows, max_domains=max_domains)
    ref["peaks"] = ref_rows

    def make_tracks():
        tracks = {i: {"history": [], "gamma": None, "intensity": None,
                      "last_step": -10**9} for i in range(1, max_domains + 1)}
        angle = float(ref.get("angle", ref_index))
        for row in ref_rows:
            did = int(row["domain_id"])
            tracks[did]["history"] = [(angle, float(row["B0"]))]
            tracks[did]["gamma"] = float(row.get("gamma", .02))
            tracks[did]["intensity"] = float(row.get("intensity", 1e-12))
            tracks[did]["last_step"] = 0
        return tracks

    def process(indices, tracks):
        for step, k in enumerate(indices, start=1):
            s = ordered[k]
            if (not s.get("success")) or str(s.get("review_status", "")).lower() == "excluded":
                continue
            angle = float(s.get("angle", k))
            rows = _tracking_rows(s, max_domains=max_domains)
            if not rows:
                s["peaks"] = []
                continue

            assigned = {}
            used_rows = set()
            used_ids = set()

            # Any manually locked assignments at this spectrum are hard anchors.
            for j, row in enumerate(rows):
                if bool(row.get("domain_locked", False)):
                    did = int(row.get("domain_id", 0) or 0)
                    if 1 <= did <= max_domains and did not in used_ids:
                        assigned[j] = did
                        used_rows.add(j)
                        used_ids.add(did)

            active_ids = [did for did, tr in tracks.items()
                          if tr["history"] and did not in used_ids
                          and (step - tr["last_step"] <= max_gap + 1)]
            free_rows = [j for j in range(len(rows)) if j not in used_rows]

            if active_ids and free_rows:
                C = np.array([float(rows[j]["B0"]) for j in free_rows], float)
                G = np.array([max(float(rows[j].get("gamma", .02)), 1e-6) for j in free_rows], float)
                I = np.array([max(float(rows[j].get("intensity", 1e-12)), 1e-12) for j in free_rows], float)
                span = max(float(np.ptp(C)) if len(C) > 1 else .5, .3)
                cost = np.empty((len(active_ids), len(free_rows)), float)
                predictions = []
                for ii, did in enumerate(active_ids):
                    tr = tracks[did]
                    pred = _predict_track_B0(tr["history"], angle)
                    predictions.append(pred)
                    pg = max(float(tr["gamma"] or np.median(G)), 1e-6)
                    pi = max(float(tr["intensity"] or np.median(I)), 1e-12)
                    cost[ii] = ((C - pred) / max(.03, .10 * span)) ** 2
                    cost[ii] += .06 * ((np.log(G) - np.log(pg)) / 1.0) ** 2
                    cost[ii] += .015 * ((np.log(I) - np.log(pi)) / 1.7) ** 2
                ri, ci = linear_sum_assignment(cost)
                for ii, jj_local in zip(ri, ci):
                    j = free_rows[jj_local]
                    did = active_ids[ii]
                    pred = predictions[ii]
                    max_jump = max(.11, 5.0 * float(rows[j].get("gamma", .02)))
                    if abs(float(rows[j]["B0"]) - pred) <= max_jump:
                        assigned[j] = did
                        used_rows.add(j)
                        used_ids.add(did)

            # Unmatched components use still-free tracks. Prefer old predicted tracks;
            # truly new IDs are filled in field order.
            free_ids = [d for d in range(1, max_domains + 1) if d not in used_ids]
            unassigned_rows = [j for j in range(len(rows)) if j not in used_rows]
            old_free = [d for d in free_ids if tracks[d]["history"]]
            for j in list(unassigned_rows):
                if not old_free:
                    break
                c = float(rows[j]["B0"])
                did = min(old_free, key=lambda d: abs(c - _predict_track_B0(tracks[d]["history"], angle)))
                if abs(c - _predict_track_B0(tracks[did]["history"], angle)) <= .20:
                    assigned[j] = did
                    old_free.remove(did); free_ids.remove(did); unassigned_rows.remove(j)
            for j, did in zip(sorted(unassigned_rows, key=lambda q: float(rows[q]["B0"])), sorted(free_ids)):
                assigned[j] = did

            for j, row in enumerate(rows):
                did = int(assigned.get(j, min(j + 1, max_domains)))
                row["domain_id"] = did
                row["peak"] = did
                row.setdefault("domain_locked", False)
                tr = tracks[did]
                tr["history"].append((angle, float(row["B0"])))
                tr["history"] = tr["history"][-4:]
                tr["gamma"] = float(row.get("gamma", tr["gamma"] or .02))
                tr["intensity"] = float(row.get("intensity", tr["intensity"] or 1e-12))
                tr["last_step"] = step
            s["peaks"] = rows

    process(range(ref_index + 1, len(ordered)), make_tracks())
    process(range(ref_index - 1, -1, -1), make_tracks())


def track_domain_branches(series_fits, max_domains=4, max_gap=3):
    """Track persistent domain IDs, anchored to a user-selected reference.

    A spectrum marked ``tracking_reference=True`` may contain one through four
    fitted components with unique manually assigned domain IDs.  The
    tracker propagates those identities forward and backward in angle.  Any
    row marked ``domain_locked=True`` is treated as a hard anchor and is never
    reassigned.  If no reference is selected, the previous automatic behavior
    is approximated by choosing the successful spectrum with the largest
    number of resolved components and labeling it left-to-right once.
    """
    groups = {}
    for s in series_fits:
        groups.setdefault(_series_metadata_key(s), []).append(s)

    for _direction, group in groups.items():
        ordered = sorted(group, key=lambda q: (
            float(q.get("angle", np.nan)),
            str(q.get("name", q.get("filename", ""))),
        ))
        candidates = [i for i, s in enumerate(ordered)
                      if s.get("success") and str(s.get("review_status", "")).lower() != "excluded"]
        if not candidates:
            continue
        refs = [i for i in candidates if bool(ordered[i].get("tracking_reference", False))]
        if refs:
            ref_index = refs[0]
            rows = _tracking_rows(ordered[ref_index], max_domains=max_domains)
            if not 1 <= len(rows) <= max_domains:
                raise ValueError(
                    f"The tracking reference must contain between 1 and {max_domains} fitted domains."
                )
        else:
            # Safe fallback until the user chooses a reference explicitly.
            ref_index = max(candidates, key=lambda i: (len(_tracking_rows(ordered[i], max_domains)),
                                                       -float(ordered[i].get("details", {}).get("rmse", np.inf))))
            rows = _tracking_rows(ordered[ref_index], max_domains=max_domains)
            for did, row in enumerate(sorted(rows, key=lambda r: float(r["B0"])), 1):
                row["domain_id"] = did
                row["peak"] = did
                row.setdefault("domain_locked", False)
            ordered[ref_index]["peaks"] = rows

        _track_one_direction_from_reference(ordered, ref_index,
                                            max_domains=max_domains, max_gap=max_gap)

        # Review tracking continuity separately from fit quality.
        previous = {}
        for s in ordered:
            if not s.get("success") or str(s.get("review_status", "")).lower() == "excluded":
                continue
            warnings = []
            for row in s.get("peaks", []):
                did = int(row.get("domain_id", 0) or 0)
                if did in previous:
                    jump = abs(float(row["B0"]) - previous[did])
                    if jump > .16 and not bool(row.get("domain_locked", False)):
                        warnings.append(f"Domain {did} tracking jump {jump:.3f} T")
                previous[did] = float(row["B0"])
            s["tracking_reasons"] = warnings
            s["tracking_status"] = "needs_review" if warnings else "ok"
    return series_fits

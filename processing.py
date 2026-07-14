import numpy as np


# ======================
# Grid utilities
# ======================

def ensure_monotonic_unique(B, *ys):
    """
    Sort B and remove duplicate field values.

    Returns:
        B_unique,
        y1_unique,
        y2_unique,
        ...
    """

    B = np.asarray(B)

    order = np.argsort(B)

    B = B[order]

    ys_sorted = [
        np.asarray(y)[order]
        for y in ys
    ]

    Bu, idx = np.unique(
        B,
        return_index=True
    )

    ys_u = [
        y[idx]
        for y in ys_sorted
    ]

    return (Bu, *ys_u)


def choose_reference_grid(curves):
    """
    Choose the densest B grid.

    Used for SUM/AVG interpolation.
    """

    best = max(
        curves,
        key=lambda c: c["B"].size
    )

    return best["B"]


# ======================
# Baseline centering
# ======================

def baseline_center_cluster(cluster):
    """
    Median-center CH1 and CH2.

    Returns:
        new curve list
    """

    base1 = float(
        np.median(
            np.concatenate(
                [c["ch1"] for c in cluster]
            )
        )
    )

    all_ch2 = np.concatenate(
        [c["ch2"] for c in cluster]
    )

    if np.any(np.isfinite(all_ch2)):
        base2 = float(
            np.nanmedian(all_ch2)
        )
    else:
        base2 = 0.0

    curves = []

    for c in cluster:

        curves.append({

            **c,

            "y1":
                c["ch1"] - base1,

            "y2":
                c["ch2"] - base2

        })

    return curves


def baseline_center_curve(curve):
    """Center one measurement without using any other measurement.

    This is the only centering operation used by fitting.  It guarantees that
    repeated files at the same angle/temperature/frequency remain independent.
    """
    ch1 = np.asarray(curve["ch1"], float)
    ch2 = np.asarray(curve["ch2"], float)
    base1 = float(np.nanmedian(ch1))
    base2 = float(np.nanmedian(ch2)) if np.any(np.isfinite(ch2)) else 0.0
    return {**curve, "y1": ch1 - base1, "y2": ch2 - base2}


def baseline_center_individual(curves):
    """Return individually centered measurements; never average or combine."""
    return [baseline_center_curve(curve) for curve in curves]


# ======================
# Interpolation
# ======================

def interpolate_curve(
    Bref,
    B,
    y
):
    """
    Interpolate y onto Bref.
    """

    Bu, yu = ensure_monotonic_unique(
        B,
        y
    )

    return np.interp(
        Bref,
        Bu,
        yu
    )


def interpolate_curve_pair(
    Bref,
    B,
    y1,
    y2
):
    """
    Interpolate CH1 and CH2.
    """

    Bu, y1u, y2u = ensure_monotonic_unique(
        B,
        y1,
        y2
    )

    y1i = np.interp(
        Bref,
        Bu,
        y1u
    )

    if np.any(np.isfinite(y2u)):

        y2clean = y2u.copy()

        finite = np.isfinite(y2clean)

        if np.sum(finite) >= 2:

            y2clean[~finite] = np.interp(
                Bu[~finite],
                Bu[finite],
                y2clean[finite]
            )

            y2i = np.interp(
                Bref,
                Bu,
                y2clean
            )

        else:

            y2i = np.zeros_like(Bref)

    else:

        y2i = np.zeros_like(Bref)

    return y1i, y2i


# ======================
# SUM / AVG
# ======================

def compute_stack(
    curves,
    mode="sum"
):
    """
    Stack selected curves.

    mode:
        sum
        avg

    Returns:
        Bref
        CH1
        CH2
    """

    if not curves:
        return None, None, None

    Bref = choose_reference_grid(
        curves
    )

    Bref, _ = ensure_monotonic_unique(
        Bref,
        np.zeros_like(Bref)
    )

    Y1 = []
    Y2 = []

    for c in curves:

        y1i, y2i = interpolate_curve_pair(
            Bref,
            c["B"],
            c["y1"],
            c["y2"]
        )

        Y1.append(y1i)
        Y2.append(y2i)

    Y1 = np.vstack(Y1)
    Y2 = np.vstack(Y2)

    y1_sum = np.sum(
        Y1,
        axis=0
    )

    y2_sum = np.sum(
        Y2,
        axis=0
    )

    if mode.lower() == "avg":

        n = max(
            1,
            Y1.shape[0]
        )

        return (
            Bref,
            y1_sum / n,
            y2_sum / n
        )

    return (
        Bref,
        y1_sum,
        y2_sum
    )


def compute_sum(curves):
    return compute_stack(
        curves,
        mode="sum"
    )


def compute_avg(curves):
    return compute_stack(
        curves,
        mode="avg"
    )


# ======================
# Shifted plot helpers
# ======================

def auto_offset_step(curves):
    """
    Determine reasonable vertical spacing
    for shifted plots.
    """

    if not curves:
        return 1.0

    p2ps = [

        float(
            np.ptp(
                c["y1"]
            )
        )

        for c in curves
    ]

    typical = float(
        np.median(p2ps)
    )

    return 0.25 * typical


# ======================
# Direction helpers
# ======================

def split_up_down(curves):

    up = [
        c for c in curves
        if c["direction"] == "up"
    ]

    down = [
        c for c in curves
        if c["direction"] == "down"
    ]

    return up, down


# ======================
# Angle helpers
# ======================

def mean_angle(curves):

    if not curves:
        return np.nan

    return float(
        np.mean(
            [c["angle"] for c in curves]
        )
    )


def angle_span(curves):

    if not curves:
        return np.nan, np.nan

    angles = [
        c["angle"]
        for c in curves
    ]

    return (
        float(np.min(angles)),
        float(np.max(angles))
    )

import os
import csv
import numpy as np

from data_io import (
    make_export_header_from_original,
    write_dat_3col,
)


# =====================================================
# Helpers
# =====================================================

def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def short_file_list(names, max_show=12):

    names = list(names)

    if len(names) <= max_show:
        return str(names)

    head = names[:max_show]

    rest = len(names) - max_show

    return str(
        head[:-1]
        +
        [f"... (+{rest} more)"]
    )


# =====================================================
# SUM / AVG export
# =====================================================

def export_dat_curve(
    out_dir,
    mode_tag,
    freq_GHz,
    mean_angle,
    selected_files,
    selected_indices,
    original_header,
    B,
    y1,
    y2,
):
    """
    Export summed/averaged ESR curve.
    """

    safe_makedirs(out_dir)

    nsel = len(selected_files)

    base_name = (
        f"{mode_tag}"
        f"_f{freq_GHz:.1f}GHz"
        f"_ang{mean_angle:.2f}"
        f"_n{nsel}"
    ).replace(" ", "_")

    dat_path = os.path.join(
        out_dir,
        base_name + ".dat"
    )

    extra = [

        f"# ---- ESR {mode_tag} EXPORT ----",

        f"# FREQ_GHZ={freq_GHz}",

        f"# ANGLE_MEAN_DEG={mean_angle}",

        f"# MODE={mode_tag}",

        f"# N_SELECTED={nsel}",

        f"# SELECTED_INDICES={selected_indices}",

        f"# SELECTED_FILES={short_file_list(selected_files)}",

        "# Columns:",
        "# B(T)    CH1    CH2",
    ]

    header_lines = make_export_header_from_original(
        original_header,
        extra
    )

    write_dat_3col(
        dat_path,
        header_lines,
        B,
        y1,
        y2
    )

    return dat_path


# =====================================================
# PNG export
# =====================================================

def export_figure(
    fig,
    out_dir,
    filename,
    dpi=220
):

    safe_makedirs(out_dir)

    path = os.path.join(
        out_dir,
        filename
    )

    fig.savefig(
        path,
        dpi=dpi
    )

    return path


# =====================================================
# Fit result export
# =====================================================

def export_fit_results_csv(
    fit_results,
    out_dir,
    filename="fit_results.csv"
):
    """
    Save all fitted peaks.
    """

    safe_makedirs(out_dir)

    path = os.path.join(
        out_dir,
        filename
    )

    if not fit_results:
        return path

    fields = sorted({key for row in fit_results for key in row.keys()})

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for row in fit_results:
            writer.writerow(row)

    return path


def export_fitted_measurements(series_fits, fit_results, out_dir,
                               summary_filename="fit_results.csv"):
    """Save summary parameters and one measured/fit/residual CSV per file."""
    from fit_models import evaluate_fit_curve

    safe_makedirs(out_dir)
    curves_dir = os.path.join(out_dir, "fitted_measurements")
    safe_makedirs(curves_dir)
    summary = export_fit_results_csv(fit_results, out_dir, summary_filename)
    saved = []

    for index, fit in enumerate(series_fits, start=1):
        if not fit.get("success"):
            continue
        B = np.asarray(fit["B"], float)
        measured = np.asarray(fit["y"], float)
        model = fit.get("details", {}).get("model", "mixed")
        fitted = evaluate_fit_curve(B, fit["popt"], model=model)
        residual = measured - fitted
        name = str(fit.get("name", fit.get("measurement_id", f"measurement_{index:04d}")))
        stem = os.path.splitext(os.path.basename(name))[0]
        safe_stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
        angle = float(fit.get("angle", np.nan))
        direction = str(fit.get("direction", "unknown")).upper()
        filename = f"{index:04d}_{safe_stem}_{direction}_{angle:.3f}deg_fit.csv"
        path = os.path.join(curves_dir, filename)
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["# measurement_id", fit.get("measurement_id", name)])
            writer.writerow(["# source_file", fit.get("path", "")])
            writer.writerow(["# angle_deg", angle])
            writer.writerow(["# direction", fit.get("direction", "")])
            writer.writerow(["# temperature_K", fit.get("temperature_K", "")])
            writer.writerow(["# frequency_GHz", fit.get("frequency_GHz", "")])
            writer.writerow(["B_T", "signal_measured", "signal_fitted", "residual"])
            writer.writerows(zip(B, measured, fitted, residual))
        saved.append(path)
    return {"summary": summary, "curves": saved, "directory": out_dir}


# =====================================================
# B0 vs Angle export
# =====================================================

def export_B0_table(
    fit_results,
    out_dir,
    filename="B0_vs_angle.csv"
):
    """
    Simplified export
    angle, peak, B0
    """

    safe_makedirs(out_dir)

    path = os.path.join(
        out_dir,
        filename
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "angle_deg",
            "peak",
            "B0_T"
        ])

        for r in fit_results:

            writer.writerow([
                r["angle"],
                r["peak"],
                r["B0"]
            ])

    return path


# =====================================================
# g factor export
# =====================================================

def export_g_table(
    fit_results,
    out_dir,
    filename="g_vs_angle.csv"
):
    """
    Simplified export
    angle, peak, g
    """

    safe_makedirs(out_dir)

    path = os.path.join(
        out_dir,
        filename
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "angle_deg",
            "peak",
            "g"
        ])

        for r in fit_results:

            writer.writerow([
                r["angle"],
                r["peak"],
                r["g"]
            ])

    return path


# =====================================================
# Waterfall export
# =====================================================

def export_waterfall_data(
    angle_curves,
    out_dir,
    filename="waterfall_data.csv"
):
    """
    Export every angle curve
    in long format.

    angle,B,y
    """

    safe_makedirs(out_dir)

    path = os.path.join(
        out_dir,
        filename
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "angle_deg",
            "B_T",
            "signal"
        ])

        for c in angle_curves:

            angle = c["angle"]

            for Bv, yv in zip(
                c["B"],
                c["y"]
            ):

                writer.writerow([
                    angle,
                    Bv,
                    yv
                ])

    return path


# =====================================================
# Heatmap matrix export
# =====================================================

def export_heatmap_matrix(
    angles,
    B,
    Z,
    out_dir,
    filename="heatmap_matrix.npz"
):
    """
    Save heatmap as numpy archive.

    Reload:

        data=np.load(...)
        angles=data["angles"]
        B=data["B"]
        Z=data["Z"]
    """

    safe_makedirs(out_dir)

    path = os.path.join(
        out_dir,
        filename
    )

    np.savez(
        path,
        angles=angles,
        B=B,
        Z=Z
    )

    return path

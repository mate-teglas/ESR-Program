import numpy as np
import matplotlib.pyplot as plt

# Fixed colors for the four positional domains. The same domain always keeps
# the same color even when another domain is absent at a particular angle.
_DOMAIN_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]

def _domain_color(domain_id):
    i = int(domain_id) - 1
    return _DOMAIN_COLORS[i] if 0 <= i < 4 else "0.5"

from processing import (
    compute_stack,
    auto_offset_step,
    mean_angle,
    split_up_down,
    ensure_monotonic_unique,
)


def _attach_domain_toggles(fig, ax, domain_lines):
    """Add four interactive checkboxes to domain-analysis plots.

    ``domain_lines`` maps domain IDs 1..4 to the Matplotlib line objects that
    belong to that domain (possibly one line per sweep direction).
    """
    from matplotlib.widgets import CheckButtons
    fig.subplots_adjust(right=0.80)
    labels = [f"Domain {i}" for i in range(1, 5)]
    checks = CheckButtons(fig.add_axes([0.82, 0.62, 0.15, 0.22]), labels, [True]*4)

    def _toggle(label):
        did = int(label.split()[-1])
        lines = domain_lines.get(did, [])
        visible = not all(line.get_visible() for line in lines) if lines else False
        for line in lines:
            line.set_visible(visible)
        # Hide legend entries automatically by rebuilding from visible lines.
        handles, labs = ax.get_legend_handles_labels()
        pairs = [(h, l) for h, l in zip(handles, labs) if h.get_visible()]
        if pairs:
            ax.legend([q[0] for q in pairs], [q[1] for q in pairs], loc="best", fontsize=8)
        elif ax.get_legend() is not None:
            ax.get_legend().remove()
        fig.canvas.draw_idle()

    checks.on_clicked(_toggle)
    fig._esr_domain_checks = checks
    return checks
from fit_models import (
    multi_peak_derivative,
    multi_peak_mixed,
    extract_peak_table,
)


def plot_shifted(curves, freq_GHz=162.0, title_prefix="ESR", x_label_pos=1.05, y_shift_frac=0.15, offset_step=0.0):
    if not curves:
        return None

    if offset_step <= 0:
        offset_step = auto_offset_step(curves)

    y_shift = y_shift_frac * offset_step
    ang = mean_angle(curves)

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    for i, c in enumerate(curves, start=1):
        off = (i - 1) * offset_step
        color = "red" if c["direction"] == "up" else "c"

        ax.plot(c["B"], c["y1"] + off, color=color, lw=1.2)

        numtxt = f"#{c['num']}" if c.get("num") is not None else c["name"]

        try:
            yy = float(np.interp(x_label_pos, c["B"], c["y1"] + off)) + y_shift
            ax.text(x_label_pos, yy, f"{i:02d} | {numtxt}", fontsize=8, va="center", ha="left")
        except Exception:
            pass

    ax.set_xlabel("Magnetic field B (T)")
    ax.set_ylabel("Signal CH1 (centered + shifted)")
    ax.set_title(
        f"{title_prefix} | SHIFTED | f={freq_GHz:.3f} GHz | angle≈{ang:.2f}°\n"
        f"Selected {len(curves)} curve(s)",
        fontsize=12,
    )
    ax.grid(alpha=0.2)
    return fig


def plot_up_down_overlay(curves, mode="sum", freq_GHz=162.0, title_prefix="ESR"):
    up, down = split_up_down(curves)

    if not up or not down:
        raise ValueError("Need both UP and DOWN curves for PlotOver.")

    Bref_up, y_up, _ = compute_stack(up, mode=mode)
    Bref_down, y_down, _ = compute_stack(down, mode=mode)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    ax.plot(Bref_up, y_up, lw=2.0, label=f"{mode.upper()} UP (n={len(up)})")
    ax.plot(Bref_down, y_down, lw=2.0, label=f"{mode.upper()} DOWN (n={len(down)})")

    ax.set_xlabel("Magnetic field B (T)")
    ax.set_ylabel("Signal CH1 (centered)")
    ax.set_title(
        f"{title_prefix} | PlotOver {mode.upper()} | f={freq_GHz:.3f} GHz\n"
        f"UP={len(up)}, DOWN={len(down)}",
        fontsize=12,
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    return fig


def plot_waterfall(angle_curves, freq_GHz=162.0, title_prefix="ESR", scale=1.0):
    """
    Waterfall where the y-axis is the actual angle.

    Each ESR trace is plotted as:
        y_plot = angle + scale * normalized_signal

    This means when you zoom in, the angle values remain meaningful
    on the y-axis.
    """

    if not angle_curves:
        return None

    angle_curves = sorted(angle_curves, key=lambda c: c["angle"])

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    for c in angle_curves:
        angle = float(c["angle"])
        B = c["B"]
        y = c["y"]

        # Use the left/start baseline as zero, so each trace starts at its angle
        nbase = max(5, int(0.05 * len(y)))  # first 5% of points
        baseline = np.nanmedian(y[:nbase])

        y0 = y - baseline

        amp = np.nanmax(np.abs(y0))
        if amp > 0:
            y0 = y0 / amp

        y_plot = angle + scale * y0

        ax.plot(B, y_plot, lw=1.1)

    ax.set_xlabel("Magnetic field B (T)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title(
        f"{title_prefix} | Waterfall | f={freq_GHz:.3f} GHz\n"
        f"{len(angle_curves)} angle trace(s)",
        fontsize=12,
    )

    ax.grid(alpha=0.2)

    return fig


def plot_heatmap(angle_curves, freq_GHz=162.0, title_prefix="ESR"):
    """
    Builds B-angle intensity map from angle averaged/summed curves.
    """
    if not angle_curves:
        return None

    angle_curves = sorted(angle_curves, key=lambda c: c["angle"])

    # Common reference grid: densest B grid
    best = max(angle_curves, key=lambda c: c["B"].size)
    Bref, _ = ensure_monotonic_unique(best["B"], np.zeros_like(best["B"]))

    angles = []
    rows = []

    for c in angle_curves:
        Bc, yc = ensure_monotonic_unique(c["B"], c["y"])
        yi = np.interp(Bref, Bc, yc)

        angles.append(c["angle"])
        rows.append(yi)

    Z = np.vstack(rows)
    angles = np.asarray(angles)

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[Bref.min(), Bref.max(), angles.min(), angles.max()],
    )

    fig.colorbar(im, ax=ax, label="Signal CH1")
    ax.set_xlabel("Magnetic field B (T)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title(
        f"{title_prefix} | Angle-field heatmap | f={freq_GHz:.3f} GHz\n"
        f"{len(angle_curves)} angle trace(s)",
        fontsize=12,
    )

    return fig


def plot_fit(B, y, popt, model="derivative", title="ESR fit"):
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    if model == "mixed":
        yfit = multi_peak_mixed(B, *popt)
    else:
        yfit = multi_peak_derivative(B, *popt)

    ax.plot(B, y, lw=1.2, label="Data")
    ax.plot(B, yfit, lw=2.0, label="Fit")

    peaks = extract_peak_table(popt, model=model)

    for p in peaks:
        ax.axvline(p["B0"], ls="--", lw=1.0, alpha=0.6)
        ax.text(p["B0"], np.max(y), f"B0={p['B0']:.4g} T", rotation=90, fontsize=8, va="top")

    ax.set_xlabel("Magnetic field B (T)")
    ax.set_ylabel("Signal CH1")
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    return fig


def plot_B0_vs_angle(fit_results, title_prefix="ESR"):
    if not fit_results:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=False)
    directions = sorted(set(r.get("direction", "unknown") for r in fit_results))
    domain_lines = {i: [] for i in range(1, 5)}
    for direction in directions:
        for pid in range(1, 5):
            rows = sorted([r for r in fit_results if int(r.get("domain_id", r.get("peak", 0))) == pid and r.get("direction", "unknown") == direction], key=lambda r: r["angle"])
            if rows:
                line, = ax.plot([r["angle"] for r in rows], [r["B0"] for r in rows], marker="o", lw=1.5, label=f"{direction.upper()} domain {pid}", color=_domain_color(pid))
                domain_lines[pid].append(line)
    ax.set_xlabel("Angle (deg)"); ax.set_ylabel("B0 (T)")
    ax.set_title(f"{title_prefix} | B0 vs angle", fontsize=12)
    ax.grid(alpha=0.2); ax.legend(loc="best", fontsize=8)
    _attach_domain_toggles(fig, ax, domain_lines)
    return fig

def plot_g_vs_angle(fit_results, title_prefix="ESR"):
    if not fit_results:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=False)
    directions = sorted(set(r.get("direction", "unknown") for r in fit_results))
    domain_lines = {i: [] for i in range(1, 5)}
    for direction in directions:
        for pid in range(1, 5):
            rows = sorted([r for r in fit_results if int(r.get("domain_id", r.get("peak", 0))) == pid and r.get("direction", "unknown") == direction], key=lambda r: r["angle"])
            if rows:
                line, = ax.plot([r["angle"] for r in rows], [r["g"] for r in rows], marker="o", lw=1.5, label=f"{direction.upper()} domain {pid}", color=_domain_color(pid))
                domain_lines[pid].append(line)
    ax.set_xlabel("Angle (deg)"); ax.set_ylabel("g factor")
    ax.set_title(f"{title_prefix} | g vs angle", fontsize=12)
    ax.grid(alpha=0.2); ax.legend(loc="best", fontsize=8)
    _attach_domain_toggles(fig, ax, domain_lines)
    return fig

def _group_fit_rows(fit_results):
    directions = sorted(set(r.get("direction", "unknown") for r in fit_results))
    # Always expose exactly four positional domains in evaluation plots.
    return directions, range(1, 5)


def plot_linewidth_vs_angle(fit_results, title_prefix="ESR"):
    if not fit_results:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=False)
    directions, domains = _group_fit_rows(fit_results)
    domain_lines = {i: [] for i in range(1, 5)}
    for direction in directions:
        for did in domains:
            rows = sorted([r for r in fit_results if r.get("direction", "unknown") == direction and int(r.get("domain_id", r.get("peak", 0))) == did], key=lambda r: r["angle"])
            if rows:
                line, = ax.plot([r["angle"] for r in rows], [r["gamma"] for r in rows], marker="o", lw=1.4, label=f"{direction.upper()} domain {did}", color=_domain_color(did))
                domain_lines[did].append(line)
    ax.set_xlabel("Angle (deg)"); ax.set_ylabel("Linewidth gamma (T)")
    ax.set_title(f"{title_prefix} | Linewidth vs angle"); ax.grid(alpha=0.2); ax.legend(loc="best", fontsize=8)
    _attach_domain_toggles(fig, ax, domain_lines)
    return fig

def plot_intensity_vs_angle(fit_results, normalized=False, title_prefix="ESR"):
    if not fit_results:
        return None
    key = "intensity_fraction" if normalized else "intensity"
    ylabel = "Intensity fraction" if normalized else "Integrated absorptive intensity"
    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=False)
    directions, domains = _group_fit_rows(fit_results)
    domain_lines = {i: [] for i in range(1, 5)}
    for direction in directions:
        for did in domains:
            rows = sorted([r for r in fit_results if r.get("direction", "unknown") == direction and int(r.get("domain_id", r.get("peak", 0))) == did and key in r], key=lambda r: r["angle"])
            if rows:
                line, = ax.plot([r["angle"] for r in rows], [r[key] for r in rows], marker="o", lw=1.4, label=f"{direction.upper()} domain {did}", color=_domain_color(did))
                domain_lines[did].append(line)
    ax.set_xlabel("Angle (deg)"); ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} | {ylabel} vs angle"); ax.grid(alpha=0.2); ax.legend(loc="best", fontsize=8)
    _attach_domain_toggles(fig, ax, domain_lines)
    return fig

def plot_fit_heatmaps(series_fits, mode="fitted", title_prefix="ESR"):
    good = [s for s in series_fits if s.get("success") and str(s.get("review_status", "")).lower() != "excluded"]
    if not good:
        return None
    good = sorted(good, key=lambda s: float(s["angle"]))
    best = max(good, key=lambda s: len(s["B"]))
    Bref = np.asarray(best["B"], float)
    rows = []
    angles = []
    from fit_models import evaluate_fit_curve
    for s in good:
        B = np.asarray(s["B"], float)
        y = np.asarray(s["y"], float)
        yf = evaluate_fit_curve(B, s["popt"], s["details"]["model"])
        if mode == "fitted":
            z = yf
            label = "Fitted signal"
        elif mode == "residual":
            z = y - yf
            label = "Residual"
        else:
            z = y
            label = "Measured signal"
        rows.append(np.interp(Bref, B, z))
        angles.append(float(s["angle"]))
    Z = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    im = ax.imshow(Z, aspect="auto", origin="lower", extent=[Bref.min(), Bref.max(), min(angles), max(angles)])
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("Magnetic field B (T)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title(f"{title_prefix} | {label} heatmap")
    return fig


def plot_batch_fit_inspection(series_fits, max_panels=12, title_prefix="ESR",
                              on_refit=None, on_status_change=None,
                              on_set_reference=None,
                              default_min_components=2,
                              default_max_components=4, default_weak_sigma=4.0):
    """Interactive fit browser with manual correction support.

    Normal replacement calls::
        on_refit(spectrum, min_components, max_components, weak_sigma)

    Manual-center replacement calls::
        on_refit(spectrum, n, n, weak_sigma, manual_centers=[...])

    In manual mode, left-click the visible resonance centers in the upper plot.
    Right-click removes the nearest selected center. The selected centers are
    used as the initial physical resonances, and their count is forced.
    """
    good = [s for s in series_fits if s.get("success")]
    if not good:
        return None
    good.sort(key=lambda s: (str(s.get("direction", "")),
                             float(s.get("angle", np.nan)),
                             str(s.get("name", s.get("filename", "")))))

    from matplotlib.widgets import Button, TextBox, CheckButtons
    from fit_models import evaluate_fit_curve, mixed_derivative_peak, derivative_lorentzian

    fig = plt.figure(figsize=(15.8, 9.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0],
                          left=0.065, right=0.77, bottom=0.25, top=0.92,
                          hspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax)
    state = dict(index=0, show_fit=True, show_components=True,
                 show_baseline=False, show_residual=True,
                 show_previous=True, show_next=True, status="",
                 pick_mode=False, manual_centers=[])
    component_colors = [_domain_color(i) for i in range(1, 5)]
    info_text = fig.text(0.795, 0.88, "", ha="left", va="top",
                         family="monospace", fontsize=9.0)
    status_text = fig.text(0.795, 0.12, "", ha="left", va="top",
                           fontsize=9.0, weight="bold")
    browser_widgets = {}

    def _name_of(s):
        return str(s.get("name", s.get("filename", "unknown")))

    def _rows_and_curves(B, s):
        model = s["details"]["model"]
        rows = sorted(s.get("peaks", []), key=lambda r: int(r.get("domain_id", 99)))
        curves = []
        for row in rows:
            if model == "mixed":
                yc = mixed_derivative_peak(B, row["A"], row["theta"],
                                           row["B0"], row["gamma"])
            else:
                yc = derivative_lorentzian(B, row["A"], row["B0"], row["gamma"])
            curves.append((row, yc))
        return rows, curves

    def _update(_=None):
        i = int(np.clip(state["index"], 0, len(good)-1))
        state["index"] = i
        s = good[i]
        B = np.asarray(s["B"], float); y = np.asarray(s["y"], float)
        model = s["details"]["model"]
        yf = evaluate_fit_curve(B, s["popt"], model)
        residual = y - yf
        rows, components = _rows_and_curves(B, s)
        component_sum = np.sum([q[1] for q in components], axis=0) if components else np.zeros_like(B)
        baseline = yf - component_sum
        ax.clear(); ax_res.clear()
        # Neighbor spectra make branch motion visually obvious while reviewing.
        if state["show_previous"] and i > 0:
            sp = good[i-1]
            ax.plot(np.asarray(sp["B"], float), np.asarray(sp["y"], float),
                    lw=1.0, ls="--", alpha=.42, color="0.35",
                    label=f"Previous {float(sp.get('angle', np.nan)):.3g}°", zorder=1)
            for prow in sp.get("peaks", []):
                pdid = int(prow.get("domain_id", prow.get("peak", 1)))
                pc = component_colors[min(max(pdid-1, 0), 3)]
                ax.axvline(float(prow["B0"]), lw=.65, ls="--", alpha=.28, color=pc, zorder=1)
                ax.text(float(prow["B0"]), .985, f"D{pdid}−", transform=ax.get_xaxis_transform(),
                        color=pc, alpha=.65, ha="center", va="top", fontsize=7)
        if state["show_next"] and i + 1 < len(good):
            sn = good[i+1]
            ax.plot(np.asarray(sn["B"], float), np.asarray(sn["y"], float),
                    lw=1.0, ls=":", alpha=.48, color="tab:purple",
                    label=f"Next {float(sn.get('angle', np.nan)):.3g}°", zorder=1)
            for nrow in sn.get("peaks", []):
                ndid = int(nrow.get("domain_id", nrow.get("peak", 1)))
                nc = component_colors[min(max(ndid-1, 0), 3)]
                ax.axvline(float(nrow["B0"]), lw=.65, ls=":", alpha=.30, color=nc, zorder=1)
                ax.text(float(nrow["B0"]), .925, f"D{ndid}+", transform=ax.get_xaxis_transform(),
                        color=nc, alpha=.70, ha="center", va="top", fontsize=7)
        ax.plot(B, y, lw=1.0, label="Data", zorder=4)
        if state["show_fit"]:
            ax.plot(B, yf, lw=2.0, label="Total fit", zorder=5)
        if state["show_components"]:
            for row, yc in components:
                did = int(row.get("domain_id", row.get("peak", 1)))
                color = component_colors[min(max(did-1, 0), 3)]
                component_curve = baseline + yc
                ax.plot(B, component_curve, lw=1.25, ls="--", color=color,
                        label=f"Domain {did}", zorder=3)
                ax.axvline(row["B0"], lw=.8, ls=":", alpha=.65, color=color)
                y_center = float(np.interp(float(row["B0"]), B, component_curve))
                lock_tag = "🔒" if bool(row.get("domain_locked", False)) else ""
                ax.annotate(f"D{did}{lock_tag}", xy=(float(row["B0"]), y_center),
                            xytext=(0, 10), textcoords="offset points",
                            ha="center", va="bottom", color=color, fontsize=9,
                            weight="bold", zorder=10,
                            bbox=dict(boxstyle="round,pad=.16", fc="white", ec=color, alpha=.78))
        if state["show_baseline"]:
            ax.plot(B, baseline, lw=1.0, ls="-.", label="Baseline")
        # Manual center markers are deliberately visually distinct.
        for j, c in enumerate(sorted(state["manual_centers"]), 1):
            ax.axvline(c, color="magenta", lw=1.8, ls="--", alpha=.9, zorder=8)
            ax.text(c, ax.get_ylim()[1] if ax.get_ylim() else np.nanmax(y),
                    f" M{j}", color="magenta", rotation=90, va="top", ha="left",
                    fontsize=8, zorder=9)
        if state["show_residual"]:
            ax_res.plot(B, residual, lw=.85)
            ax_res.axhline(0, lw=.8, alpha=.5)
        direction = str(s.get("direction", "")).upper()
        angle = float(s.get("angle", np.nan))
        details = s.get("details", {})
        quality = str(details.get("quality", "unknown"))
        ncomp = int(details.get("n_components", len(rows)))
        mode_tag = " | PICK CENTERS" if state["pick_mode"] else ""
        ax.set_title(f"{title_prefix} fit browser | {i+1}/{len(good)} | {direction} | "
                     f"{angle:.3f}° | {_name_of(s)} | n={ncomp} | {quality}{mode_tag}")
        ax.set_ylabel("Signal CH1"); ax.grid(alpha=.18)
        ax.legend(loc="best", fontsize=8, ncol=2); ax.tick_params(labelbottom=False)
        ax_res.set_xlabel("Magnetic field B (T)"); ax_res.set_ylabel("Residual")
        ax_res.grid(alpha=.18)
        rmse = float(details.get("rmse", np.sqrt(np.mean(residual**2))))
        lines = [f"Spectrum {i+1} / {len(good)}", f"File      {_name_of(s)}",
                 f"Direction {direction}", f"Angle     {angle:.5g}°",
                 f"Model     {model}", f"Components {ncomp}",
                 f"Quality   {quality}",
                 f"Review    {s.get('review_status', 'unreviewed')}",
                 f"Tracking  {s.get('tracking_status', 'unreviewed')}",
                 f"Reference {'YES' if s.get('tracking_reference', False) else 'no'}",
                 f"RMSE      {rmse:.4g}", ""]
        reasons = s.get("review_reasons", [])
        if reasons:
            lines += ["Review reasons:"] + [f"  - {reason}" for reason in reasons] + [""]
        for row in rows:
            did = int(row.get("domain_id", row.get("peak", 0)))
            lines += [f"Domain {did}{' [LOCKED]' if row.get('domain_locked', False) else ''}",
                      f"  B0    {row['B0']:.6f} T",
                      f"  gamma {row['gamma']:.6f} T",
                      f"  frac  {float(row.get('intensity_fraction', np.nan)):.4f}"]
        if state["manual_centers"]:
            lines += ["", "Manual centers:"] + [f"  M{k}: {c:.6f} T" for k,c in enumerate(sorted(state["manual_centers"]),1)]
        # Reference-ID boxes correspond to fitted resonances sorted left-to-right.
        if "reference_boxes" in browser_widgets:
            left_rows = sorted(rows, key=lambda r: float(r["B0"]))
            for j, box in enumerate(browser_widgets["reference_boxes"]):
                value = int(left_rows[j].get("domain_id", j+1)) if j < len(left_rows) else j+1
                if box.text != str(value):
                    box.set_val(str(value))
        info_text.set_text("\n".join(lines))
        status_text.set_text(state.get("status", ""))
        fig.canvas.draw_idle()

    def _clear_picks(_e=None):
        state["manual_centers"] = []
        state["status"] = "Manual centers cleared."
        _update()

    def _safe_export_name(s):
        direction = str(s.get("direction", "unknown")).upper()
        angle = float(s.get("angle", np.nan))
        name = _name_of(s)
        stem = name.rsplit(".", 1)[0] if "." in name else name
        stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
        return f"{direction}_angle_{angle:08.3f}_{stem}_fit_review"

    def _make_export_figure(s):
        """Create a clean publication/review figure without browser controls."""
        B = np.asarray(s["B"], float)
        y = np.asarray(s["y"], float)
        model = s["details"]["model"]
        yf = evaluate_fit_curve(B, s["popt"], model)
        residual = y - yf
        rows, components = _rows_and_curves(B, s)
        component_sum = np.sum([q[1] for q in components], axis=0) if components else np.zeros_like(B)
        baseline = yf - component_sum

        out = plt.figure(figsize=(12.5, 7.8))
        gs2 = out.add_gridspec(2, 2, width_ratios=[4.8, 1.35], height_ratios=[3.2, 1.0],
                              left=.075, right=.97, bottom=.09, top=.92,
                              hspace=.08, wspace=.08)
        a = out.add_subplot(gs2[0,0])
        ar = out.add_subplot(gs2[1,0], sharex=a)
        at = out.add_subplot(gs2[:,1]); at.axis("off")
        a.plot(B, y, lw=1.0, color="0.20", label="Data")
        a.plot(B, yf, lw=2.0, color="tab:orange", label="Total fit")
        for row, yc in components:
            did = int(row.get("domain_id", row.get("peak", 1)))
            color = component_colors[min(max(did-1, 0), 3)]
            curve = baseline + yc
            a.plot(B, curve, lw=1.2, ls="--", color=color, label=f"Domain {did}")
            b0=float(row["B0"]); y0=float(np.interp(b0, B, curve))
            a.annotate(f"D{did}", xy=(b0,y0), xytext=(0,10), textcoords="offset points",
                       ha="center", va="bottom", color=color, weight="bold", fontsize=9,
                       bbox=dict(boxstyle="round,pad=.16", fc="white", ec=color, alpha=.9))
        if state.get("show_baseline", False):
            a.plot(B, baseline, lw=1.0, ls="-.", color="0.45", label="Baseline")
        ar.plot(B, residual, lw=.85, color="tab:blue")
        ar.axhline(0, lw=.8, color="0.35", alpha=.5)
        direction=str(s.get("direction", "")).upper(); angle=float(s.get("angle", np.nan))
        quality=str(s.get("details",{}).get("quality","unknown"))
        a.set_title(f"{title_prefix} fit | {direction} | {angle:.3f}° | {quality}")
        a.set_ylabel("Signal CH1"); a.grid(alpha=.16); a.tick_params(labelbottom=False)
        ar.set_xlabel("Magnetic field B (T)"); ar.set_ylabel("Residual"); ar.grid(alpha=.16)
        h,l=a.get_legend_handles_labels(); seen=set(); hh=[]; ll=[]
        for x1,x2 in zip(h,l):
            if x2 not in seen: seen.add(x2); hh.append(x1); ll.append(x2)
        a.legend(hh,ll,loc="best",fontsize=8,ncol=2)
        rmse=float(s.get("details",{}).get("rmse", np.sqrt(np.mean(residual**2))))
        lines=[f"File: {_name_of(s)}", f"Direction: {direction}", f"Angle: {angle:.5g}°",
               f"Fit quality: {quality}", f"Review: {s.get('review_status','unreviewed')}",
               f"Tracking: {s.get('tracking_status','unreviewed')}", f"RMSE: {rmse:.5g}", "",
               "ID      B0 (T)    gamma (T)   fraction   lock"]
        for row in sorted(rows,key=lambda r:int(r.get("domain_id",99))):
            did=int(row.get("domain_id",row.get("peak",0)))
            lines.append(f"D{did:<2}   {float(row['B0']):8.5f}   {float(row['gamma']):8.5f}   "
                         f"{float(row.get('intensity_fraction',row.get('fraction',np.nan))):8.4f}   "
                         f"{'yes' if row.get('domain_locked',False) else 'no'}")
        at.text(0,1,"\n".join(lines),ha="left",va="top",family="monospace",fontsize=9)
        return out

    def _save_current(_=None):
        try:
            from tkinter import filedialog
            s = good[state["index"]]
            initial = _safe_export_name(s) + ".png"
            path = filedialog.asksaveasfilename(title="Save clean fit figure", initialfile=initial,
                                                defaultextension=".png",
                                                filetypes=[("PNG image","*.png"),("PDF document","*.pdf"),
                                                           ("SVG vector","*.svg")])
            if not path: return
            out = _make_export_figure(s)
            out.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(out)
            state["status"] = f"Saved clean figure: {path}"
        except Exception as exc:
            state["status"] = f"Save failed: {exc}"
        _update()

    def _export_group(kind):
        try:
            from tkinter import filedialog
            folder = filedialog.askdirectory(title=f"Choose folder for {kind} fit figures")
            if not folder: return
            from pathlib import Path
            folder = Path(folder)
            if kind == "accepted":
                selected=[q for q in good if str(q.get("review_status","")).lower()=="accepted"]
            else:
                selected=[q for q in good if str(q.get("review_status","")).lower()=="needs_review" or
                          str(q.get("tracking_status","")).lower()=="needs_review" or
                          str(q.get("details",{}).get("quality","")).lower()=="needs_review"]
            if not selected:
                state["status"] = f"No {kind} spectra to export."; _update(); return
            for k,q in enumerate(selected,1):
                out=_make_export_figure(q)
                out.savefig(folder / (_safe_export_name(q)+".png"), dpi=300, bbox_inches="tight")
                plt.close(out)
            state["status"] = f"Exported {len(selected)} {kind} clean figures to {folder}."
        except Exception as exc:
            state["status"] = f"Batch export failed: {exc}"
        _update()

    def _set_index(j):
        state["index"] = int(np.clip(j, 0, len(good)-1))
        state["manual_centers"] = []
        state["pick_mode"] = False
        _update()

    def _move(delta): _set_index(state["index"] + delta)
    def _first(_e=None): _set_index(0)
    def _last(_e=None): _set_index(len(good)-1)
    def _jump(text):
        q = str(text).strip()
        if not q: return
        if q.lstrip("+-").isdigit():
            j = int(q)-1
            if 0 <= j < len(good): _set_index(j); return
        try:
            a = float(q); _set_index(int(np.argmin([abs(float(s.get("angle", np.nan))-a) for s in good]))); return
        except Exception: pass
        for j, s in enumerate(good):
            if q.lower() in _name_of(s).lower(): _set_index(j); return

    def _toggle(label):
        mp = {"Total fit":"show_fit", "Domains":"show_components",
              "Baseline":"show_baseline", "Residual":"show_residual",
              "Previous":"show_previous", "Next":"show_next"}
        state[mp[label]] = not state[mp[label]]; _update()

    def _refit(_event=None, use_manual=False):
        if on_refit is None:
            state["status"] = "Refit callback is not connected."; _update(); return
        try:
            sigma = float(sigma_box.text)
            manual = sorted(set(float(x) for x in state["manual_centers"])) if use_manual else []
            if manual:
                if not (1 <= len(manual) <= 4):
                    state["status"] = "Select between 1 and 4 centers."; _update(); return
                minc = maxc = len(manual)
            else:
                minc = int(float(min_box.text)); maxc = int(float(max_box.text))
                minc = max(1, min(4, minc)); maxc = max(minc, min(4, maxc))
            state["status"] = "Refitting current spectrum…"; _update()
            try:
                try:
                    center_tol = float(center_tol_box.text)
                except Exception:
                    center_tol = 0.08
                if center_tol <= 0:
                    raise ValueError("±ΔB must be positive.")
                ok = bool(on_refit(good[state["index"]], minc, maxc, sigma,
                                   manual_centers=manual or None,
                                   center_tolerance=center_tol))
            except TypeError:
                ok = bool(on_refit(good[state["index"]], minc, maxc, sigma))
            if ok:
                state["status"] = ("Manual-center fit replaced and domains re-tracked."
                                   if manual else "Fit replaced and domains re-tracked.")
                state["manual_centers"] = []
                state["pick_mode"] = False
            else:
                state["status"] = "Refit failed; old fit kept."
        except Exception as exc:
            state["status"] = f"Refit error: {exc}"
        _update()

    def _set_reference(_event=None):
        if on_set_reference is None:
            state["status"] = "Reference callback is not connected."; _update(); return
        s = good[state["index"]]
        rows = sorted(s.get("peaks", []), key=lambda r: float(r["B0"]))
        if len(rows) != 4:
            state["status"] = "Reference spectrum must contain exactly four fitted domains."; _update(); return
        try:
            ids = [int(float(box.text)) for box in browser_widgets["reference_boxes"]]
            if sorted(ids) != [1,2,3,4]:
                raise ValueError("Use each domain number 1,2,3,4 exactly once.")
            ok = bool(on_set_reference(s, ids))
            state["status"] = ("Tracking reference saved; domains re-tracked forward and backward."
                               if ok else "Reference assignment failed.")
        except Exception as exc:
            state["status"] = f"Reference error: {exc}"
        _update()

    def _set_review_status(value):
        s = good[state["index"]]
        s["review_status"] = value
        if on_status_change is not None:
            try:
                on_status_change(s, value)
            except Exception as exc:
                state["status"] = f"Status update error: {exc}"
                _update(); return
        state["status"] = f"Marked {value}."
        _update()

    def _review_indices():
        out = []
        for j, q in enumerate(good):
            status = str(q.get("review_status", "")).lower()
            quality = str(q.get("details", {}).get("quality", "")).lower()
            if status == "needs_review" or (status in ("", "unreviewed") and quality == "needs_review"):
                out.append(j)
        return out

    def _jump_review(step):
        inds = _review_indices()
        if not inds:
            state["status"] = "No spectra are marked needs_review."; _update(); return
        cur = state["index"]
        if step > 0:
            candidates = [j for j in inds if j > cur]
            _set_index(candidates[0] if candidates else inds[0])
        else:
            candidates = [j for j in inds if j < cur]
            _set_index(candidates[-1] if candidates else inds[-1])

    def _toggle_pick(_event=None):
        state["pick_mode"] = not state["pick_mode"]
        state["status"] = ("Pick mode ON: left-click centers; right-click removes nearest."
                           if state["pick_mode"] else "Pick mode OFF.")
        _update()

    def _toggle_all_locks(_=None):
        if on_assign_domain is None:
            state["status"] = "Domain-edit callback is not connected."; _update(); return
        s = good[state["index"]]
        rows = list(s.get("peaks", []))
        if not rows:
            state["status"] = "No fitted domains on this spectrum."; _update(); return
        target_lock = not all(bool(r.get("domain_locked", False)) for r in rows)
        ok_all = True
        for row in rows:
            did=int(row.get("domain_id",row.get("peak",0)))
            try:
                ok_all = bool(on_assign_domain(s, float(row["B0"]), did, target_lock, "current")) and ok_all
            except Exception:
                ok_all = False
        state["status"] = (("Locked" if target_lock else "Unlocked") +
                           f" all {len(rows)} domains on current spectrum." if ok_all else
                           "Could not update all domain locks.")
        _update()

    def _on_click(event):
        if not state["pick_mode"] or event.inaxes is not ax or event.xdata is None:
            return
        B = np.asarray(good[state["index"]]["B"], float)
        x = float(np.clip(event.xdata, np.min(B), np.max(B)))
        if event.button == 1:
            if len(state["manual_centers"]) >= 4:
                state["status"] = "Maximum of four centers reached."
            else:
                # Snap to the nearest sampled field point for reproducibility.
                x = float(B[int(np.argmin(np.abs(B-x)))])
                if all(abs(x-c) > max(np.median(np.diff(np.sort(B)))*3, 1e-4)
                       for c in state["manual_centers"]):
                    state["manual_centers"].append(x)
                    state["status"] = f"Added center at {x:.6f} T."
        elif event.button == 3 and state["manual_centers"]:
            j = int(np.argmin(np.abs(np.asarray(state["manual_centers"])-x)))
            removed = state["manual_centers"].pop(j)
            state["status"] = f"Removed center at {removed:.6f} T."
        _update()

    def _on_key(event):
        key=(event.key or "").lower()
        if key in ("right","down","n"): _move(1)
        elif key in ("left","up","p"): _move(-1)
        elif key in ("shift+right","pagedown"): _move(10)
        elif key in ("shift+left","pageup"): _move(-10)
        elif key=="home": _first()
        elif key=="end": _last()
        elif key=="r": _refit()
        elif key=="a": _set_review_status("accepted")
        elif key=="q": _set_review_status("needs_review")
        elif key=="x": _set_review_status("excluded")
        elif key=="]": _jump_review(1)
        elif key=="[": _jump_review(-1)
        elif key=="m": _toggle_pick()
        elif key=="t": _set_reference()
        elif key=="enter" and state["manual_centers"]: _refit(use_manual=True)
        elif key in ("delete","backspace"): _clear_picks()
    def _on_scroll(event): _move(-1 if event.button=="up" else 1)

    # Choose a four-line spectrum as the tracking reference. The four boxes
    # name the fitted resonances from left to right and may be any permutation
    # of 1,2,3,4.
    fig.text(.065,.225,"Tracking reference — IDs left → right:",ha="left",va="center",weight="bold")
    reference_boxes=[]
    for j, x in enumerate((.265,.302,.339,.376), start=1):
        box=TextBox(fig.add_axes([x,.207,.032,.035]),"",initial=str(j))
        reference_boxes.append(box)
    set_reference_btn=Button(fig.add_axes([.415,.207,.145,.035]),"Set reference (T)")
    set_reference_btn.on_clicked(_set_reference)
    fig.text(.57,.225,"Requires exactly 4 fitted lines",ha="left",va="center",fontsize=8.5)
    browser_widgets["reference_boxes"] = reference_boxes
    browser_widgets["set_reference_btn"] = set_reference_btn

    y0,h=.105,.040
    controls=[]
    for x,w,label,cb in [(0.065,.042,"|<",_first),(0.112,.048,"-10",lambda e:_move(-10)),
                         (0.165,.042,"<",lambda e:_move(-1)),(0.212,.042,">",lambda e:_move(1)),
                         (0.259,.048,"+10",lambda e:_move(10)),(0.312,.042,">|",_last)]:
        b=Button(fig.add_axes([x,y0,w,h]),label); b.on_clicked(cb); controls.append(b)
    fig.text(.385,y0+.020,"Jump:",ha="right",va="center")
    jump_box=TextBox(fig.add_axes([.392,y0,.09,h]),"",initial="1"); jump_box.on_submit(_jump)
    checks=CheckButtons(fig.add_axes([.50,.065,.15,.125]),
                        ["Total fit","Domains","Baseline","Residual","Previous","Next"],
                        [True,True,False,True,True,True])
    checks.on_clicked(_toggle)

    prev_review_btn=Button(fig.add_axes([.665,.145,.055,.035]),"< Review")
    prev_review_btn.on_clicked(lambda e:_jump_review(-1))
    next_review_btn=Button(fig.add_axes([.722,.145,.055,.035]),"Review >")
    next_review_btn.on_clicked(lambda e:_jump_review(1))
    accept_btn=Button(fig.add_axes([.665,.105,.052,.035]),"Accept")
    accept_btn.on_clicked(lambda e:_set_review_status("accepted"))
    review_btn=Button(fig.add_axes([.720,.105,.057,.035]),"Review")
    review_btn.on_clicked(lambda e:_set_review_status("needs_review"))
    exclude_btn=Button(fig.add_axes([.665,.065,.112,.035]),"Exclude spectrum")
    exclude_btn.on_clicked(lambda e:_set_review_status("excluded"))

    fig.text(.065,.065,"Automatic replacement:",ha="left",va="center",weight="bold")
    fig.text(.185,.065,"min",ha="right",va="center")
    min_box=TextBox(fig.add_axes([.19,.047,.038,.035]),"",initial=str(default_min_components))
    fig.text(.255,.065,"max",ha="right",va="center")
    max_box=TextBox(fig.add_axes([.26,.047,.038,.035]),"",initial=str(default_max_components))
    fig.text(.325,.065,"σ",ha="right",va="center")
    sigma_box=TextBox(fig.add_axes([.33,.047,.045,.035]),"",initial=str(default_weak_sigma))
    refit_btn=Button(fig.add_axes([.39,.047,.12,.035]),"Replace auto fit"); refit_btn.on_clicked(_refit)

    pick_btn=Button(fig.add_axes([.065,.012,.11,.035]),"Pick centers (M)"); pick_btn.on_clicked(_toggle_pick)
    clear_btn=Button(fig.add_axes([.18,.012,.09,.035]),"Clear picks"); clear_btn.on_clicked(_clear_picks)
    fig.text(.278,.029,"±ΔB",ha="left",va="center",fontsize=8.5)
    center_tol_box=TextBox(fig.add_axes([.315,.012,.05,.035]),"",initial="0.08")
    manual_btn=Button(fig.add_axes([.375,.012,.14,.035]),"Refit from picks"); manual_btn.on_clicked(lambda e:_refit(use_manual=True))
    fig.text(.52,.029,"Centers stay within ±ΔB; Enter refits",ha="left",va="center",fontsize=8.5)

    fig.text(.795,.31,"Navigation\n←/→ previous / next\nShift+←/→ jump 10\nHome / End first / last\nMouse wheel previous / next\n\nManual correction\nM toggles pick mode\nLeft-click resonance centers\nRight-click removes nearest\nEnter refits from picks\nDelete clears picks",ha="left",va="top",fontsize=8.5)
    fig.canvas.mpl_connect("key_press_event",_on_key)
    fig.canvas.mpl_connect("scroll_event",_on_scroll)
    fig.canvas.mpl_connect("button_press_event",_on_click)
    _update()
    fig._esr_fit_browser=dict(buttons=controls,refit_btn=refit_btn,pick_btn=pick_btn,
                              accept_btn=accept_btn,review_btn=review_btn,exclude_btn=exclude_btn,
                              prev_review_btn=prev_review_btn,next_review_btn=next_review_btn,
                              clear_btn=clear_btn,manual_btn=manual_btn,jump_box=jump_box,
                              center_tol_box=center_tol_box,
                              reference_boxes=reference_boxes,set_reference_btn=set_reference_btn,
                              checks=checks,min_box=min_box,max_box=max_box,sigma_box=sigma_box,
                              state=state,update=_update)
    return fig

def plot_avoided_crossing_diagnostic(fit_results, title_prefix="ESR"):
    """Simple diagnostic: plot nearest-domain field separation versus angle."""
    if not fit_results:
        return None
    by_key = {}
    for r in fit_results:
        by_key.setdefault((r.get("direction", "unknown"), r["angle"]), []).append(r)
    data = []
    for (direction, angle), rows in by_key.items():
        vals = sorted(float(r["B0"]) for r in rows)
        if len(vals) >= 2:
            gaps = np.diff(vals)
            data.append((direction, float(angle), float(np.min(gaps))))
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for direction in sorted(set(d[0] for d in data)):
        q = sorted([d for d in data if d[0] == direction], key=lambda x: x[1])
        ax.plot([x[1] for x in q], [x[2] for x in q], marker="o", label=direction.upper())
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Minimum adjacent resonance separation (T)")
    ax.set_title(f"{title_prefix} | Crossing / avoided-crossing diagnostic")
    ax.grid(alpha=.2)
    ax.legend()
    return fig

# ================================================================
# Clean review/edit browser (overrides the earlier implementation)
# ================================================================
def plot_batch_fit_inspection(series_fits, max_panels=12, title_prefix="ESR",
                              on_refit=None, on_status_change=None,
                              on_set_reference=None, on_assign_domain=None,
                              default_min_components=2,
                              default_max_components=4, default_weak_sigma=4.0):
    """Interactive fit review and correction browser.

    Review mode is intentionally uncluttered. Edit mode exposes automatic and
    manual refitting, reference selection, and direct domain reassignment.

    Domain reassignment workflow:
      * press E to enter Edit mode;
      * click a fitted resonance or its D1..D4 label;
      * press 1, 2, 3, or 4 to assign the selected component;
      * press L to lock/unlock it;
      * choose whether the correction applies only here, forward, backward,
        or to the full direction series.

    The individual component curves remain available through the Components
    checkbox and are visible by default.
    """
    good = [s for s in series_fits if s.get("success")]
    if not good:
        return None
    good.sort(key=lambda s: (str(s.get("direction", "")),
                             float(s.get("angle", np.nan)),
                             str(s.get("name", s.get("filename", "")))))

    from matplotlib.widgets import Button, TextBox, CheckButtons, RadioButtons
    from fit_models import evaluate_fit_curve, mixed_derivative_peak, derivative_lorentzian

    fig = plt.figure(figsize=(16.4, 9.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.0],
                          left=0.055, right=0.76, bottom=0.19, top=0.94,
                          hspace=0.07)
    ax = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax)

    state = dict(index=0, edit_mode=False, show_fit=True,
                 show_components=True, show_labels=True,
                 show_baseline=False, show_residual=True,
                 show_previous=True, show_next=True,
                 pick_mode=False, manual_centers=[],
                 selected_B0=None, selected_domain=None,
                 status="Review mode. Press E to edit this spectrum.",
                 apply_scope="current", lock_assignment=True)

    component_colors = [_domain_color(i) for i in range(1, 5)]
    info_text = fig.text(0.785, 0.91, "", ha="left", va="top",
                         family="monospace", fontsize=9.2)
    help_text = fig.text(0.785, 0.29, "", ha="left", va="top", fontsize=8.6)
    status_text = fig.text(0.055, 0.165, "", ha="left", va="center",
                           fontsize=9.2, weight="bold")

    browser_widgets = {}
    edit_axes = []
    edit_widgets = []

    def _name_of(s):
        return str(s.get("name", s.get("filename", "unknown")))

    def _rows_and_curves(B, s):
        model = s["details"]["model"]
        rows = sorted(s.get("peaks", []), key=lambda r: int(r.get("domain_id", 99)))
        curves = []
        for row in rows:
            if model == "mixed":
                yc = mixed_derivative_peak(B, row["A"], row["theta"],
                                           row["B0"], row["gamma"])
            else:
                yc = derivative_lorentzian(B, row["A"], row["B0"], row["gamma"])
            curves.append((row, yc))
        return rows, curves

    def _current_row(s=None):
        if s is None:
            s = good[state["index"]]
        rows = list(s.get("peaks", []))
        if not rows or state["selected_B0"] is None:
            return None
        return min(rows, key=lambda r: abs(float(r["B0"]) - float(state["selected_B0"])))

    def _refresh_edit_visibility():
        visible = bool(state["edit_mode"])
        for a in edit_axes:
            a.set_visible(visible)
        # Some Matplotlib widget artists can remain visible even when their
        # parent axes is toggled on older macOS backends. Hide every artist
        # explicitly as well, especially RadioButtons circles and labels.
        for widget in edit_widgets:
            try:
                widget.ax.set_visible(visible)
                for artist in widget.ax.get_children():
                    artist.set_visible(visible)
            except Exception:
                pass
        fig.canvas.draw_idle()

    def _update(_=None):
        i = int(np.clip(state["index"], 0, len(good)-1))
        state["index"] = i
        s = good[i]
        B = np.asarray(s["B"], float)
        y = np.asarray(s["y"], float)
        model = s["details"]["model"]
        yf = evaluate_fit_curve(B, s["popt"], model)
        residual = y - yf
        rows, components = _rows_and_curves(B, s)
        component_sum = np.sum([q[1] for q in components], axis=0) if components else np.zeros_like(B)
        baseline = yf - component_sum

        ax.clear(); ax_res.clear()

        # Neighbor spectra are context only: muted curves plus domain positions.
        if state["show_previous"] and i > 0:
            sp = good[i-1]
            ax.plot(np.asarray(sp["B"], float), np.asarray(sp["y"], float),
                    lw=.9, ls="--", alpha=.30, color="0.42",
                    label=f"Previous {float(sp.get('angle', np.nan)):.3g}°", zorder=1)
            for row in sp.get("peaks", []):
                did = int(row.get("domain_id", row.get("peak", 1)))
                c = component_colors[min(max(did-1, 0), 3)]
                ax.axvline(float(row["B0"]), lw=.7, ls="--", alpha=.25, color=c, zorder=1)
        if state["show_next"] and i + 1 < len(good):
            sn = good[i+1]
            ax.plot(np.asarray(sn["B"], float), np.asarray(sn["y"], float),
                    lw=.9, ls=":", alpha=.32, color="0.30",
                    label=f"Next {float(sn.get('angle', np.nan)):.3g}°", zorder=1)
            for row in sn.get("peaks", []):
                did = int(row.get("domain_id", row.get("peak", 1)))
                c = component_colors[min(max(did-1, 0), 3)]
                ax.axvline(float(row["B0"]), lw=.7, ls=":", alpha=.25, color=c, zorder=1)

        ax.plot(B, y, lw=1.0, label="Data", color="0.22", zorder=4)
        if state["show_fit"]:
            ax.plot(B, yf, lw=2.1, label="Total fit", color="tab:orange", zorder=5)

        for row, yc in components:
            did = int(row.get("domain_id", row.get("peak", 1)))
            color = component_colors[min(max(did-1, 0), 3)]
            curve = baseline + yc
            if state["show_components"]:
                ax.plot(B, curve, lw=1.25, ls="--", color=color,
                        label=f"Domain {did}", alpha=.90, zorder=3)
            if state["show_labels"]:
                b0 = float(row["B0"])
                y0 = float(np.interp(b0, B, curve if state["show_components"] else yf))
                selected = (state["selected_B0"] is not None and
                            abs(b0 - float(state["selected_B0"])) <= 1e-8)
                lock_tag = " L" if bool(row.get("domain_locked", False)) else ""
                ax.annotate(f"D{did}{lock_tag}", xy=(b0, y0), xytext=(0, 11),
                            textcoords="offset points", ha="center", va="bottom",
                            color=color, fontsize=9.5, weight="bold", zorder=10,
                            bbox=dict(boxstyle="round,pad=.19", fc="white", ec=color,
                                      lw=2.2 if selected else 1.0,
                                      alpha=.95 if selected else .82))
                ax.axvline(b0, color=color, lw=.75, ls=":", alpha=.48, zorder=2)

        if state["show_baseline"]:
            ax.plot(B, baseline, lw=1.1, ls="-.", label="Baseline", color="0.45")

        for j, c0 in enumerate(sorted(state["manual_centers"]), 1):
            ax.axvline(c0, color="magenta", lw=1.7, ls="--", alpha=.9, zorder=8)
            ax.text(c0, .98, f"M{j}", transform=ax.get_xaxis_transform(),
                    color="magenta", ha="center", va="top", fontsize=8, weight="bold")

        if state["show_residual"]:
            ax_res.plot(B, residual, lw=.85, color="tab:blue")
            ax_res.axhline(0, lw=.8, alpha=.5, color="0.35")
        else:
            ax_res.text(.5, .5, "Residual hidden", transform=ax_res.transAxes,
                        ha="center", va="center", color="0.5")

        direction = str(s.get("direction", "")).upper()
        angle = float(s.get("angle", np.nan))
        details = s.get("details", {})
        quality = str(details.get("quality", "unknown"))
        ncomp = int(details.get("n_components", len(rows)))
        mode = "EDIT" if state["edit_mode"] else "REVIEW"
        pick = " | PICK CENTERS" if state["pick_mode"] else ""
        ax.set_title(f"{title_prefix} fit review | {i+1}/{len(good)} | {direction} | "
                     f"{angle:.3f}° | n={ncomp} | {quality} | {mode}{pick}")
        ax.set_ylabel("Signal CH1"); ax.grid(alpha=.16)
        ax.tick_params(labelbottom=False)
        handles, labels = ax.get_legend_handles_labels()
        # De-duplicate domain entries and keep the legend compact.
        seen = set(); h2=[]; l2=[]
        for h, lab in zip(handles, labels):
            if lab not in seen:
                seen.add(lab); h2.append(h); l2.append(lab)
        ax.legend(h2, l2, loc="best", fontsize=8, ncol=2)
        ax_res.set_xlabel("Magnetic field B (T)"); ax_res.set_ylabel("Residual")
        ax_res.grid(alpha=.16)

        rmse = float(details.get("rmse", np.sqrt(np.mean(residual**2))))
        table = [
            f"Spectrum {i+1}/{len(good)}   {_name_of(s)}",
            f"{direction}   angle {angle:.5g}°",
            f"Fit: {quality}   review: {s.get('review_status', 'unreviewed')}",
            f"Tracking: {s.get('tracking_status', 'unreviewed')}",
            f"Reference: {'YES' if s.get('tracking_reference', False) else 'no'}",
            f"RMSE: {rmse:.4g}", "",
            "ID   B0 (T)    gamma (T)   fraction   lock",
            "--   ------    ---------   --------   ----",
        ]
        for row in sorted(rows, key=lambda r: int(r.get("domain_id", 99))):
            did = int(row.get("domain_id", row.get("peak", 0)))
            table.append(f"D{did}   {float(row['B0']):.5f}    {float(row['gamma']):.5f}     "
                         f"{float(row.get('intensity_fraction', np.nan)):.3f}      "
                         f"{'YES' if row.get('domain_locked', False) else 'no'}")
        reasons = list(s.get("review_reasons", [])) + list(s.get("tracking_reasons", []))
        if reasons:
            table += ["", "Warnings:"] + [f"• {r}" for r in reasons[:6]]
        selected = _current_row(s)
        if selected is not None:
            table += ["", f"Selected: D{int(selected.get('domain_id',0))} at {float(selected['B0']):.5f} T"]
        info_text.set_text("\n".join(table))

        if state["edit_mode"]:
            help_text.set_text(
                "EDIT MODE\n"
                "Click a fitted line/label, then press 1–4 to rename.\n"
                "L locks/unlocks the selected assignment.\n"
                "M enables manual center picking; Enter refits.\n"
                "T sets this four-line spectrum as the reference.\n"
                "E returns to clean review mode."
            )
        else:
            help_text.set_text(
                "REVIEW MODE\n"
                "←/→ previous/next   Shift+←/→ jump 10\n"
                "A accept   Q needs review   X exclude\n"
                "[ / ] previous/next review item\n"
                "E edit fit or domain identities\n"
                "Use Components to show/hide fitted domain curves."
            )
        status_text.set_text(state.get("status", ""))
        _refresh_edit_visibility()
        fig.canvas.draw_idle()

    def _safe_export_name(s):
        direction = str(s.get("direction", "unknown")).upper()
        angle = float(s.get("angle", np.nan))
        name = _name_of(s)
        stem = name.rsplit(".", 1)[0] if "." in name else name
        stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
        return f"{direction}_angle_{angle:08.3f}_{stem}_fit_review"

    def _make_export_figure(s):
        """Create a clean publication/review figure without browser controls."""
        B = np.asarray(s["B"], float)
        y = np.asarray(s["y"], float)
        model = s["details"]["model"]
        yf = evaluate_fit_curve(B, s["popt"], model)
        residual = y - yf
        rows, components = _rows_and_curves(B, s)
        component_sum = np.sum([q[1] for q in components], axis=0) if components else np.zeros_like(B)
        baseline = yf - component_sum

        out = plt.figure(figsize=(12.5, 7.8))
        gs2 = out.add_gridspec(2, 2, width_ratios=[4.8, 1.35], height_ratios=[3.2, 1.0],
                              left=.075, right=.97, bottom=.09, top=.92,
                              hspace=.08, wspace=.08)
        a = out.add_subplot(gs2[0,0])
        ar = out.add_subplot(gs2[1,0], sharex=a)
        at = out.add_subplot(gs2[:,1]); at.axis("off")
        a.plot(B, y, lw=1.0, color="0.20", label="Data")
        a.plot(B, yf, lw=2.0, color="tab:orange", label="Total fit")
        for row, yc in components:
            did = int(row.get("domain_id", row.get("peak", 1)))
            color = component_colors[min(max(did-1, 0), 3)]
            curve = baseline + yc
            a.plot(B, curve, lw=1.2, ls="--", color=color, label=f"Domain {did}")
            b0=float(row["B0"]); y0=float(np.interp(b0, B, curve))
            a.annotate(f"D{did}", xy=(b0,y0), xytext=(0,10), textcoords="offset points",
                       ha="center", va="bottom", color=color, weight="bold", fontsize=9,
                       bbox=dict(boxstyle="round,pad=.16", fc="white", ec=color, alpha=.9))
        if state.get("show_baseline", False):
            a.plot(B, baseline, lw=1.0, ls="-.", color="0.45", label="Baseline")
        ar.plot(B, residual, lw=.85, color="tab:blue")
        ar.axhline(0, lw=.8, color="0.35", alpha=.5)
        direction=str(s.get("direction", "")).upper(); angle=float(s.get("angle", np.nan))
        quality=str(s.get("details",{}).get("quality","unknown"))
        a.set_title(f"{title_prefix} fit | {direction} | {angle:.3f}° | {quality}")
        a.set_ylabel("Signal CH1"); a.grid(alpha=.16); a.tick_params(labelbottom=False)
        ar.set_xlabel("Magnetic field B (T)"); ar.set_ylabel("Residual"); ar.grid(alpha=.16)
        h,l=a.get_legend_handles_labels(); seen=set(); hh=[]; ll=[]
        for x1,x2 in zip(h,l):
            if x2 not in seen: seen.add(x2); hh.append(x1); ll.append(x2)
        a.legend(hh,ll,loc="best",fontsize=8,ncol=2)
        rmse=float(s.get("details",{}).get("rmse", np.sqrt(np.mean(residual**2))))
        lines=[f"File: {_name_of(s)}", f"Direction: {direction}", f"Angle: {angle:.5g}°",
               f"Fit quality: {quality}", f"Review: {s.get('review_status','unreviewed')}",
               f"Tracking: {s.get('tracking_status','unreviewed')}", f"RMSE: {rmse:.5g}", "",
               "ID      B0 (T)    gamma (T)   fraction   lock"]
        for row in sorted(rows,key=lambda r:int(r.get("domain_id",99))):
            did=int(row.get("domain_id",row.get("peak",0)))
            lines.append(f"D{did:<2}   {float(row['B0']):8.5f}   {float(row['gamma']):8.5f}   "
                         f"{float(row.get('intensity_fraction',row.get('fraction',np.nan))):8.4f}   "
                         f"{'yes' if row.get('domain_locked',False) else 'no'}")
        at.text(0,1,"\n".join(lines),ha="left",va="top",family="monospace",fontsize=9)
        return out

    def _save_current(_=None):
        try:
            from tkinter import filedialog
            s = good[state["index"]]
            initial = _safe_export_name(s) + ".png"
            path = filedialog.asksaveasfilename(title="Save clean fit figure", initialfile=initial,
                                                defaultextension=".png",
                                                filetypes=[("PNG image","*.png"),("PDF document","*.pdf"),
                                                           ("SVG vector","*.svg")])
            if not path: return
            out = _make_export_figure(s)
            out.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(out)
            state["status"] = f"Saved clean figure: {path}"
        except Exception as exc:
            state["status"] = f"Save failed: {exc}"
        _update()

    def _export_group(kind):
        try:
            from tkinter import filedialog
            folder = filedialog.askdirectory(title=f"Choose folder for {kind} fit figures")
            if not folder: return
            from pathlib import Path
            folder = Path(folder)
            if kind == "accepted":
                selected=[q for q in good if str(q.get("review_status","")).lower()=="accepted"]
            else:
                selected=[q for q in good if str(q.get("review_status","")).lower()=="needs_review" or
                          str(q.get("tracking_status","")).lower()=="needs_review" or
                          str(q.get("details",{}).get("quality","")).lower()=="needs_review"]
            if not selected:
                state["status"] = f"No {kind} spectra to export."; _update(); return
            for k,q in enumerate(selected,1):
                out=_make_export_figure(q)
                out.savefig(folder / (_safe_export_name(q)+".png"), dpi=300, bbox_inches="tight")
                plt.close(out)
            state["status"] = f"Exported {len(selected)} {kind} clean figures to {folder}."
        except Exception as exc:
            state["status"] = f"Batch export failed: {exc}"
        _update()

    def _set_index(j):
        state["index"] = int(np.clip(j, 0, len(good)-1))
        state["manual_centers"] = []
        state["pick_mode"] = False
        state["selected_B0"] = None
        state["selected_domain"] = None
        _update()

    def _move(delta): _set_index(state["index"] + delta)
    def _first(_=None): _set_index(0)
    def _last(_=None): _set_index(len(good)-1)

    def _jump(text):
        q = str(text).strip()
        if not q: return
        if q.lstrip("+-").isdigit():
            j = int(q)-1
            if 0 <= j < len(good): _set_index(j); return
        try:
            a = float(q)
            _set_index(int(np.argmin([abs(float(s.get("angle", np.nan))-a) for s in good])))
            return
        except Exception:
            pass
        for j, s in enumerate(good):
            if q.lower() in _name_of(s).lower():
                _set_index(j); return
        state["status"] = f"No spectrum matched '{q}'."; _update()

    def _toggle(label):
        mp = {"Total fit":"show_fit", "Components":"show_components",
              "Domain labels":"show_labels", "Baseline":"show_baseline",
              "Residual":"show_residual", "Previous":"show_previous", "Next":"show_next"}
        state[mp[label]] = not state[mp[label]]
        _update()

    def _toggle_edit(_=None):
        state["edit_mode"] = not state["edit_mode"]
        state["pick_mode"] = False
        state["status"] = ("Edit mode: click a domain label/center and press 1–4."
                           if state["edit_mode"] else
                           "Review mode: use A/Q/X and arrows; press E to edit.")
        _update()

    def _clear_picks(_=None):
        state["manual_centers"] = []
        state["status"] = "Manual centers cleared."
        _update()

    def _toggle_pick(_=None):
        if not state["edit_mode"]:
            state["edit_mode"] = True
        state["pick_mode"] = not state["pick_mode"]
        state["status"] = ("Pick mode ON: left-click centers; right-click removes nearest."
                           if state["pick_mode"] else "Pick mode OFF.")
        _update()

    def _refit(_event=None, use_manual=False):
        if on_refit is None:
            state["status"] = "Refit callback is not connected."; _update(); return
        try:
            sigma = float(sigma_box.text)
            manual = sorted(set(float(x) for x in state["manual_centers"])) if use_manual else []
            if manual:
                minc = maxc = len(manual)
            else:
                minc = max(1, min(4, int(float(min_box.text))))
                maxc = max(minc, min(4, int(float(max_box.text))))
            center_tol = float(center_tol_box.text)
            if center_tol <= 0:
                raise ValueError("±ΔB must be positive.")
            state["status"] = "Refitting current spectrum…"; _update()
            ok = bool(on_refit(good[state["index"]], minc, maxc, sigma,
                               manual_centers=manual or None,
                               center_tolerance=center_tol))
            if ok:
                state["status"] = "Fit replaced; domain tracking recalculated."
                state["manual_centers"] = []
                state["pick_mode"] = False
                state["selected_B0"] = None
            else:
                state["status"] = "Refit failed; old fit kept."
        except Exception as exc:
            state["status"] = f"Refit error: {exc}"
        _update()

    def _set_reference(_=None):
        if on_set_reference is None:
            state["status"] = "Reference callback is not connected."; _update(); return
        s = good[state["index"]]
        rows = sorted(s.get("peaks", []), key=lambda r: float(r["B0"]))
        if not 1 <= len(rows) <= 4:
            state["status"] = "Reference requires between one and four fitted lines."; _update(); return
        ids = [int(r.get("domain_id", 0)) for r in rows]
        if len(set(ids)) != len(ids) or any(x not in (1,2,3,4) for x in ids):
            state["status"] = "Rename components to unique domain IDs from D1–D4 first."; _update(); return
        ok = bool(on_set_reference(s, ids))
        state["status"] = ("Reference set and tracking recalculated." if ok else "Reference failed.")
        _update()

    def _set_review_status(value):
        s = good[state["index"]]
        s["review_status"] = value
        if on_status_change is not None:
            on_status_change(s, value)
        state["status"] = f"Marked {value}."
        _update()

    def _review_indices():
        out=[]
        for j, q in enumerate(good):
            status=str(q.get("review_status", "")).lower()
            quality=str(q.get("details", {}).get("quality", "")).lower()
            tracking=str(q.get("tracking_status", "")).lower()
            if status == "needs_review" or tracking == "needs_review" or (
                    status in ("", "unreviewed") and quality == "needs_review"):
                out.append(j)
        return out

    def _jump_review(step):
        inds=_review_indices()
        if not inds:
            state["status"]="No spectra are marked needs_review."; _update(); return
        cur=state["index"]
        if step>0:
            q=[j for j in inds if j>cur]; _set_index(q[0] if q else inds[0])
        else:
            q=[j for j in inds if j<cur]; _set_index(q[-1] if q else inds[-1])

    def _assign_domain(new_id=None, toggle_lock=False):
        if on_assign_domain is None:
            state["status"] = "Domain-edit callback is not connected."; _update(); return
        s = good[state["index"]]
        row = _current_row(s)
        if row is None:
            state["status"] = "Click a fitted domain first."; _update(); return
        did = int(row.get("domain_id", row.get("peak", 0)))
        target = did if new_id is None else int(new_id)
        lock_value = (not bool(row.get("domain_locked", False))) if toggle_lock else bool(state["lock_assignment"])
        try:
            ok = bool(on_assign_domain(s, float(row["B0"]), target,
                                       lock_value, state["apply_scope"]))
            if ok:
                state["selected_B0"] = float(row["B0"])
                state["status"] = (f"Assigned selected line to D{target}; "
                                   f"scope={state['apply_scope']}; lock={'on' if lock_value else 'off'}.")
            else:
                state["status"] = "Domain reassignment failed."
        except Exception as exc:
            state["status"] = f"Domain assignment error: {exc}"
        _update()

    def _toggle_all_locks(_=None):
        """Lock or unlock every fitted domain in the current spectrum."""
        if on_assign_domain is None:
            state["status"] = "Domain-edit callback is not connected."
            _update()
            return

        s = good[state["index"]]
        rows = list(s.get("peaks", []))
        if not rows:
            state["status"] = "No fitted domains on this spectrum."
            _update()
            return

        # If every component is locked, unlock all. Otherwise lock all.
        target_lock = not all(bool(r.get("domain_locked", False)) for r in rows)
        ok_all = True
        for row in rows:
            did = int(row.get("domain_id", row.get("peak", 0)))
            try:
                ok = bool(on_assign_domain(
                    s, float(row["B0"]), did, target_lock, "current"
                ))
                ok_all = ok_all and ok
            except Exception:
                ok_all = False

        if ok_all:
            action = "Locked" if target_lock else "Unlocked"
            state["status"] = f"{action} all {len(rows)} domains on current spectrum."
        else:
            state["status"] = "Could not update all domain locks."
        _update()

    def _on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        B = np.asarray(good[state["index"]]["B"], float)
        x = float(np.clip(event.xdata, np.min(B), np.max(B)))
        if state["edit_mode"] and state["pick_mode"]:
            if event.button == 1:
                if len(state["manual_centers"]) < 4:
                    x = float(B[int(np.argmin(np.abs(B-x)))])
                    if all(abs(x-c) > max(np.median(np.diff(np.sort(B)))*3, 1e-4)
                           for c in state["manual_centers"]):
                        state["manual_centers"].append(x)
                        state["status"] = f"Added center at {x:.6f} T."
                else:
                    state["status"] = "Maximum of four centers reached."
            elif event.button == 3 and state["manual_centers"]:
                j=int(np.argmin(np.abs(np.asarray(state["manual_centers"])-x)))
                removed=state["manual_centers"].pop(j)
                state["status"] = f"Removed center at {removed:.6f} T."
            _update(); return

        if state["edit_mode"] and event.button == 1:
            rows = good[state["index"]].get("peaks", [])
            if not rows:
                return
            row = min(rows, key=lambda r: abs(float(r["B0"])-x))
            span=max(float(np.ptp(B)), .1)
            if abs(float(row["B0"])-x) <= max(.06, .025*span):
                state["selected_B0"] = float(row["B0"])
                state["selected_domain"] = int(row.get("domain_id", 0))
                state["status"] = (f"Selected D{state['selected_domain']} at "
                                   f"{state['selected_B0']:.5f} T. Press 1–4 to rename.")
                _update()

    def _on_key(event):
        key=(event.key or "").lower()
        if key in ("right","down","n"): _move(1)
        elif key in ("left","up","p"): _move(-1)
        elif key in ("shift+right","pagedown"): _move(10)
        elif key in ("shift+left","pageup"): _move(-10)
        elif key=="home": _first()
        elif key=="end": _last()
        elif key=="e": _toggle_edit()
        elif key=="r" and state["edit_mode"]: _refit()
        elif key=="a": _set_review_status("accepted")
        elif key=="q": _set_review_status("needs_review")
        elif key=="x": _set_review_status("excluded")
        elif key=="]": _jump_review(1)
        elif key=="[": _jump_review(-1)
        elif key=="m": _toggle_pick()
        elif key=="t" and state["edit_mode"]: _set_reference()
        elif key=="l" and state["edit_mode"]: _assign_domain(toggle_lock=True)
        elif key in ("1","2","3","4") and state["edit_mode"]: _assign_domain(int(key))
        elif key=="enter" and state["manual_centers"]: _refit(use_manual=True)
        elif key in ("delete","backspace"): _clear_picks()

    def _on_scroll(event): _move(-1 if event.button=="up" else 1)

    # Always-visible compact review controls.
    nav=[]
    y=.105; h=.038
    for x,w,label,cb in [(0.055,.040,"|<",_first),(0.099,.045,"-10",lambda e:_move(-10)),
                         (0.148,.040,"<",lambda e:_move(-1)),(0.192,.040,">",lambda e:_move(1)),
                         (0.236,.045,"+10",lambda e:_move(10)),(0.285,.040,">|",_last)]:
        b=Button(fig.add_axes([x,y,w,h]),label); b.on_clicked(cb); nav.append(b)
    fig.text(.355,y+.019,"Jump",ha="right",va="center",fontsize=8.5)
    jump_box=TextBox(fig.add_axes([.362,y,.085,h]),"",initial="1"); jump_box.on_submit(_jump)

    checks=CheckButtons(fig.add_axes([.46,.055,.145,.105]),
                        ["Total fit","Components","Domain labels","Baseline","Residual","Previous","Next"],
                        [True,True,True,False,True,True,True])
    checks.on_clicked(_toggle)

    edit_btn=Button(fig.add_axes([.615,.105,.068,.038]),"Edit (E)"); edit_btn.on_clicked(_toggle_edit)
    prev_review=Button(fig.add_axes([.69,.105,.055,.038]),"< Review"); prev_review.on_clicked(lambda e:_jump_review(-1))
    next_review=Button(fig.add_axes([.748,.105,.055,.038]),"Review >"); next_review.on_clicked(lambda e:_jump_review(1))
    accept=Button(fig.add_axes([.615,.058,.055,.038]),"Accept"); accept.on_clicked(lambda e:_set_review_status("accepted"))
    review=Button(fig.add_axes([.673,.058,.057,.038]),"Review"); review.on_clicked(lambda e:_set_review_status("needs_review"))
    exclude=Button(fig.add_axes([.733,.058,.070,.038]),"Exclude"); exclude.on_clicked(lambda e:_set_review_status("excluded"))

    # Clean export controls. These create figures without browser widgets.
    save_current_btn=Button(fig.add_axes([.815,.105,.075,.038]),"Save image")
    save_current_btn.on_clicked(_save_current)
    export_accepted_btn=Button(fig.add_axes([.815,.058,.075,.038]),"Accepted PNGs")
    export_accepted_btn.on_clicked(lambda e:_export_group("accepted"))
    export_review_btn=Button(fig.add_axes([.895,.058,.075,.038]),"Review PNGs")
    export_review_btn.on_clicked(lambda e:_export_group("review"))

    # Edit-only controls: hidden in review mode.
    fig.text(.055,.034,"Fit:",ha="left",va="center",fontsize=8.5,weight="bold")
    min_box=TextBox(fig.add_axes([.085,.016,.035,.034]),"min",initial=str(default_min_components))
    max_box=TextBox(fig.add_axes([.145,.016,.035,.034]),"max",initial=str(default_max_components))
    sigma_box=TextBox(fig.add_axes([.205,.016,.043,.034]),"σ",initial=str(default_weak_sigma))
    refit_btn=Button(fig.add_axes([.263,.016,.090,.034]),"Auto refit"); refit_btn.on_clicked(_refit)
    pick_btn=Button(fig.add_axes([.362,.016,.082,.034]),"Pick centers"); pick_btn.on_clicked(_toggle_pick)
    clear_btn=Button(fig.add_axes([.450,.016,.060,.034]),"Clear"); clear_btn.on_clicked(_clear_picks)
    center_tol_box=TextBox(fig.add_axes([.535,.016,.045,.034]),"±ΔB",initial="0.08")
    manual_btn=Button(fig.add_axes([.595,.016,.095,.034]),"Refit picks"); manual_btn.on_clicked(lambda e:_refit(use_manual=True))
    reference_btn=Button(fig.add_axes([.700,.016,.103,.034]),"Set reference"); reference_btn.on_clicked(_set_reference)

    # Domain edit scope and lock behavior live in a compact right-side panel.
    scope_ax=fig.add_axes([.785,.43,.16,.13])
    scope_radio=RadioButtons(scope_ax,["current","after →","← before","everything"],active=0)
    scope_ax.set_title("Rename + track scope",fontsize=9)
    def _scope(label):
        state["apply_scope"] = {
            "after →": "after",
            "← before": "before",
        }.get(label, label)
    scope_radio.on_clicked(_scope)
    lock_ax=fig.add_axes([.785,.365,.13,.05])
    lock_check=CheckButtons(lock_ax,["Lock assignment"],[True])
    def _lock(_label): state["lock_assignment"]=not state["lock_assignment"]
    lock_check.on_clicked(_lock)
    lock_all_btn=Button(fig.add_axes([.785,.315,.13,.038]),"Lock/unlock all")
    lock_all_btn.on_clicked(_toggle_all_locks)

    edit_axes.extend([min_box.ax,max_box.ax,sigma_box.ax,refit_btn.ax,pick_btn.ax,
                      clear_btn.ax,center_tol_box.ax,manual_btn.ax,reference_btn.ax,
                      scope_ax,lock_ax,lock_all_btn.ax])
    edit_widgets.extend([min_box,max_box,sigma_box,refit_btn,pick_btn,clear_btn,
                         center_tol_box,manual_btn,reference_btn,scope_radio,
                         lock_check,lock_all_btn])
    # Text label for Fit is not an Axes and cannot be hidden; it is unobtrusive.

    fig.canvas.mpl_connect("key_press_event",_on_key)
    fig.canvas.mpl_connect("scroll_event",_on_scroll)
    fig.canvas.mpl_connect("button_press_event",_on_click)
    _refresh_edit_visibility()
    _update()

    fig._esr_fit_browser=dict(buttons=nav, checks=checks, edit_btn=edit_btn,
                              prev_review=prev_review,next_review=next_review,
                              accept=accept,review=review,exclude=exclude,
                              jump_box=jump_box,min_box=min_box,max_box=max_box,
                              sigma_box=sigma_box,center_tol_box=center_tol_box,
                              refit_btn=refit_btn,pick_btn=pick_btn,clear_btn=clear_btn,
                              manual_btn=manual_btn,reference_btn=reference_btn,
                              scope_radio=scope_radio,lock_check=lock_check,
                              lock_all_btn=lock_all_btn,save_current_btn=save_current_btn,
                              export_accepted_btn=export_accepted_btn,
                              export_review_btn=export_review_btn,
                              state=state,update=_update)
    return fig

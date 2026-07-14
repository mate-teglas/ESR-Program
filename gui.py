import os
import csv
import pickle
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt

from data_io import (
    load_folder,
    cluster_by_angle,
    take_last_n_by_number_then_name,
    metadata_value_key,
    split_files_by_metadata,
)

from processing import (
    baseline_center_cluster,
    baseline_center_individual,
)

from fit_models import (
    fit_curve,
    fit_all_angles,
    extract_peak_table,
    g_factor,
    multi_peak_derivative,
    multi_peak_mixed,
    rebuild_fit_rows,
    track_domain_branches,
)

from plotting import (
    plot_shifted,
    plot_waterfall,
    plot_heatmap,
    plot_fit,
    plot_B0_vs_angle,
    plot_g_vs_angle,
    plot_up_down_overlay,
    plot_linewidth_vs_angle,
    plot_intensity_vs_angle,
    plot_fit_heatmaps,
    plot_batch_fit_inspection,
    plot_avoided_crossing_diagnostic,
)

from exports import (
    export_fit_results_csv,
    export_fitted_measurements,
    export_dat_curve,
    export_figure,
)


DEFAULT_FREQ_GHZ = 162.0
DEFAULT_N_PER_ANGLE = 64
DEFAULT_ANGLE_TOL = 0.9


class ESRApp:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title("ESR Analysis Suite V5")

        self.root.geometry("1400x900")

        # ====================================
        # Data containers
        # ====================================

        self.folder = ""

        self.items = []
        self.filtered_items = []

        self.clusters = []

        self.curves = []

        self.cluster_index = 0

        self.fit_results = []
        self.series_fit_results = []
        self.fit_output_dir = ""

        # ====================================
        # Main notebook
        # ====================================

        self.notebook = ttk.Notebook(self.root)

        self.notebook.pack(
            fill="both",
            expand=True
        )

        self.tab_data = ttk.Frame(self.notebook)
        self.tab_processing = ttk.Frame(self.notebook)
        self.tab_fitting = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)

        self.notebook.add(
            self.tab_data,
            text="Data"
        )

        self.notebook.add(
            self.tab_processing,
            text="Processing"
        )

        self.notebook.add(
            self.tab_fitting,
            text="Fitting"
        )

        self.notebook.add(
            self.tab_analysis,
            text="Analysis"
        )

        self.build_data_tab()
        self.build_processing_tab()
        self.build_fitting_tab()
        self.build_analysis_tab()

    # ==================================================
    # DATA TAB
    # ==================================================

    def build_data_tab(self):

        frm = self.tab_data

        row = 0

        tk.Button(
            frm,
            text="Select Folder",
            command=self.select_folder
        ).grid(
            row=row,
            column=0,
            padx=5,
            pady=5,
            sticky="w"
        )

        self.lbl_folder = tk.Label(
            frm,
            text="No folder selected",
            anchor="w"
        )

        self.lbl_folder.grid(
            row=row,
            column=1,
            columnspan=5,
            sticky="we"
        )

        row += 1

        tk.Label(
            frm,
            text="Frequency (GHz)"
        ).grid(
            row=row,
            column=0,
            sticky="w"
        )

        self.ent_freq = tk.Entry(frm)

        self.ent_freq.insert(
            0,
            str(DEFAULT_FREQ_GHZ)
        )

        self.ent_freq.grid(
            row=row,
            column=1,
            sticky="w"
        )

        tk.Label(frm, text="Material name").grid(row=row, column=2, sticky="w")
        self.ent_material = tk.Entry(frm, width=18)
        self.ent_material.insert(0, "Sample")
        self.ent_material.grid(row=row, column=3, sticky="w")

        tk.Label(
            frm,
            text="Angle tolerance"
        ).grid(
            row=row,
            column=4,
            sticky="w"
        )

        self.ent_tol = tk.Entry(frm)

        self.ent_tol.insert(
            0,
            str(DEFAULT_ANGLE_TOL)
        )

        self.ent_tol.grid(
            row=row,
            column=5,
            sticky="w"
        )

        # Kept as an internal compatibility value only. Fitting never uses it
        # to combine measurements.
        self.ent_n = tk.Entry(frm)
        self.ent_n.insert(0, str(DEFAULT_N_PER_ANGLE))
        self.ent_n.grid_remove()

        tk.Label(
            frm,
            text="Prefix"
        ).grid(
            row=row,
            column=6,
            sticky="w"
        )

        self.ent_prefix = tk.Entry(frm)

        self.ent_prefix.insert(
            0,
            "k"
        )

        self.ent_prefix.grid(
            row=row,
            column=7,
            sticky="w"
        )

        tk.Label(
            frm,
            text="Extension"
        ).grid(
            row=row,
            column=8,
            sticky="w"
        )

        self.ent_ext = tk.Entry(frm)

        self.ent_ext.insert(
            0,
            ".dat"
        )

        self.ent_ext.grid(
            row=row,
            column=9,
            sticky="w"
        )

        row += 1

        tk.Label(frm, text="Temperature filter").grid(row=row, column=0, sticky="w")
        self.temperature_filter = tk.StringVar(value="All")
        self.temperature_combo = ttk.Combobox(
            frm, textvariable=self.temperature_filter, state="readonly", width=16,
            values=["All"],
        )
        self.temperature_combo.grid(row=row, column=1, sticky="w")

        tk.Label(frm, text="Frequency filter").grid(row=row, column=2, sticky="w")
        self.frequency_filter = tk.StringVar(value="All")
        self.frequency_combo = ttk.Combobox(
            frm, textvariable=self.frequency_filter, state="readonly", width=16,
            values=["All"],
        )
        self.frequency_combo.grid(row=row, column=3, sticky="w")

        tk.Button(frm, text="Apply filters", command=self.apply_metadata_filters).grid(
            row=row, column=4, padx=5
        )
        tk.Button(frm, text="Clear filters", command=self.clear_metadata_filters).grid(
            row=row, column=5, padx=5
        )
        self.lbl_filter_summary = tk.Label(frm, text="0 measurements", anchor="w")
        self.lbl_filter_summary.grid(row=row, column=6, columnspan=3, sticky="w")

        row += 1
        tk.Label(frm, text="Separate files by metadata").grid(row=row, column=0, sticky="w")
        self.split_mode = tk.StringVar(value="copy")
        ttk.Combobox(frm, textvariable=self.split_mode, state="readonly", width=8,
                     values=["copy", "move"]).grid(row=row, column=1, sticky="w")
        tk.Button(frm, text="Create separated folders", command=self.do_split_filtered_files).grid(
            row=row, column=2, columnspan=2, sticky="w", padx=5
        )
        tk.Button(frm, text="Choose fit output directory", command=self.select_fit_output_dir).grid(
            row=row, column=4, columnspan=2, sticky="w", padx=5
        )
        self.lbl_fit_output = tk.Label(frm, text="Default: <data>/FIT_out", anchor="w")
        self.lbl_fit_output.grid(row=row, column=6, columnspan=4, sticky="w")

        row += 1

        tk.Button(
            frm,
            text="Load Data",
            command=self.load_data
        ).grid(
            row=row,
            column=0,
            pady=5
        )

        tk.Label(
            frm,
            text="Angle Cluster"
        ).grid(
            row=row,
            column=1,
            sticky="w"
        )

        self.cluster_var = tk.StringVar()

        self.cluster_combo = ttk.Combobox(
            frm,
            textvariable=self.cluster_var,
            state="readonly",
            width=40
        )

        tk.Button(
            frm,
            text="Select/Exclude Angle Clusters",
            command=self.open_cluster_selector
        ).grid(
            row=row,
            column=5,
            padx=5,
            sticky="w"
        )

        self.cluster_combo.grid(
            row=row,
            column=2,
            columnspan=3,
            sticky="we"
        )

        self.cluster_combo.bind(
            "<<ComboboxSelected>>",
            self.on_cluster_change
        )

        # Selection buttons

        row += 1

        tk.Button(
            frm,
            text="Select ALL",
            command=self.select_all
        ).grid(row=row,column=0)

        tk.Button(
            frm,
            text="Select NONE",
            command=self.select_none
        ).grid(row=row,column=1)

        tk.Button(
            frm,
            text="Select UP",
            command=self.select_up
        ).grid(row=row,column=2)

        tk.Button(
            frm,
            text="Select DOWN",
            command=self.select_down
        ).grid(row=row,column=3)

        # Scrollable checkbox region

        row += 1

        self.checkbox_frame = tk.Frame(frm)

        self.checkbox_frame.grid(
            row=row,
            column=0,
            columnspan=6,
            sticky="nsew"
        )

        frm.grid_rowconfigure(
            row,
            weight=1
        )

        self.checkbox_canvas = tk.Canvas(
            self.checkbox_frame
        )

        self.checkbox_scroll = tk.Scrollbar(
            self.checkbox_frame,
            orient="vertical",
            command=self.checkbox_canvas.yview
        )

        self.checkbox_inner = tk.Frame(
            self.checkbox_canvas
        )

        self.checkbox_canvas.create_window(
            (0,0),
            window=self.checkbox_inner,
            anchor="nw"
        )

        self.checkbox_canvas.configure(
            yscrollcommand=self.checkbox_scroll.set
        )

        self.checkbox_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        self.checkbox_scroll.pack(
            side="right",
            fill="y"
        )

        self.check_vars = []

    def open_cluster_selector(self):

        if not self.clusters:
            messagebox.showwarning("No data", "Load data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Select angle clusters for Waterfall / Heatmap / Fit All")
        win.geometry("500x700")

        canvas = tk.Canvas(win)
        scrollbar = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas)

        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for i, cluster in enumerate(self.clusters):

            if i >= len(self.cluster_include_vars):
                self.cluster_include_vars.append(tk.BooleanVar(value=True))

            ang = float(np.mean([c["angle"] for c in cluster]))
            n = len(cluster)

            txt = f"{i + 1:02d} | angle≈{ang:.2f}° | N={n}"

            chk = tk.Checkbutton(
                inner,
                text=txt,
                variable=self.cluster_include_vars[i],
                anchor="w",
                justify="left"
            )

            chk.pack(fill="x", padx=8, pady=2)

        btns = tk.Frame(win)
        btns.pack(fill="x", pady=5)

        tk.Button(
            btns,
            text="Select all",
            command=lambda: [v.set(True) for v in self.cluster_include_vars]
        ).pack(side="left", padx=5)

        tk.Button(
            btns,
            text="Select none",
            command=lambda: [v.set(False) for v in self.cluster_include_vars]
        ).pack(side="left", padx=5)

        tk.Button(
            btns,
            text="Close",
            command=win.destroy
        ).pack(side="right", padx=5)



    # ==================================================
    # PROCESSING TAB
    # ==================================================

    def build_processing_tab(self):

        frm = self.tab_processing

        tk.Label(
            frm,
            text="Processing"
        ).pack(
            pady=20
        )

        self.btn_plot_shifted = tk.Button(
            frm,
            text="Plot Shifted",
            command=self.do_plot_shifted
        )

        self.btn_plot_shifted.pack(
            pady=5
        )

        self.btn_save_shifted = tk.Button(
            frm,
            text="Save Shifted",
            command=self.do_save_shifted
        )

        self.btn_save_shifted.pack(
            pady=5
        )

        self.btn_sum = tk.Button(
            frm,
            text="SUM Export",
            command=self.do_sum_export
        )

        self.btn_sum.pack(
            pady=5
        )

        tk.Label(
            frm,
            text="Fitting and analysis always preserve individual measurements; no averaging is performed.",
            fg="dark green",
        ).pack(pady=8)

    def do_plot_shifted(self):

        curves = self.get_selected_curves()

        if not curves:
            messagebox.showwarning(
                "No curves",
                "Select at least one curve."
            )

            return

        try:
            freq = float(
                self.ent_freq.get()
            )
        except Exception:
            freq = DEFAULT_FREQ_GHZ

        fig = plot_shifted(
            curves,
            freq_GHz=freq
        )

        if fig is not None:
            import matplotlib.pyplot as plt
            plt.show()

    def do_save_shifted(self):

        curves = self.get_selected_curves()

        if not curves:
            return

        try:
            freq = float(
                self.ent_freq.get()
            )
        except Exception:
            freq = DEFAULT_FREQ_GHZ

        fig = plot_shifted(
            curves,
            freq_GHz=freq
        )

        if fig is None:
            return

        angle = np.mean(
            [c["angle"] for c in curves]
        )

        path = export_figure(
            fig,
            self.get_fit_output_dir(),
            f"SHIFTED_f{freq:.1f}GHz_ang{angle:.2f}.png"
        )

        messagebox.showinfo(
            "Saved",
            path
        )

    def do_sum_export(self):

        self.export_stack(
            mode="sum"
        )

    def do_avg_export(self):

        self.export_stack(
            mode="avg"
        )

    def export_stack(
            self,
            mode="sum"
    ):

        from processing import compute_stack

        curves = self.get_selected_curves()

        if not curves:
            messagebox.showwarning(
                "No curves",
                "Select at least one curve."
            )

            return

        try:
            freq = float(
                self.ent_freq.get()
            )
        except Exception:
            freq = DEFAULT_FREQ_GHZ

        Bref, y1, y2 = compute_stack(
            curves,
            mode=mode
        )

        angle = np.mean(
            [c["angle"] for c in curves]
        )

        path = export_dat_curve(

            out_dir=self.get_fit_output_dir(),

            mode_tag=mode.upper(),

            freq_GHz=freq,

            mean_angle=angle,

            selected_files=[
                c["name"]
                for c in curves
            ],

            selected_indices=[],

            original_header=self.items[0]["header"],

            B=Bref,

            y1=y1,

            y2=y2,
        )

        messagebox.showinfo(
            "Export complete",
            path
        )





    # ==================================================
    # FITTING TAB
    # ==================================================

    def build_fitting_tab(self):

        frm = self.tab_fitting

        tk.Label(
            frm,
            text="Fit Model"
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        self.fit_model = tk.StringVar(
            value="mixed"
        )

        tk.Radiobutton(
            frm,
            text="Derivative",
            variable=self.fit_model,
            value="derivative"
        ).grid(
            row=1,
            column=0,
            sticky="w"
        )

        tk.Radiobutton(
            frm,
            text="Mixed Abs/Disp",
            variable=self.fit_model,
            value="mixed"
        ).grid(
            row=2,
            column=0,
            sticky="w"
        )

        # Fit UP and DOWN sweeps separately.  The selected direction is used
        # by "Fit All Angles" and "Auto Fit All + Plot".
        tk.Label(
            frm,
            text="Fit direction"
        ).grid(
            row=0,
            column=6,
            sticky="w",
            padx=(18, 0)
        )

        self.fit_direction = tk.StringVar(value="up")

        tk.Radiobutton(
            frm,
            text="Up",
            variable=self.fit_direction,
            value="up"
        ).grid(
            row=1,
            column=6,
            sticky="w",
            padx=(18, 0)
        )

        tk.Radiobutton(
            frm,
            text="Down",
            variable=self.fit_direction,
            value="down"
        ).grid(
            row=2,
            column=6,
            sticky="w",
            padx=(18, 0)
        )

        tk.Label(
            frm,
            text="Maximum Lorentzians"
        ).grid(
            row=0,
            column=1
        )

        self.peak_var = tk.IntVar(
            value=4
        )

        self.peak_spin = tk.Spinbox(
            frm,
            from_=1,
            to=4,
            textvariable=self.peak_var,
            width=5
        )

        self.peak_spin.grid(
            row=1,
            column=1
        )

        tk.Label(
            frm,
            text="Minimum Lorentzians"
        ).grid(row=0, column=4)

        self.min_peak_var = tk.IntVar(value=2)
        self.min_peak_spin = tk.Spinbox(
            frm, from_=1, to=4, textvariable=self.min_peak_var, width=5
        )
        self.min_peak_spin.grid(row=1, column=4)

        tk.Label(
            frm,
            text="Weak-line threshold (sigma)"
        ).grid(row=0, column=5, sticky="w")

        self.ent_weak_sigma = tk.Entry(frm, width=8)
        self.ent_weak_sigma.insert(0, "4.0")
        self.ent_weak_sigma.grid(row=1, column=5, sticky="w")

        self.fast_batch_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            frm, text="Fast batch fitting", variable=self.fast_batch_var
        ).grid(row=2, column=4, columnspan=2, sticky="w")

        tk.Label(
            frm,
            text="Fit Bmin (T)"
        ).grid(
            row=0,
            column=2,
            sticky="w"
        )

        self.ent_fit_bmin = tk.Entry(frm, width=10)
        self.ent_fit_bmin.insert(0, "2.5")
        self.ent_fit_bmin.grid(row=1, column=2, sticky="w")

        tk.Label(
            frm,
            text="Fit Bmax (T)"
        ).grid(
            row=0,
            column=3,
            sticky="w"
        )

        self.ent_fit_bmax = tk.Entry(frm, width=10)
        self.ent_fit_bmax.insert(0, "4.6")
        self.ent_fit_bmax.grid(row=1, column=3, sticky="w")

        self.auto_fit_range_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            frm, text="Auto fit range", variable=self.auto_fit_range_var
        ).grid(row=2, column=2, columnspan=2, sticky="w")

        tk.Button(
            frm,
            text="Fit Selected",
            command=self.do_fit_selected
        ).grid(
            row=3,
            column=0,
            pady=10
        )

        tk.Button(
            frm,
            text="Fit All Angles",
            command=self.do_fit_all_angles
        ).grid(
            row=3,
            column=1,
            pady=10
        )

        tk.Button(
            frm,
            text="Fit Selected by Mouse",
            command=self.do_fit_selected_mouse
        ).grid(
            row=4,
            column=0,
            pady=10
        )

        tk.Button(
            frm,
            text="Auto Fit All + Plot",
            command=self.do_auto_fit_all_and_plot
        ).grid(
            row=4,
            column=1,
            pady=10
        )

        self.auto_export_fits_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            frm, text="Save fitted curves/results after Fit All",
            variable=self.auto_export_fits_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=8)

        preset_frame = tk.Frame(frm)
        preset_frame.grid(row=6, column=0, columnspan=7, sticky="w", pady=8)
        tk.Label(preset_frame, text="Component presets:").pack(side="left", padx=(0, 6))
        tk.Button(
            preset_frame,
            text="Two dominant only",
            command=lambda: self.set_component_limits(2, 2),
        ).pack(side="left", padx=3)
        tk.Button(
            preset_frame,
            text="Automatic 2–4",
            command=lambda: self.set_component_limits(2, 4),
        ).pack(side="left", padx=3)
        tk.Button(
            preset_frame,
            text="Force four",
            command=lambda: self.set_component_limits(4, 4),
        ).pack(side="left", padx=3)
        tk.Label(
            frm,
            text=("The maximum is a hard upper bound: with Maximum Lorentzians = 2, "
                  "no third or fourth component can be added."),
            fg="dark green",
        ).grid(row=7, column=0, columnspan=7, sticky="w", pady=(0, 8))

    def set_component_limits(self, minimum, maximum):
        """Set and normalize the hard Lorentzian-count bounds."""
        minimum = max(1, min(4, int(minimum)))
        maximum = max(minimum, min(4, int(maximum)))
        self.min_peak_var.set(minimum)
        self.peak_var.set(maximum)

    def do_auto_fit_all_and_plot(self):

        self.do_fit_all_angles()

        if not self.fit_results:
            messagebox.showwarning(
                "No fit results",
                "Automatic fitting produced no results."
            )
            return

        direction = self.fit_direction.get().lower()

        fig1 = plot_B0_vs_angle(
            self.fit_results, title_prefix=self.get_title_prefix()
        )

        fig2 = plot_g_vs_angle(
            self.fit_results, title_prefix=self.get_title_prefix()
        )

        # Keep raw-analysis plots on the same sweep direction by default.
        if hasattr(self, "analysis_direction"):
            self.analysis_direction.set(direction)

        plt.show()

    def do_fit_selected_mouse(self):

        curves = self.get_selected_curves()

        if not curves:
            messagebox.showwarning(
                "No curves",
                "Select at least one curve."
            )
            return

        if len(curves) != 1:
            messagebox.showwarning(
                "Choose one measurement",
                "Mouse fitting operates on exactly one measurement. The program never averages selected files.",
            )
            return

        try:
            freq = float(self.ent_freq.get())
        except Exception:
            freq = DEFAULT_FREQ_GHZ

        model = self.fit_model.get()
        npeaks = int(self.peak_var.get())

        from fit_models import (
            extract_peak_table,
            g_factor,
            multi_peak_derivative,
            multi_peak_mixed,
        )

        Bref = np.asarray(curves[0]["B"], float)
        y1 = np.asarray(curves[0]["y1"], float)

        fig, ax = plt.subplots(
            figsize=(10, 7),
            constrained_layout=True
        )

        ax.plot(
            Bref,
            y1,
            lw=1.2,
            label=curves[0]["name"]
        )

        ax.set_title(
            "Click LEFT-RIGHT pairs for fit windows, then press ENTER"
        )
        ax.set_xlabel("Magnetic field B (T)")
        ax.set_ylabel("Signal CH1")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

        pts = plt.ginput(
            n=-1,
            timeout=0
        )

        if len(pts) < 2:
            plt.close(fig)
            return

        if len(pts) % 2 != 0:
            messagebox.showwarning(
                "Selection error",
                "You must select pairs of left/right limits."
            )
            plt.close(fig)
            return

        fit_windows = []

        for i in range(0, len(pts), 2):
            x1 = pts[i][0]
            x2 = pts[i + 1][0]
            bmin = min(x1, x2)
            bmax = max(x1, x2)
            fit_windows.append((bmin, bmax))

        angle = float(curves[0]["angle"])

        total_added = 0
        failed_windows = 0

        for window_index, (bmin, bmax) in enumerate(fit_windows, start=1):

            mask = (
                    (Bref >= bmin)
                    &
                    (Bref <= bmax)
            )

            if np.sum(mask) < 20:
                failed_windows += 1
                continue

            Bfit = Bref[mask]
            yfit = y1[mask]

            try:
                popt, pcov = fit_curve(
                    Bfit,
                    yfit,
                    npeaks=npeaks,
                    model=model
                )
            except Exception:
                failed_windows += 1
                continue

            if model == "mixed":
                y_model = multi_peak_mixed(
                    Bfit,
                    *popt
                )
            else:
                y_model = multi_peak_derivative(
                    Bfit,
                    *popt
                )

            ax.axvspan(
                bmin,
                bmax,
                alpha=0.12
            )

            ax.plot(
                Bfit,
                y_model,
                lw=2.3,
                label=f"Fit {window_index}: {bmin:.3f}-{bmax:.3f} T"
            )

            peak_rows = extract_peak_table(
                popt,
                model=model
            )

            for p in peak_rows:
                row = {
                    "angle": angle,
                    "window": window_index,
                    "peak": p["peak"],
                    "B0": p["B0"],
                    "gamma": p["gamma"],
                    "g": g_factor(freq, p["B0"]),
                    "model": model,
                    "npeaks": npeaks,
                    "fit_bmin": bmin,
                    "fit_bmax": bmax,
                }

                if "A" in p:
                    row["A"] = p["A"]

                if "theta" in p:
                    row["theta"] = p["theta"]

                self.fit_results.append(row)
                total_added += 1

        ax.legend(loc="best")
        fig.canvas.draw_idle()

        messagebox.showinfo(
            "Fit complete",
            f"Fit complete.\n"
            f"Windows selected: {len(fit_windows)}\n"
            f"Peak result(s) added: {total_added}\n"
            f"Failed/skipped windows: {failed_windows}"
        )

        plt.show()

    def apply_fit_window(self, B, y):

        B = np.asarray(B, float)
        y = np.asarray(y, float)

        if getattr(self, "auto_fit_range_var", None) is not None and self.auto_fit_range_var.get():
            from fit_models import auto_fit_window
            bmin, bmax = auto_fit_window(B, y, fallback=(2.5, 4.6))
            self.ent_fit_bmin.delete(0, "end")
            self.ent_fit_bmin.insert(0, f"{bmin:.4g}")
            self.ent_fit_bmax.delete(0, "end")
            self.ent_fit_bmax.insert(0, f"{bmax:.4g}")
        else:
            try:
                bmin = float(self.ent_fit_bmin.get())
            except Exception:
                bmin = float(np.min(B))
            try:
                bmax = float(self.ent_fit_bmax.get())
            except Exception:
                bmax = float(np.max(B))

        mask = (B >= bmin) & (B <= bmax)
        if np.sum(mask) < 20:
            messagebox.showwarning("Bad fit range", "Fit range contains too few points.")
            return B, y
        return B[mask], y[mask]

    def get_fit_options(self):

        # Both limits are hard constraints passed to fit_curve. In particular,
        # max_components=2 makes residual discovery stop after two physical
        # Lorentzians even when further structure is visible.
        max_components = max(1, min(4, int(self.peak_var.get())))
        min_components = max(1, min(4, int(self.min_peak_var.get())))
        min_components = max(1, min(min_components, max_components))
        self.peak_var.set(max_components)
        self.min_peak_var.set(min_components)

        try:
            weak_sigma = float(self.ent_weak_sigma.get())
        except Exception:
            weak_sigma = 4.0
        weak_sigma = max(2.0, weak_sigma)

        return max_components, min_components, weak_sigma

    def do_fit_selected(self):

        curves = self.get_selected_curves()

        if not curves:
            messagebox.showwarning(
                "No curves",
                "Select at least one curve."
            )
            return

        try:
            freq = float(self.ent_freq.get())
        except Exception:
            freq = DEFAULT_FREQ_GHZ

        model = self.fit_model.get()
        npeaks, min_components, weak_sigma = self.get_fit_options()

        from fit_models import fit_spectrum_series
        series = []
        for curve in curves:
            Bfit, yfit = self.apply_fit_window(curve["B"], curve["y1"])
            series.append({
                "measurement_id": curve["name"], "name": curve["name"],
                "path": curve.get("path", ""), "angle": float(curve["angle"]),
                "direction": curve.get("direction", "unknown"),
                "temperature_K": curve.get("temperature_K"),
                "frequency_GHz": curve.get("frequency_GHz"),
                "B": Bfit, "y": yfit,
            })
        fitted = fit_spectrum_series(
            series, npeaks=npeaks, model=model,
            min_components=min_components, weak_sigma=weak_sigma,
            fast_batch=False,
        )
        # Replace stored versions of these measurements, never append duplicate
        # rows for an already fitted file.
        selected_ids = {x["measurement_id"] for x in series}
        self.series_fit_results = [
            x for x in self.series_fit_results
            if str(x.get("measurement_id", x.get("name", ""))) not in selected_ids
        ] + fitted
        self._rebuild_current_fit_rows()
        failed = [x for x in fitted if not x.get("success")]
        if len(fitted) == 1 and fitted[0].get("success"):
            s = fitted[0]
            fig = plot_fit(s["B"], s["y"], s["popt"], model=model,
                           title=f"{s['name']} | {s['angle']:.3f}° | individual measurement")
            plt.show()
        messagebox.showinfo(
            "Fit complete",
            f"Measurements fitted separately: {len(fitted)-len(failed)}\nFailed: {len(failed)}",
        )

    def do_fit_all_angles(self):

        if not self.clusters:
            messagebox.showwarning("No data", "Load data first.")
            return

        try:
            freq = float(self.ent_freq.get())
        except Exception:
            freq = DEFAULT_FREQ_GHZ

        model = self.fit_model.get()
        npeaks, min_components, weak_sigma = self.get_fit_options()

        from processing import baseline_center_individual
        from fit_models import fit_spectrum_series

        # Fit exactly one sweep direction at a time.  Mixing UP and DOWN
        # would combine potentially hysteretic spectra and can corrupt domain
        # tracking, intensity ratios, and crossing analysis.
        direction = self.fit_direction.get().lower()
        if direction not in ("up", "down"):
            direction = "up"

        series = []

        for cluster in self.get_included_clusters():
            curves_all = baseline_center_individual(cluster)
            if not curves_all:
                continue

            curves = [c for c in curves_all if c["direction"] == direction]
            if not curves:
                continue

            for curve in curves:
                try:
                    Bfit, yfit = self.apply_fit_window(curve["B"], curve["y1"])
                    series.append({
                        "measurement_id": curve["name"],
                        "name": curve["name"],
                        "path": curve.get("path", ""),
                        "angle": float(curve["angle"]),
                        "direction": direction,
                        "temperature_K": curve.get("temperature_K"),
                        "frequency_GHz": curve.get("frequency_GHz"),
                        "B": Bfit,
                        "y": yfit,
                    })
                except Exception:
                    continue

        if not series:
            messagebox.showwarning(
                "No spectra",
                f"No {direction.upper()} spectra were found in the included angle clusters."
            )
            return

        fitted = fit_spectrum_series(
            series,
            npeaks=npeaks,
            model=model,
            min_components=min_components,
            weak_sigma=weak_sigma,
            fast_batch=bool(self.fast_batch_var.get()),
        )
        failed = sum(1 for fit in fitted if not fit.get("success"))
        results = rebuild_fit_rows(fitted, freq)

        # Replace previous batch results; do not mix UP and DOWN analyses.
        self.fit_results = results
        self.series_fit_results = fitted

        if hasattr(self, "analysis_direction"):
            self.analysis_direction.set(direction)

        messagebox.showinfo(
            "Fit complete",
            f"Automatic {direction.upper()}-sweep fitting complete.\n"
            f"Spectra fitted: {len(fitted) - failed}\n"
            f"Peak rows saved: {len(results)}\n"
            f"Failed spectra: {failed}"
        )
        if self.auto_export_fits_var.get():
            self.do_export_all_fitted(silent=True)

    def build_analysis_tab(self):

        frm = self.tab_analysis

        tk.Button(
            frm,
            text="Waterfall",
            command=self.do_waterfall
        ).pack(
            pady=10
        )

        tk.Button(
            frm,
            text="Export All Fitted Measurements",
            command=self.do_export_all_fitted,
        ).pack(pady=10)

        tk.Button(
            frm,
            text="Heatmap",
            command=self.do_heatmap
        ).pack(
            pady=10
        )

        tk.Button(
            frm,
            text="B0 vs Angle",
            command=self.do_B0_vs_angle
        ).pack(
            pady=10
        )

        tk.Button(
            frm,
            text="g vs Angle",
            command=self.do_g_vs_angle
        ).pack(
            pady=10
        )

        session_frame = tk.Frame(frm)
        session_frame.pack(pady=8)
        tk.Button(session_frame, text="Save Fit Session", command=self.do_save_fit_session).pack(side="left", padx=4)
        tk.Button(session_frame, text="Load Fit Session", command=self.do_load_fit_session).pack(side="left", padx=4)
        tk.Button(session_frame, text="Load Fit CSV (plots)", command=self.do_load_fit_csv).pack(side="left", padx=4)

        tk.Button(
            frm,
            text="Export Fit Results",
            command=self.do_export_fit_results
        ).pack(
            pady=10
        )

        tk.Button(
            frm,
            text="Save Waterfall PNG",
            command=self.do_save_waterfall
        ).pack(
            pady=10
        )

        tk.Button(
            frm,
            text="Save Heatmap PNG",
            command=self.do_save_heatmap
        ).pack(pady=10)

        tk.Button(frm, text="Linewidth vs Angle", command=self.do_linewidth_vs_angle).pack(pady=6)
        tk.Button(frm, text="Intensity vs Angle", command=self.do_intensity_vs_angle).pack(pady=6)
        tk.Button(frm, text="Domain Fractions vs Angle", command=self.do_fraction_vs_angle).pack(pady=6)
        tk.Button(frm, text="Fitted Signal Heatmap", command=self.do_fitted_heatmap).pack(pady=6)
        tk.Button(frm, text="Residual Heatmap", command=self.do_residual_heatmap).pack(pady=6)
        tk.Button(frm, text="Open Interactive Fit Browser", command=self.do_inspect_batch_fits).pack(pady=6)
        tk.Button(frm, text="Crossing Diagnostic", command=self.do_crossing_diagnostic).pack(pady=6)

        tk.Label(
            frm,
            text="Direction for Waterfall/Heatmap"
        ).pack(pady=(10, 2))

        self.analysis_direction = tk.StringVar(value="all")

        tk.OptionMenu(
            frm,
            self.analysis_direction,
            "all",
            "up",
            "down"
        ).pack(pady=(0, 10))

        tk.Label(frm, text="Waterfall vertical scale").pack(pady=(10, 2))

        self.ent_waterfall_scale = tk.Entry(frm, width=10)
        self.ent_waterfall_scale.insert(0, "3.0")
        self.ent_waterfall_scale.pack(pady=(0, 10))

    def build_angle_curves_for_analysis(self):

        if not self.clusters:
            return []

        direction_mode = self.analysis_direction.get().lower()

        angle_curves = []

        from processing import baseline_center_individual

        for cluster in self.get_included_clusters():

            curves = baseline_center_individual(cluster)

            if direction_mode in ["up", "down"]:
                curves = [
                    c for c in curves
                    if c["direction"] == direction_mode
                ]

            if not curves:
                continue

            for curve in curves:
                angle_curves.append({
                    "measurement_id": curve["name"],
                    "name": curve["name"],
                    "angle": float(curve["angle"]),
                    "B": np.asarray(curve["B"], float),
                    "y": np.asarray(curve["y1"], float),
                    "direction": curve.get("direction", direction_mode),
                    "temperature_K": curve.get("temperature_K"),
                    "frequency_GHz": curve.get("frequency_GHz"),
                })

        return angle_curves

    def do_waterfall(self):

        angle_curves = self.build_angle_curves_for_analysis()

        if not angle_curves:
            messagebox.showwarning("No data", "No included angle curves.")
            return

        try:
            freq = float(self.ent_freq.get())
        except Exception:
            freq = DEFAULT_FREQ_GHZ

        try:
            scale = float(self.ent_waterfall_scale.get())
        except Exception:
            scale = 3.0

        direction = self.analysis_direction.get().upper()

        fig = plot_waterfall(
            angle_curves,
            freq_GHz=freq,
            scale = scale,
            title_prefix=f"ESR {direction}"
            if self.get_title_prefix() == "ESR" else f"{self.get_title_prefix()} {direction}"
        )

        plt.show()

    def do_heatmap(self):

        angle_curves = self.build_angle_curves_for_analysis()

        if not angle_curves:
            messagebox.showwarning("No data", "No included angle curves.")
            return

        try:
            freq = float(self.ent_freq.get())
        except Exception:
            freq = DEFAULT_FREQ_GHZ

        direction = self.analysis_direction.get().upper()

        fig = plot_heatmap(
            angle_curves,
            freq_GHz=freq,
            title_prefix=f"ESR {direction}"
            if self.get_title_prefix() == "ESR" else f"{self.get_title_prefix()} {direction}"
        )

        plt.show()

    def do_save_waterfall(self):

        angle_curves = self.build_angle_curves_for_analysis()

        if not angle_curves:
            messagebox.showwarning("No data", "No included angle curves.")
            return

        try:
            freq = float(self.ent_freq.get())
        except Exception:
            freq = DEFAULT_FREQ_GHZ

        try:
            scale = float(self.ent_waterfall_scale.get())
        except Exception:
            scale = 3.0

        direction = self.analysis_direction.get().upper()

        fig = plot_waterfall(
            angle_curves,
            freq_GHz=freq,
            scale=scale,
            title_prefix=f"ESR {direction}"
            if self.get_title_prefix() == "ESR" else f"{self.get_title_prefix()} {direction}"
        )

        out_dir = self.get_fit_output_dir()

        direction = self.analysis_direction.get().upper()

        filename = f"WATERFALL_{direction}_f{freq:.1f}GHz_n{len(angle_curves)}.png"

        path = export_figure(
            fig,
            out_dir,
            filename
        )

        plt.close(fig)

        messagebox.showinfo("Saved", path)

    def do_save_heatmap(self):

        angle_curves = self.build_angle_curves_for_analysis()

        if not angle_curves:
            messagebox.showwarning("No data", "No included angle curves.")
            return

        try:
            freq = float(self.ent_freq.get())
        except Exception:
            freq = DEFAULT_FREQ_GHZ

        fig = plot_heatmap(
            angle_curves,
            freq_GHz=freq,
            title_prefix=(f"ESR {self.analysis_direction.get().upper()}"
                          if self.get_title_prefix() == "ESR"
                          else f"{self.get_title_prefix()} {self.analysis_direction.get().upper()}")
        )

        out_dir = self.get_fit_output_dir()

        direction = self.analysis_direction.get().upper()

        filename = f"HEATMAP_{direction}_f{freq:.1f}GHz_n{len(angle_curves)}.png"

        path = export_figure(
            fig,
            out_dir,
            filename
        )

        plt.close(fig)

        messagebox.showinfo("Saved", path)

    def do_B0_vs_angle(self):

        if not self.fit_results:
            messagebox.showwarning(
                "No fit results",
                "Run Fit All Angles first."
            )

            return

        fig = plot_B0_vs_angle(
            self.fit_results, title_prefix=self.get_title_prefix()
        )

        plt.show()

    def do_g_vs_angle(self):

        if not self.fit_results:
            messagebox.showwarning(
                "No fit results",
                "Run Fit All Angles first."
            )

            return

        fig = plot_g_vs_angle(
            self.fit_results, title_prefix=self.get_title_prefix()
        )

        plt.show()

    def do_export_fit_results(self):

        if not self.fit_results:
            messagebox.showwarning(
                "No fit results",
                "Run Fit All Angles first."
            )

            return

        path = export_fit_results_csv(
            self.fit_results,
            self.get_fit_output_dir(),
        )

        messagebox.showinfo(
            "Export complete",
            path
        )

    def do_export_all_fitted(self, silent=False):
        if not self.series_fit_results:
            if not silent:
                messagebox.showwarning("No fitted measurements", "Run Fit All first.")
            return None
        try:
            result = export_fitted_measurements(
                self.series_fit_results, self.fit_results, self.get_fit_output_dir()
            )
            if not silent:
                messagebox.showinfo(
                    "Fitted measurements exported",
                    f"Summary: {result['summary']}\n"
                    f"Individual fitted curves: {len(result['curves'])}\n"
                    f"Directory: {result['directory']}",
                )
            return result
        except Exception as exc:
            if not silent:
                messagebox.showerror("Export failed", str(exc))
            return None

    def _need_fit_results(self):
        if not self.fit_results:
            messagebox.showwarning("No fit results", "Run Fit All Angles first.")
            return False
        return True

    def do_linewidth_vs_angle(self):
        if self._need_fit_results():
            plt.show(block=False) if False else None
            fig = plot_linewidth_vs_angle(self.fit_results, title_prefix=self.get_title_prefix())
            plt.show()

    def do_intensity_vs_angle(self):
        if self._need_fit_results():
            fig = plot_intensity_vs_angle(self.fit_results, normalized=False, title_prefix=self.get_title_prefix())
            plt.show()

    def do_fraction_vs_angle(self):
        if self._need_fit_results():
            fig = plot_intensity_vs_angle(self.fit_results, normalized=True, title_prefix=self.get_title_prefix())
            plt.show()

    def do_fitted_heatmap(self):
        if not self.series_fit_results:
            messagebox.showwarning("No detailed fits", "Run Fit All Angles first.")
            return
            fig = plot_fit_heatmaps(self.series_fit_results, mode="fitted", title_prefix=self.get_title_prefix())
        plt.show()

    def do_residual_heatmap(self):
        if not self.series_fit_results:
            messagebox.showwarning("No detailed fits", "Run Fit All Angles first.")
            return
            fig = plot_fit_heatmaps(self.series_fit_results, mode="residual", title_prefix=self.get_title_prefix())
        plt.show()

    def _rebuild_current_fit_rows(self):
        """Re-track all domains and rebuild the rows used by plots/exports."""
        try:
            freq = float(self.ent_freq.get())
        except Exception:
            freq = DEFAULT_FREQ_GHZ
        track_domain_branches(self.series_fit_results, max_domains=4)
        self.fit_results = rebuild_fit_rows(self.series_fit_results, freq)

    def _refit_from_browser(self, spectrum, min_components, max_components, weak_sigma,
                            manual_centers=None, center_tolerance=0.08):
        """Replace one stored fit, then globally re-track domain identities.

        When ``manual_centers`` is supplied, exactly those physical resonance
        centers are used to initialize the fit and the component count is
        forced to the number of selected centers.
        """
        from fit_models import fit_curve, extract_peak_table, build_initial_guess
        try:
            model = str(spectrum.get("details", {}).get("model", self.fit_model.get()))
            p0 = None
            if manual_centers:
                centers = sorted(float(c) for c in manual_centers)
                if not 1 <= len(centers) <= 4:
                    raise ValueError("Select between 1 and 4 resonance centers.")
                min_components = max_components = len(centers)
                p0 = build_initial_guess(
                    spectrum["B"], spectrum["y"], len(centers),
                    mixed=(model == "mixed"), centers=centers,
                )
            # Deliberately do a thorough fit for manually reviewed spectra.
            popt, pcov, details = fit_curve(
                spectrum["B"], spectrum["y"],
                npeaks=int(max_components), model=model,
                min_components=int(min_components), p0=p0,
                weak_sigma=float(weak_sigma), fast=False,
                manual_centers=centers if manual_centers else None,
                center_tolerance=float(center_tolerance),
                return_details=True,
            )
            peaks = sorted(extract_peak_table(popt, model=model), key=lambda r: r["B0"])[:4]
            # Replace, rather than append: old saved values for this spectrum
            # cease to exist in both the detailed and flattened result lists.
            spectrum["popt"] = popt
            spectrum["pcov"] = pcov
            spectrum["details"] = details
            spectrum["peaks"] = peaks
            spectrum["success"] = True
            spectrum.pop("error", None)
            spectrum["review_status"] = "unreviewed"
            spectrum.pop("review_reasons", None)
            self._rebuild_current_fit_rows()
            return True
        except Exception as exc:
            messagebox.showerror("Refit failed", str(exc))
            return False


    def _assign_domain_from_browser(self, spectrum, selected_B0, new_domain_id,
                                    lock_assignment=True, scope="current"):
        """Rename/swap persistent domain IDs from the interactive browser.

        ``scope`` follows the browser order and may be ``current``, ``after``,
        ``before``, or ``everything`` (legacy aliases forward/backward/all are
        accepted).  The same pair of tracked IDs is swapped through that part
        of the *same temperature/frequency/direction series*.  Missing domains
        remain gaps. Edited rows are optionally locked so later re-tracking
        cannot undo the checked correction.
        """
        try:
            rows = list(spectrum.get("peaks", []))
            if not rows:
                raise ValueError("The selected spectrum has no fitted components.")
            selected = min(rows, key=lambda r: abs(float(r["B0"]) - float(selected_B0)))
            old_id = int(selected.get("domain_id", selected.get("peak", 0)))
            new_id = int(new_domain_id)
            if new_id not in (1, 2, 3, 4):
                raise ValueError("Domain ID must be 1, 2, 3, or 4.")
            direction = str(spectrum.get("direction", "unknown")).lower()
            scope = str(scope or "current").lower()
            scope = {"forward": "after", "backward": "before", "all": "everything"}.get(scope, scope)
            if scope not in ("current", "after", "before", "everything"):
                raise ValueError("Rename scope must be current, after, before, or everything.")

            temp_key = metadata_value_key(spectrum.get("temperature_K"))
            freq_key = metadata_value_key(spectrum.get("frequency_GHz"))

            def same_series(other):
                return (
                    str(other.get("direction", "unknown")).lower() == direction
                    and metadata_value_key(other.get("temperature_K")) == temp_key
                    and metadata_value_key(other.get("frequency_GHz")) == freq_key
                )

            # This is the same ordering used by the interactive browser.  It
            # makes "after" exactly the spectra reached with the right arrow
            # and "before" those reached with the left arrow, including when
            # repeated measurements share the same angle.
            ordered = sorted(
                [q for q in self.series_fit_results if same_series(q)],
                key=lambda q: (
                    float(q.get("angle", np.nan)),
                    str(q.get("name", q.get("filename", q.get("measurement_id", "")))),
                ),
            )
            try:
                current_index = next(i for i, q in enumerate(ordered) if q is spectrum)
            except StopIteration:
                sid = str(spectrum.get("measurement_id", spectrum.get("name", "")))
                current_index = next(
                    i for i, q in enumerate(ordered)
                    if str(q.get("measurement_id", q.get("name", ""))) == sid
                )

            if scope == "current":
                targets = [ordered[current_index]]
            elif scope == "after":
                targets = ordered[current_index:]
            elif scope == "before":
                targets = ordered[:current_index + 1]
            else:
                targets = ordered

            for q in targets:
                qrows = list(q.get("peaks", []))
                for row in qrows:
                    did = int(row.get("domain_id", row.get("peak", 0)))
                    if did == old_id:
                        row["domain_id"] = new_id
                        row["peak"] = new_id
                        row["domain_locked"] = bool(lock_assignment)
                    elif did == new_id:
                        row["domain_id"] = old_id
                        row["peak"] = old_id
                        row["domain_locked"] = bool(lock_assignment)
                q["peaks"] = qrows
                q["tracking_status"] = "manually_propagated"
                q["tracking_reasons"] = []

            # Ensure the specifically clicked row is always updated, even if
            # scope matching was affected by object reloading.
            selected["domain_id"] = new_id
            selected["peak"] = new_id
            selected["domain_locked"] = bool(lock_assignment)
            spectrum["peaks"] = rows

            self._rebuild_current_fit_rows()
            return True
        except Exception as exc:
            messagebox.showerror("Domain reassignment failed", str(exc))
            return False

    def _set_tracking_reference(self, spectrum, domain_ids):
        """Use a trusted 1--4-line spectrum as the domain-tracking anchor.

        ``domain_ids`` correspond to the fitted resonances sorted from left to
        right.  They must be unique values in 1..4.  The assignments are
        locked, all older reference flags for this sweep direction are removed,
        and tracking is recalculated both forward and backward in angle.
        """
        try:
            ids = [int(x) for x in domain_ids]
            rows = sorted(spectrum.get("peaks", []), key=lambda r: float(r["B0"]))
            if not 1 <= len(rows) <= 4:
                raise ValueError("Choose a reference spectrum with between one and four fitted domains.")
            if len(ids) != len(rows) or len(set(ids)) != len(ids) or any(x not in (1,2,3,4) for x in ids):
                raise ValueError("Reference labels must be unique domain IDs from 1 to 4.")
            direction = str(spectrum.get("direction", "unknown")).lower()
            temp = spectrum.get("temperature_K")
            frequency = spectrum.get("frequency_GHz")
            for other in self.series_fit_results:
                same_series = (
                    str(other.get("direction", "unknown")).lower() == direction
                    and metadata_value_key(other.get("temperature_K")) == metadata_value_key(temp)
                    and metadata_value_key(other.get("frequency_GHz")) == metadata_value_key(frequency)
                )
                if same_series:
                    other["tracking_reference"] = False
                    for row in other.get("peaks", []):
                        # Keep only explicit locks on the new reference.  This
                        # avoids obsolete assignments fighting the new anchor.
                        row["domain_locked"] = False
            for row, did in zip(rows, ids):
                row["domain_id"] = did
                row["peak"] = did
                row["domain_locked"] = True
            spectrum["peaks"] = rows
            spectrum["tracking_reference"] = True
            spectrum["tracking_status"] = "reference"
            spectrum["tracking_reasons"] = []
            self._rebuild_current_fit_rows()
            return True
        except Exception as exc:
            messagebox.showerror("Reference assignment failed", str(exc))
            return False

    def _set_spectrum_review_status(self, spectrum, status):
        spectrum["review_status"] = str(status)
        self._rebuild_current_fit_rows()
        return True

    def _session_settings(self):
        return {
            "material_name": self.ent_material.get().strip(),
            "fit_model": self.fit_model.get(),
            "fit_direction": self.fit_direction.get(),
            "max_components": int(self.peak_var.get()),
            "min_components": int(self.min_peak_var.get()),
            "weak_sigma": self.ent_weak_sigma.get(),
            "fit_bmin": self.ent_fit_bmin.get(),
            "fit_bmax": self.ent_fit_bmax.get(),
            "fast_batch": bool(self.fast_batch_var.get()),
            "auto_fit_range": bool(self.auto_fit_range_var.get()),
            "frequency_GHz": self.ent_freq.get(),
            "folder": self.folder,
            "fit_output_dir": self.fit_output_dir,
            "temperature_filter": self.temperature_filter.get(),
            "frequency_filter": self.frequency_filter.get(),
        }

    def do_save_fit_session(self):
        if not self.series_fit_results:
            messagebox.showwarning("No fit session", "Run Fit All Angles first.")
            return
        os.makedirs(self.get_fit_output_dir(), exist_ok=True)
        path = filedialog.asksaveasfilename(
            defaultextension=".esrsession",
            filetypes=[("ESR fit session", "*.esrsession"), ("All files", "*")],
            initialfile=f"esr_{self.fit_direction.get()}_fit.esrsession",
            initialdir=self.get_fit_output_dir(),
        )
        if not path:
            return
        payload = {
            "version": 2,
            "settings": self._session_settings(),
            "series_fit_results": self.series_fit_results,
            "fit_results": self.fit_results,
        }
        try:
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            messagebox.showinfo("Session saved", path)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def do_load_fit_session(self):
        path = filedialog.askopenfilename(
            filetypes=[("ESR fit session", "*.esrsession"), ("All files", "*")]
        )
        if not path:
            return
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict) or "series_fit_results" not in payload:
                raise ValueError("This is not a valid ESR fit session.")
            self.series_fit_results = payload.get("series_fit_results", [])
            settings = payload.get("settings", {})
            if settings.get("fit_model") in ("mixed", "derivative"):
                self.fit_model.set(settings["fit_model"])
            if settings.get("fit_direction") in ("up", "down"):
                self.fit_direction.set(settings["fit_direction"])
                self.analysis_direction.set(settings["fit_direction"])
            self.peak_var.set(int(settings.get("max_components", self.peak_var.get())))
            self.min_peak_var.set(int(settings.get("min_components", self.min_peak_var.get())))
            for entry, key in [(self.ent_weak_sigma, "weak_sigma"),
                               (self.ent_fit_bmin, "fit_bmin"),
                               (self.ent_fit_bmax, "fit_bmax"),
                               (self.ent_freq, "frequency_GHz")]:
                if key in settings:
                    entry.delete(0, "end"); entry.insert(0, str(settings[key]))
            self.fast_batch_var.set(bool(settings.get("fast_batch", True)))
            self.auto_fit_range_var.set(bool(settings.get("auto_fit_range", False)))
            self.folder = settings.get("folder", self.folder)
            self.fit_output_dir = settings.get("fit_output_dir", self.fit_output_dir)
            if self.fit_output_dir:
                self.lbl_fit_output.config(text=self.fit_output_dir)
            if "material_name" in settings:
                self.ent_material.delete(0, "end")
                self.ent_material.insert(0, str(settings["material_name"]))
            self._rebuild_current_fit_rows()
            messagebox.showinfo("Session loaded",
                                f"Loaded {len(self.series_fit_results)} spectra.\nUse the interactive browser to continue review.")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def do_load_fit_csv(self):
        """Load flattened fit parameters for parameter plots.

        CSV files do not contain measured arrays or full model vectors, so the
        interactive fit browser and fitted/residual heatmaps require an
        ``.esrsession`` instead.
        """
        path = filedialog.askopenfilename(
            filetypes=[("Fit results CSV", "*.csv"), ("All files", "*")]
        )
        if not path:
            return
        numeric = {
            "angle", "B0", "gamma", "A", "theta", "intensity",
            "intensity_fraction", "g", "rmse", "bic", "temperature_K",
            "frequency_GHz",
        }
        integer = {"peak", "domain_id", "npeaks"}
        try:
            rows = []
            with open(path, newline="") as handle:
                for raw in csv.DictReader(handle):
                    row = dict(raw)
                    for key in numeric:
                        if key in row and row[key] not in ("", None):
                            row[key] = float(row[key])
                    for key in integer:
                        if key in row and row[key] not in ("", None):
                            row[key] = int(float(row[key]))
                    rows.append(row)
            if not rows or not all("angle" in r and "B0" in r for r in rows):
                raise ValueError("CSV does not contain fit result columns angle and B0.")
            self.fit_results = rows
            self.series_fit_results = []
            messagebox.showinfo(
                "Fit CSV loaded",
                f"Loaded {len(rows)} fitted-domain rows. Parameter plots are available.\n"
                "Load an .esrsession for the interactive browser and heatmaps.",
            )
        except Exception as exc:
            messagebox.showerror("CSV load failed", str(exc))

    def do_inspect_batch_fits(self):
        if not self.series_fit_results:
            messagebox.showwarning("No detailed fits", "Run Fit All Angles first.")
            return
        try:
            maxc, minc, sigma = self.get_fit_options()
        except Exception:
            minc, maxc, sigma = 2, 4, 4.0
        fig = plot_batch_fit_inspection(
            self.series_fit_results,
            title_prefix=self.get_title_prefix(),
            on_refit=self._refit_from_browser,
            on_status_change=self._set_spectrum_review_status,
            on_set_reference=self._set_tracking_reference,
            on_assign_domain=self._assign_domain_from_browser,
            default_min_components=minc,
            default_max_components=maxc,
            default_weak_sigma=sigma,
        )
        plt.show()

    def do_crossing_diagnostic(self):
        if self._need_fit_results():
            fig = plot_avoided_crossing_diagnostic(self.fit_results, title_prefix=self.get_title_prefix())
            if fig is None:
                messagebox.showwarning("Not enough domains", "Need at least two fitted domains per angle.")
                return
            plt.show()

    # ==================================================
    # STUBS
    # ==================================================

    def select_folder(self):
        folder = filedialog.askdirectory()

        if not folder:
            return

        self.folder = folder

        self.lbl_folder.config(
            text=folder
        )

        if not self.fit_output_dir:
            self.lbl_fit_output.config(text=os.path.join(folder, "FIT_out"))

    def select_fit_output_dir(self):
        folder = filedialog.askdirectory(title="Choose directory for fitted results")
        if folder:
            self.fit_output_dir = folder
            self.lbl_fit_output.config(text=folder)

    def get_fit_output_dir(self):
        return self.fit_output_dir or os.path.join(self.folder or os.getcwd(), "FIT_out")

    def get_title_prefix(self):
        material = self.ent_material.get().strip() if hasattr(self, "ent_material") else ""
        return material or "ESR"

    @staticmethod
    def _filter_label(value, unit):
        if value is None:
            return "Unknown"
        return f"{float(value):.4f}".rstrip("0").rstrip(".") + unit

    def _populate_metadata_filters(self):
        temperatures = sorted({metadata_value_key(x.get("temperature_K")) for x in self.items
                               if metadata_value_key(x.get("temperature_K")) is not None})
        frequencies = sorted({metadata_value_key(x.get("frequency_GHz")) for x in self.items
                              if metadata_value_key(x.get("frequency_GHz")) is not None})
        t_values = ["All"] + [self._filter_label(v, " K") for v in temperatures]
        f_values = ["All"] + [self._filter_label(v, " GHz") for v in frequencies]
        if any(x.get("temperature_K") is None for x in self.items):
            t_values.append("Unknown")
        if any(x.get("frequency_GHz") is None for x in self.items):
            f_values.append("Unknown")
        self.temperature_combo["values"] = t_values
        self.frequency_combo["values"] = f_values
        self.temperature_filter.set("All")
        self.frequency_filter.set("All")

    @staticmethod
    def _matches_metadata_filter(item_value, selected, unit):
        if selected == "All":
            return True
        if selected == "Unknown":
            return item_value is None
        try:
            requested = float(selected.replace(unit, "").strip())
            return (item_value is not None and
                    metadata_value_key(item_value) == metadata_value_key(requested))
        except Exception:
            return False

    def apply_metadata_filters(self):
        tsel = self.temperature_filter.get()
        fsel = self.frequency_filter.get()
        if fsel not in ("All", "Unknown"):
            try:
                parsed_frequency = float(fsel.replace("GHz", "").strip())
                self.ent_freq.delete(0, "end")
                self.ent_freq.insert(0, f"{parsed_frequency:g}")
            except Exception:
                pass
        self.filtered_items = [
            item for item in self.items
            if self._matches_metadata_filter(item.get("temperature_K"), tsel, "K")
            and self._matches_metadata_filter(item.get("frequency_GHz"), fsel, "GHz")
        ]
        try:
            tol = float(self.ent_tol.get())
        except Exception:
            tol = DEFAULT_ANGLE_TOL
        self.clusters = cluster_by_angle(self.filtered_items, tol) if self.filtered_items else []
        self.cluster_include_vars = [tk.BooleanVar(value=True) for _ in self.clusters]
        labels = []
        for i, cluster in enumerate(self.clusters, start=1):
            ang = float(np.mean([c["angle"] for c in cluster]))
            labels.append(f"{i:02d}: {ang:.2f}° (measurements={len(cluster)})")
        self.cluster_combo["values"] = labels
        self.lbl_filter_summary.config(
            text=f"{len(self.filtered_items)} / {len(self.items)} measurements"
        )
        if labels:
            self.cluster_combo.current(0)
            self.on_cluster_change()
        else:
            self.curves = []
            self.build_checkboxes()

    def clear_metadata_filters(self):
        self.temperature_filter.set("All")
        self.frequency_filter.set("All")
        self.apply_metadata_filters()

    def do_split_filtered_files(self):
        if not self.filtered_items:
            messagebox.showwarning("No files", "Load data and apply filters first.")
            return
        destination = filedialog.askdirectory(title="Choose parent folder for separated files")
        if not destination:
            return
        mode = self.split_mode.get().lower()
        if mode == "move":
            ok = messagebox.askyesno(
                "Move original files?",
                "This will move the filtered original .dat files into new folders. Continue?",
            )
            if not ok:
                return
        try:
            written = split_files_by_metadata(
                self.filtered_items, destination, mode=mode,
                include_direction=True, include_temperature=True,
                include_frequency=True,
            )
            messagebox.showinfo(
                "Separation complete",
                f"{mode.title()}ed {len(written)} files into metadata folders under:\n{destination}",
            )
            if mode == "move":
                self.load_data()
        except Exception as exc:
            messagebox.showerror("Separation failed", str(exc))

    def load_data(self):

        if not self.folder:
            messagebox.showwarning(
                "No folder",
                "Select a folder first."
            )

            return

        try:

            tol = float(
                self.ent_tol.get()
            )

        except Exception:

            tol = DEFAULT_ANGLE_TOL

        self.items = load_folder(
            self.folder,
            prefix=self.ent_prefix.get().strip(),
            ext=self.ent_ext.get().strip()
        )

        if not self.items:
            messagebox.showerror(
                "Error",
                "No ESR files found."
            )

            return

        self._populate_metadata_filters()
        self.apply_metadata_filters()

    def get_included_clusters(self):

        if not self.clusters:
            return []

        if not self.cluster_include_vars:
            return self.clusters

        included = []

        for cluster, var in zip(self.clusters, self.cluster_include_vars):
            if var.get():
                included.append(cluster)

        return included

    def on_cluster_change(
            self,
            event=None
    ):

        idx = self.cluster_combo.current()

        if idx < 0:
            return

        self.cluster_index = idx

        self.prepare_curves()

        self.build_checkboxes()

    def select_all(self):

        for v in self.check_vars:
            v.set(True)

    def select_none(self):

        for v in self.check_vars:
            v.set(False)

    def select_up(self):

        for var, curve in zip(
                self.check_vars,
                self.curves
        ):
            var.set(
                curve["direction"] == "up"
            )

    def select_down(self):

        for var, curve in zip(
                self.check_vars,
                self.curves
        ):
            var.set(
                curve["direction"] == "down"
            )

    def prepare_curves(self):

        if not self.clusters:
            return

        try:

            n = int(
                self.ent_n.get()
            )

        except Exception:

            n = DEFAULT_N_PER_ANGLE

        cluster = self.clusters[
            self.cluster_index
        ]

        # Keep every measurement independent.  Angle clusters are navigation
        # groups only; they never define an averaging or fitting unit.
        self.curves = baseline_center_individual(cluster)

        self.curves.sort(
            key=lambda c:
            (
                c.get("num", 10 ** 9),
                c["name"]
            )
        )

    def build_checkboxes(self):

        for w in self.checkbox_inner.winfo_children():
            w.destroy()

        self.check_vars = []

        for i, c in enumerate(
                self.curves,
                start=1
        ):
            txt = (
                f"{i:02d} "
                f"{c['direction']} "
                f"{c['name']}"
            )

            var = tk.BooleanVar(
                value=True
            )

            chk = tk.Checkbutton(
                self.checkbox_inner,
                text=txt,
                variable=var,
                anchor="w",
                justify="left"
            )

            chk.pack(
                fill="x",
                padx=5,
                pady=2
            )

            self.check_vars.append(
                var
            )

        self.checkbox_inner.update_idletasks()

        self.checkbox_canvas.configure(
            scrollregion=self.checkbox_canvas.bbox("all")
        )

    def get_selected_curves(self):

        chosen = []

        for var, curve in zip(
                self.check_vars,
                self.curves
        ):

            if var.get():
                chosen.append(
                    curve
                )

        return chosen

    # ==================================================
    # RUN
    # ==================================================

    def run(self):
        self.root.mainloop()

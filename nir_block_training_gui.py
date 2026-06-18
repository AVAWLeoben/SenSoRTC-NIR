# -*- coding: utf-8 -*-
"""Block-style GUI for training SenSoRTC NIR sklearn classifiers.

Run:
    python nir_block_training_gui.py

Optional dependency for file drag/drop:
    pip install tkinterdnd2
Without tkinterdnd2, use the Add Files button.
"""

from __future__ import annotations

import os
import contextlib
import csv
import io
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
import tkinter as tk
import numpy as np
from tkinter import ttk, filedialog, messagebox, simpledialog

import yaml

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    MATPLOTLIB_AVAILABLE = True
except Exception:
    Figure = None
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None
    MATPLOTLIB_AVAILABLE = False

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False
    TkinterDnD = None
    DND_FILES = None


FILE_TYPES = [
    ("NIR spectra", "*.xlsx *.xlsm *.mat *.npy *.npz"),
    ("Excel", "*.xlsx *.xlsm"),
    ("MAT", "*.mat"),
    ("NumPy", "*.npy *.npz"),
    ("All files", "*.*"),
]


class _TkTextRedirector:
    """Thread-safe stdout/stderr redirector for the training log."""

    def __init__(self, gui):
        self.gui = gui
        self._buffer = ""

    def write(self, text):
        if not text:
            return
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.gui.root.after(0, self.gui.log_line, line)

    def flush(self):
        if self._buffer:
            line = self._buffer
            self._buffer = ""
            self.gui.root.after(0, self.gui.log_line, line)


class ParamDialog(simpledialog.Dialog):
    def __init__(self, parent, title, fields):
        self.fields = fields
        self.vars = {}
        self.result = None
        super().__init__(parent, title)

    def body(self, master):
        """Build parameter form and return the first Entry widget for focus.

        Returning the Tk root here can make simpledialog focus handling behave
        inconsistently on Windows.  Keep an explicit reference to the first
        entry widget instead.
        """
        master.columnconfigure(1, weight=1)
        first_entry = None
        for row, (name, default) in enumerate(self.fields.items()):
            ttk.Label(master, text=name).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            var = tk.StringVar(value=str(default))
            self.vars[name] = var
            entry = ttk.Entry(master, textvariable=var, width=28)
            entry.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
            if first_entry is None:
                first_entry = entry
        return first_entry

    def apply(self):
        out = {}
        for name, var in self.vars.items():
            text = var.get().strip()
            if text.lower() in ("true", "false"):
                out[name] = text.lower() == "true"
            elif "," in text:
                try:
                    out[name] = [int(x.strip()) for x in text.split(",") if x.strip()]
                except Exception:
                    out[name] = text
            else:
                try:
                    out[name] = int(text)
                except Exception:
                    try:
                        out[name] = float(text)
                    except Exception:
                        out[name] = text
        self.result = out


class NIRBlockTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SenSoRTC NIR Block Training GUI - v11")
        self.root.geometry("1450x900")
        self.root.minsize(1250, 760)

        self._configure_style()

        self.classes = []
        self.preprocessing = []
        self.model = None

        self.output_var = tk.StringVar(value="models/nir_block_model.joblib")
        self.min_mean_var = tk.StringVar(value="600")
        self.test_size_var = tk.StringVar(value="0.20")
        self.conf_var = tk.StringVar(value="0.70")
        self.random_state_var = tk.StringVar(value="42")

        self.preview_data = None
        self.preview_fig = None
        self.preview_canvas = None
        self.preview_summary = None
        self.preview_view_var = tk.StringVar(value="processed")
        self.preview_std_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._refresh_all()

    def _configure_style(self):
        """Make the first GUI pass less cramped on Windows/Tk."""
        try:
            style = ttk.Style(self.root)
            style.configure("TButton", padding=(8, 5))
            style.configure("TLabel", padding=(0, 1))
            style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
            style.configure("Big.TEntry", padding=(4, 3))
            style.configure("Primary.TButton", padding=(10, 6))
        except Exception:
            pass

    def _make_setting_entry(self, parent, label, var, col, hint="", width=16):
        """Create a larger global numeric/text setting field."""
        cell = ttk.Frame(parent)
        cell.grid(row=1, column=col, sticky="ew", padx=(0, 14), pady=(8, 0))
        cell.columnconfigure(0, weight=1)
        ttk.Label(cell, text=label).grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(cell, textvariable=var, width=width, style="Big.TEntry")
        entry.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        if hint:
            ttk.Label(cell, text=hint).grid(row=2, column=0, sticky="w")
        return entry

    def _build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)

        self.tab_builder = ttk.Frame(self.notebook)
        self.tab_preview = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_builder, text="Pipeline Builder")
        self.notebook.add(self.tab_preview, text="Spectral Preview")

        main = ttk.Frame(self.tab_builder, padding=8)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=1)
        main.rowconfigure(1, weight=1)

        top = ttk.LabelFrame(main, text="Training output and global settings", padding=12)
        top.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        for col in range(7):
            top.columnconfigure(col, weight=1)

        ttk.Label(top, text="Output model (.joblib)").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.output_var, style="Big.TEntry").grid(
            row=0, column=1, columnspan=5, sticky="ew", padx=(8, 8)
        )
        ttk.Button(top, text="Browse...", command=self.browse_output).grid(row=0, column=6, sticky="ew")

        self._make_setting_entry(
            top,
            "min_mean",
            self.min_mean_var,
            0,
            hint="MAT/NPY/NPZ only; Excel unchanged",
            width=18,
        )
        self._make_setting_entry(
            top,
            "test_size",
            self.test_size_var,
            1,
            hint="e.g. 0.20",
            width=18,
        )
        self._make_setting_entry(
            top,
            "confidence threshold",
            self.conf_var,
            2,
            hint="runtime default",
            width=18,
        )
        self._make_setting_entry(
            top,
            "random_state",
            self.random_state_var,
            3,
            hint="reproducible split/model",
            width=18,
        )

        # Data panel
        data = ttk.LabelFrame(main, text="1) Training data classes", padding=8)
        data.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        data.rowconfigure(1, weight=1)
        data.columnconfigure(0, weight=1)
        ttk.Label(data, text="Excel spectra are treated as manually picked raw spectra and are not filtered.").grid(row=0, column=0, columnspan=2, sticky="w")
        self.class_list = tk.Listbox(data, height=9, font=("Segoe UI", 10), exportselection=False)
        self.class_list.grid(row=1, column=0, sticky="nsew", pady=6)
        self.class_list.bind("<<ListboxSelect>>", lambda e: self._refresh_files())
        class_buttons = ttk.Frame(data)
        class_buttons.grid(row=1, column=1, sticky="ns", padx=4)
        ttk.Button(class_buttons, text="Add Class", command=self.add_class).pack(fill="x", pady=2)
        ttk.Button(class_buttons, text="Remove", command=self.remove_class).pack(fill="x", pady=2)
        ttk.Button(class_buttons, text="Add Files", command=self.add_files).pack(fill="x", pady=2)
        ttk.Button(class_buttons, text="Remove File", command=self.remove_file).pack(fill="x", pady=2)

        ttk.Label(data, text="Files for selected class").grid(row=2, column=0, sticky="w")
        self.file_list = tk.Listbox(data, height=13, font=("Segoe UI", 9), exportselection=False)
        self.file_list.grid(row=3, column=0, columnspan=2, sticky="nsew")
        data.rowconfigure(3, weight=1)
        if DND_AVAILABLE:
            self.file_list.drop_target_register(DND_FILES)
            self.file_list.dnd_bind("<<Drop>>", self.drop_files)

        # Pipeline panel
        pipe = ttk.LabelFrame(main, text="2) Pipeline blocks", padding=8)
        pipe.grid(row=1, column=1, sticky="nsew", padx=6)
        pipe.rowconfigure(1, weight=1)
        pipe.columnconfigure(0, weight=1)
        ttk.Label(pipe, text="Add blocks in order. Model is always last.").grid(row=0, column=0, columnspan=2, sticky="w")
        self.pipeline_list = tk.Listbox(pipe, height=20, font=("Consolas", 10), exportselection=False)
        self.pipeline_list.grid(row=1, column=0, sticky="nsew", pady=6)
        pipeline_buttons = ttk.Frame(pipe)
        pipeline_buttons.grid(row=1, column=1, sticky="ns", padx=4)
        ttk.Button(pipeline_buttons, text="+ sGolay", command=self.add_savgol).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="+ ZScore", command=self.add_zscore).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="+ SNV", command=self.add_snv).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="+ MRMR", command=self.add_mrmr).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="+ PCA", command=self.add_pca).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="+ PCA LoadSel", command=self.add_pca_loadings).pack(fill="x", pady=2)
        ttk.Separator(pipeline_buttons).pack(fill="x", pady=6)
        ttk.Button(pipeline_buttons, text="Set SNN", command=self.set_snn).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="Set SAM", command=self.set_sam).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="Set SVM Linear", command=self.set_svm_linear).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="Set SVM RBF", command=self.set_svm_rbf).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="Set Random Forest", command=self.set_random_forest).pack(fill="x", pady=2)
        ttk.Separator(pipeline_buttons).pack(fill="x", pady=6)
        ttk.Button(pipeline_buttons, text="Move Up", command=lambda: self.move_block(-1)).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="Move Down", command=lambda: self.move_block(1)).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="Edit", command=self.edit_block).pack(fill="x", pady=2)
        ttk.Button(pipeline_buttons, text="Remove", command=self.remove_block).pack(fill="x", pady=2)

        # Log/actions panel
        run = ttk.LabelFrame(main, text="3) Train", padding=8)
        run.grid(row=1, column=2, sticky="nsew", padx=(6, 0))
        run.rowconfigure(2, weight=1)
        run.columnconfigure(0, weight=1)

        buttons = ttk.Frame(run)
        buttons.grid(row=0, column=0, sticky="ew")
        ttk.Button(buttons, text="Save Config", command=self.save_config).pack(side="left", padx=2)
        ttk.Button(buttons, text="Load Config", command=self.load_config).pack(side="left", padx=2)
        ttk.Button(buttons, text="Train", command=self.train, style="Primary.TButton").pack(side="left", padx=8)
        ttk.Button(buttons, text="Validate Joblib", command=self.validate_joblib).pack(side="left", padx=2)
        ttk.Button(buttons, text="Compare Models", command=self.compare_models).pack(side="left", padx=2)
        ttk.Button(buttons, text="Refresh Preview", command=self.refresh_preview).pack(side="left", padx=2)

        log_buttons = ttk.Frame(run)
        log_buttons.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(log_buttons, text="Clear Log", command=self.clear_log).pack(side="left", padx=2)
        ttk.Button(log_buttons, text="Save Log...", command=self.save_log).pack(side="left", padx=2)
        ttk.Button(log_buttons, text="Open Output Folder", command=self.open_output_folder).pack(side="left", padx=2)

        self.log = tk.Text(run, height=34, wrap="word", font=("Consolas", 9))
        self.log.grid(row=2, column=0, sticky="nsew", pady=6)

        self._build_preview_tab()

    def log_line(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def clear_log(self):
        self.log.delete("1.0", "end")

    def save_log(self):
        default = Path(self.output_var.get().strip()).with_suffix(".training_log.txt")
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=default.name,
            initialdir=str(default.parent) if str(default.parent) else None,
            filetypes=[("Text log", "*.txt"), ("All", "*.*")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.log.get("1.0", "end-1c"))
        self.log_line(f"Saved log: {path}")

    def open_output_folder(self):
        folder = Path(self.output_var.get().strip()).parent
        if not str(folder):
            folder = Path.cwd()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(folder))
            else:
                webbrowser.open(folder.resolve().as_uri())
        except Exception as exc:
            messagebox.showerror("Open folder failed", str(exc))

    def _set_preview_summary(self, text):
        if self.preview_summary is None:
            return
        self.preview_summary.configure(state="normal")
        self.preview_summary.delete("1.0", "end")
        self.preview_summary.insert("1.0", text)
        self.preview_summary.configure(state="disabled")

    def _build_preview_tab(self):
        outer = ttk.Frame(self.tab_preview, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        controls = ttk.Frame(outer)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Preview class means, preprocessing effects, selected bands, and importance.").pack(side="left", padx=(0, 10))
        ttk.Button(controls, text="Refresh Preview", command=self.refresh_preview, style="Primary.TButton").pack(side="left", padx=2)
        ttk.Radiobutton(controls, text="Processed", variable=self.preview_view_var, value="processed", command=self._redraw_preview_from_cache).pack(side="left", padx=(10, 2))
        ttk.Radiobutton(controls, text="Raw", variable=self.preview_view_var, value="raw", command=self._redraw_preview_from_cache).pack(side="left", padx=2)
        ttk.Checkbutton(controls, text="Mean ± std", variable=self.preview_std_var, command=self._redraw_preview_from_cache).pack(side="left", padx=(10, 2))
        ttk.Button(controls, text="Export Selected Bands CSV...", command=self.export_selected_bands_csv).pack(side="left", padx=(10, 2))
        ttk.Button(controls, text="Save Figure...", command=self.save_preview_figure).pack(side="left", padx=2)
        ttk.Button(controls, text="Open Pipeline Builder", command=lambda: self.notebook.select(self.tab_builder)).pack(side="left", padx=2)

        self.preview_summary = tk.Text(outer, height=8, wrap="word", font=("Consolas", 9))
        self.preview_summary.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self._set_preview_summary("No preview yet. Click 'Refresh Preview' to load the current class files, apply the current preprocessing blocks, and plot class-mean spectra.")

        if MATPLOTLIB_AVAILABLE:
            self.preview_fig = Figure(figsize=(10, 7), dpi=100)
            self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=outer)
            self.preview_canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")
            toolbar_frame = ttk.Frame(outer)
            toolbar_frame.grid(row=3, column=0, sticky="ew")
            try:
                toolbar = NavigationToolbar2Tk(self.preview_canvas, toolbar_frame)
                toolbar.update()
            except Exception:
                pass
        else:
            ttk.Label(outer, text="Matplotlib is not available. Preview text summary still works, but plots are disabled.").grid(row=2, column=0, sticky="nw")

    def _redraw_preview_from_cache(self):
        if self.preview_data is not None:
            self._display_preview(self.preview_data)

    def export_selected_bands_csv(self):
        preview = self.preview_data
        if not preview:
            messagebox.showerror("No preview", "Refresh the spectral preview first.")
            return
        selector = preview.get("selector")
        if not selector or not selector.get("selected_bands"):
            messagebox.showerror("No selected bands", "Current preprocessing has no MRMR or PCA LoadSel selected bands to export.")
            return
        default = Path(self.output_var.get().strip()).with_suffix(".selected_bands.csv")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default.name,
            initialdir=str(default.parent) if str(default.parent) else None,
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not path:
            return
        selected = selector.get("selected_bands", []) or []
        scores = selector.get("selected_scores", []) or []
        score_label = selector.get("score_label", "importance")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "band_index", score_label, "selector", "step_name"])
            for rank, band in enumerate(selected, start=1):
                score = scores[rank - 1] if rank - 1 < len(scores) else ""
                writer.writerow([rank, band, score, selector.get("type", ""), selector.get("step_name", "")])
        self.log_line(f"Exported selected bands CSV: {path}")

    def save_preview_figure(self):
        if not MATPLOTLIB_AVAILABLE or self.preview_fig is None:
            messagebox.showerror("No figure", "Matplotlib preview figure is not available.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("All", "*.*")],
            initialfile=Path(self.output_var.get().strip()).with_suffix(".preview.png").name,
        )
        if not path:
            return
        self.preview_fig.savefig(path, dpi=160, bbox_inches="tight")
        self.log_line(f"Saved preview figure: {path}")

    def refresh_preview(self):
        try:
            cfg = self.config_dict()
        except Exception as exc:
            messagebox.showerror("Invalid config", str(exc))
            return

        output_path = Path(self.output_var.get().strip())
        config_path = output_path.with_suffix(".training_config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        self.notebook.select(self.tab_preview)
        self.log_line(f"Refreshing spectral preview from config: {config_path}")
        self._set_preview_summary("Computing preview ...")
        threading.Thread(
            target=self._run_preview,
            args=(str(config_path),),
            daemon=True,
        ).start()

    def _run_preview(self, config_path):
        writer = _TkTextRedirector(self)
        try:
            script_dir = Path(__file__).resolve().parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))

            from train_nir_blocks import compute_preview

            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                preview = compute_preview(config_path)
            writer.flush()
            self.preview_data = preview
            self.root.after(0, self._display_preview, preview)
            self.root.after(0, self.log_line, "Preview finished.")
        except Exception as exc:
            writer.flush()
            tb = traceback.format_exc()
            self.root.after(0, self.log_line, "Preview failed:")
            for line in tb.rstrip().splitlines():
                self.root.after(0, self.log_line, line)
            self.root.after(0, self._set_preview_summary, f"Preview failed:\n{exc}")
            self.root.after(0, messagebox.showerror, "Preview failed", str(exc))

    def _display_preview(self, preview):
        class_names = preview.get("class_names", [])
        selector = preview.get("selector")
        pca = preview.get("pca")
        samples = preview.get("samples_per_class", {}) or {}
        view_mode = self.preview_view_var.get()
        show_std = bool(self.preview_std_var.get())

        if view_mode == "raw":
            class_means = preview.get("raw_class_means", {}) or preview.get("class_means", {}) or {}
            class_stds = preview.get("raw_class_stds", {}) or {}
            display_stage = "Raw spectra"
            display_n = int(preview.get("raw_n_bands", 0) or 0)
        else:
            class_means = preview.get("processed_class_means", {}) or preview.get("class_means", {}) or {}
            class_stds = preview.get("processed_class_stds", {}) or preview.get("class_stds", {}) or {}
            display_stage = preview.get("display_stage_label", "Processed spectra")
            display_n = int(preview.get("display_n_features", 0) or 0)

        raw_n = int(preview.get("raw_n_bands", 0) or 0)

        lines = [
            f"View: {view_mode.upper()} | Mean ± std: {'ON' if show_std else 'OFF'}",
            f"Display stage: {display_stage}",
            f"Raw bands: {raw_n}",
            f"Displayed features: {display_n}",
            "Samples per class:",
        ]
        for name in class_names:
            lines.append(f"  - {name}: {samples.get(name, 0)} spectra")

        if selector:
            lines.append(f"Selector: {selector.get('type')} ({selector.get('step_name')})")
            selected = selector.get("selected_bands", []) or []
            lines.append(f"Selected bands ({len(selected)}): {selected}")
            scores = selector.get("selected_scores", []) or []
            if scores:
                top_pairs = ", ".join(f"{b}:{s:.4f}" for b, s in list(zip(selected, scores))[:15])
                lines.append(f"Selected band {selector.get('score_label', 'importance')} (first 15): {top_pairs}")
        if pca:
            ev = pca.get("explained_variance_ratio", []) or []
            if ev:
                cum = float(np.sum(np.asarray(ev, dtype=np.float32))) * 100.0
                lines.append(f"PCA components: {pca.get('n_components_out')} | cumulative explained variance: {cum:.2f}%")

        self._set_preview_summary("\n".join(lines))

        if not MATPLOTLIB_AVAILABLE or self.preview_fig is None or self.preview_canvas is None:
            return

        self.preview_fig.clear()
        ax1 = self.preview_fig.add_subplot(2, 1, 1)
        ax2 = self.preview_fig.add_subplot(2, 1, 2)

        if class_means:
            x = np.arange(display_n, dtype=np.int32)
            mean_stack = []
            for name in class_names:
                mean = np.asarray(class_means.get(name, []), dtype=np.float32).reshape(-1)
                if mean.size == 0:
                    continue
                std = np.asarray(class_stds.get(name, []), dtype=np.float32).reshape(-1)
                mean_stack.append(mean)
                line, = ax1.plot(x[:mean.size], mean, label=name, linewidth=1.5)
                if show_std and std.size == mean.size:
                    ax1.fill_between(x[:mean.size], mean - std, mean + std, color=line.get_color(), alpha=0.15, linewidth=0)

            if selector and selector.get("selected_bands"):
                sel = np.asarray(selector.get("selected_bands", []), dtype=np.int64)
                for b in sel:
                    if 0 <= int(b) < display_n:
                        ax1.axvline(int(b), alpha=0.15, linewidth=0.8)
                if mean_stack:
                    overall = np.mean(np.vstack(mean_stack), axis=0)
                    valid = sel[(sel >= 0) & (sel < overall.shape[0])]
                    if valid.size:
                        ax1.scatter(valid, overall[valid], s=18, label="selected bands")

            ax1.set_title(f"Class mean spectra ({view_mode})")
            ax1.set_xlabel("Band / feature index")
            ax1.set_ylabel("Feature value")
            ax1.grid(True, alpha=0.25)
            ax1.legend(loc="best")
        else:
            ax1.text(0.02, 0.5, "No class means available.", transform=ax1.transAxes)
            ax1.set_axis_off()

        if selector and selector.get("full_scores"):
            full_scores = np.asarray(selector.get("full_scores", []), dtype=np.float32).reshape(-1)
            sel = np.asarray(selector.get("selected_bands", []), dtype=np.int64).reshape(-1)
            label = str(selector.get("score_label", "importance")).title()
            x2 = np.arange(full_scores.shape[0], dtype=np.int32)
            ax2.plot(x2, full_scores, linewidth=1.2)
            valid = sel[(sel >= 0) & (sel < full_scores.shape[0])]
            if valid.size:
                ax2.scatter(valid, full_scores[valid], s=18)
            ax2.set_title(f"{selector.get('type')} {label} across bands")
            ax2.set_xlabel("Band index")
            ax2.set_ylabel(label)
            ax2.grid(True, alpha=0.25)
        elif pca and pca.get("explained_variance_ratio"):
            ev = np.asarray(pca.get("explained_variance_ratio", []), dtype=np.float32).reshape(-1)
            idx = np.arange(1, ev.shape[0] + 1, dtype=np.int32)
            ax2.bar(idx, ev * 100.0)
            ax2.plot(idx, np.cumsum(ev) * 100.0, marker="o")
            ax2.set_title("PCA explained variance")
            ax2.set_xlabel("Principal component")
            ax2.set_ylabel("Variance explained (%)")
            ax2.grid(True, alpha=0.25)
        else:
            ax2.text(0.02, 0.75, "No MRMR / PCA selector in the current preprocessing blocks.", transform=ax2.transAxes)
            ax2.text(0.02, 0.55, "Add MRMR or PCA LoadSel to see selected bands and importance, or PCA to see explained variance.", transform=ax2.transAxes)
            ax2.set_axis_off()

        self.preview_fig.tight_layout()
        self.preview_canvas.draw_idle()

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("joblib", "*.joblib"), ("All", "*.*")])
        if path:
            self.output_var.set(path)

    def selected_class_index(self):
        sel = self.class_list.curselection()
        return sel[0] if sel else None

    def add_class(self):
        name = simpledialog.askstring("Class name", "Material/class name, e.g. PP")
        if name:
            self.classes.append({"name": name.strip(), "files": []})
            self._refresh_all(select_class_idx=len(self.classes) - 1)

    def remove_class(self):
        idx = self.selected_class_index()
        if idx is not None:
            del self.classes[idx]
            self._refresh_all(select_class_idx=max(0, idx - 1))

    def add_files(self):
        idx = self.selected_class_index()
        if idx is None:
            messagebox.showwarning("No class selected", "Select or add a class first.")
            return
        files = filedialog.askopenfilenames(filetypes=FILE_TYPES)
        if files:
            self.classes[idx]["files"].extend(files)
            self._refresh_all(select_class_idx=idx)

    def drop_files(self, event):
        idx = self.selected_class_index()
        if idx is None:
            messagebox.showwarning("No class selected", "Select a class before dropping files.")
            return
        files = self.root.tk.splitlist(event.data)
        self.classes[idx]["files"].extend(files)
        self._refresh_all(select_class_idx=idx)

    def remove_file(self):
        idx = self.selected_class_index()
        fsel = self.file_list.curselection()
        if idx is not None and fsel:
            del self.classes[idx]["files"][fsel[0]]
            self._refresh_all(select_class_idx=idx)

    def _refresh_class_list(self, select_idx=None):
        """Refresh class names/counts and preserve or set the selection."""
        if select_idx is None:
            current = self.selected_class_index()
        else:
            current = select_idx

        self.class_list.delete(0, "end")
        for cls in self.classes:
            self.class_list.insert("end", f"{cls['name']} ({len(cls['files'])} files)")

        if self.classes:
            if current is None:
                current = 0
            current = max(0, min(int(current), len(self.classes) - 1))
            self.class_list.selection_clear(0, "end")
            self.class_list.selection_set(current)
            self.class_list.activate(current)

    def _refresh_files(self):
        self.file_list.delete(0, "end")
        idx = self.selected_class_index()
        if idx is None:
            return
        for f in self.classes[idx]["files"]:
            self.file_list.insert("end", f)

    def _refresh_all(self, select_class_idx=None, select_pipeline_idx=None):
        self._refresh_class_list(select_class_idx)
        self._refresh_files()
        self._refresh_pipeline(select_pipeline_idx)

    def _refresh_pipeline(self, select_idx=None):
        self.pipeline_list.delete(0, "end")
        for block in self.preprocessing:
            self.pipeline_list.insert("end", self.block_label(block))
        if self.model:
            self.pipeline_list.insert("end", self.block_label(self.model))
        if select_idx is not None and self.pipeline_list.size() > 0:
            select_idx = max(0, min(int(select_idx), self.pipeline_list.size() - 1))
            self.pipeline_list.selection_clear(0, "end")
            self.pipeline_list.selection_set(select_idx)
            self.pipeline_list.activate(select_idx)

    def block_label(self, block):
        btype = block["type"]
        p = block.get("params", {})
        if btype == "SavGol":
            return f"SavGol window={p.get('window_length')} poly={p.get('polyorder')} deriv={p.get('deriv')}"
        if btype == "ZScore":
            return "ZScore / StandardScaler"
        if btype == "SNV":
            return "SNV / per-spectrum normalization"
        if btype == "MRMR":
            return f"MRMR n={p.get('n_features')} red={p.get('redundancy_weight')}"
        if btype == "PCA":
            return f"PCA projection n={p.get('n_components')} whiten={p.get('whiten')}"
        if btype == "PCA_Loadings":
            return f"PCA LoadSel n={p.get('n_features')} PCs={p.get('n_components')}"
        if btype == "SNN":
            return f"MODEL SNN hidden={p.get('hidden_layer_sizes')}"
        if btype == "SAM":
            return f"MODEL SAM temp={p.get('temperature')}"
        if btype == "SVM_Linear":
            return f"MODEL SVM Linear C={p.get('C')}"
        if btype == "SVM_RBF":
            return f"MODEL SVM RBF C={p.get('C')} gamma={p.get('gamma')}"
        if btype == "Random_Forest":
            return f"MODEL Random Forest trees={p.get('n_estimators')} max_depth={p.get('max_depth')}"
        return str(block)

    def add_savgol(self):
        # Add immediately with sensible defaults.  This avoids a modal-dialog
        # focus issue on some Windows/Tk builds where the block was not added
        # after closing the parameter dialog.  Select the new block so the user
        # can press Edit right away to change window/polyorder/derivative.
        params = {"window_length": 15, "polyorder": 2, "deriv": 1}
        self.preprocessing.append({"type": "SavGol", "params": params})
        self._refresh_pipeline(select_idx=len(self.preprocessing) - 1)
        self.log_line("Added sGolay/SavGol block. Use Edit to change window_length, polyorder, deriv.")

    def add_zscore(self):
        self.preprocessing.append({"type": "ZScore", "params": {}})
        self._refresh_pipeline(select_idx=len(self.preprocessing) - 1)

    def add_snv(self):
        self.preprocessing.append({"type": "SNV", "params": {"eps": 1e-8}})
        self._refresh_pipeline(select_idx=len(self.preprocessing) - 1)
        self.log_line("Added SNV block. SNV normalizes each spectrum individually; it is not the same as ZScore.")

    def add_mrmr(self):
        fields = {"n_features": 30, "redundancy_weight": 1.0, "max_samples_for_fit": 20000, "random_state": 42}
        params = ParamDialog(self.root, "MRMR parameters", fields).result
        if params:
            self.preprocessing.append({"type": "MRMR", "params": params})
            self._refresh_pipeline(select_idx=len(self.preprocessing) - 1)

    def add_pca(self):
        fields = {"n_components": "0.99", "whiten": "false", "random_state": 42}
        params = ParamDialog(self.root, "PCA projection parameters", fields).result
        if params:
            self.preprocessing.append({"type": "PCA", "params": params})
            self._refresh_pipeline(select_idx=len(self.preprocessing) - 1)

    def add_pca_loadings(self):
        fields = {
            "n_features": 30,
            "n_components": 5,
            "weight_by_variance": "true",
            "max_samples_for_fit": 20000,
            "random_state": 42,
        }
        params = ParamDialog(self.root, "PCA loading band selector parameters", fields).result
        if params:
            self.preprocessing.append({"type": "PCA_Loadings", "params": params})
            self._refresh_pipeline(select_idx=len(self.preprocessing) - 1)

    def set_snn(self):
        fields = {"hidden_layer_sizes": "24", "alpha": 0.0001, "learning_rate_init": 0.001, "max_iter": 500, "early_stopping": "true", "random_state": 42}
        params = ParamDialog(self.root, "SNN/MLP parameters", fields).result
        if params:
            self.model = {"type": "SNN", "params": params}
            self._refresh_pipeline(select_idx=len(self.preprocessing))

    def set_sam(self):
        fields = {"temperature": 0.05, "eps": "1e-12"}
        params = ParamDialog(self.root, "Spectral Angle Mapper parameters", fields).result
        if params:
            self.model = {"type": "SAM", "params": params}
            self._refresh_pipeline(select_idx=len(self.preprocessing))
            self.log_line("Set SAM model. It uses the mean spectrum of each class as reference spectrum.")

    def set_svm_linear(self):
        fields = {"C": 1.0, "class_weight": "balanced", "max_iter": 10000, "random_state": 42}
        params = ParamDialog(self.root, "Linear SVM parameters", fields).result
        if params:
            self.model = {"type": "SVM_Linear", "params": params}
            self._refresh_pipeline(select_idx=len(self.preprocessing))

    def set_svm_rbf(self):
        fields = {"C": 10.0, "gamma": "scale", "class_weight": "balanced", "probability": "true"}
        params = ParamDialog(self.root, "RBF SVM parameters", fields).result
        if params:
            self.model = {"type": "SVM_RBF", "params": params}
            self._refresh_pipeline(select_idx=len(self.preprocessing))

    def set_random_forest(self):
        fields = {
            "n_estimators": 300,
            "max_depth": "None",
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": 42,
        }
        params = ParamDialog(self.root, "Random Forest parameters", fields).result
        if params:
            self.model = {"type": "Random_Forest", "params": params}
            self._refresh_pipeline(select_idx=len(self.preprocessing))

    def selected_pipeline_index(self):
        sel = self.pipeline_list.curselection()
        return sel[0] if sel else None

    def move_block(self, direction):
        idx = self.selected_pipeline_index()
        if idx is None or idx >= len(self.preprocessing):
            return
        new_idx = idx + direction
        if 0 <= new_idx < len(self.preprocessing):
            self.preprocessing[idx], self.preprocessing[new_idx] = self.preprocessing[new_idx], self.preprocessing[idx]
            self._refresh_pipeline(select_idx=new_idx)

    def remove_block(self):
        idx = self.selected_pipeline_index()
        if idx is None:
            return
        if idx < len(self.preprocessing):
            del self.preprocessing[idx]
        elif self.model:
            self.model = None
        self._refresh_pipeline()

    def edit_block(self):
        idx = self.selected_pipeline_index()
        if idx is None:
            return
        block = self.preprocessing[idx] if idx < len(self.preprocessing) else self.model
        if block is None:
            return
        params = block.get("params", {})
        fields = {k: v for k, v in params.items()}
        result = ParamDialog(self.root, f"Edit {block['type']}", fields).result
        if result is not None:
            block["params"] = result
            self._refresh_pipeline()

    def config_dict(self):
        if not self.classes:
            raise ValueError("Add at least one class.")
        if any(not c["files"] for c in self.classes):
            raise ValueError("Every class must contain at least one file.")
        if not self.model:
            raise ValueError("Set one model block.")
        return {
            "output": self.output_var.get().strip(),
            "training": {
                "min_mean": float(self.min_mean_var.get()),
                "test_size": float(self.test_size_var.get()),
                "confidence_threshold": float(self.conf_var.get()),
                "random_state": int(float(self.random_state_var.get())),
            },
            "classes": self.classes,
            "pipeline": {
                "preprocessing": self.preprocessing,
                "model": self.model,
            },
        }

    def save_config(self):
        try:
            cfg = self.config_dict()
        except Exception as exc:
            messagebox.showerror("Invalid config", str(exc))
            return
        path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self.log_line(f"Saved config: {path}")

    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")])
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self.output_var.set(str(cfg.get("output", self.output_var.get())))
        training = cfg.get("training", {}) or {}
        self.min_mean_var.set(str(training.get("min_mean", self.min_mean_var.get())))
        self.test_size_var.set(str(training.get("test_size", self.test_size_var.get())))
        self.conf_var.set(str(training.get("confidence_threshold", self.conf_var.get())))
        self.random_state_var.set(str(training.get("random_state", self.random_state_var.get())))
        self.classes = cfg.get("classes", []) or []
        pipe = cfg.get("pipeline", {}) or {}
        self.preprocessing = pipe.get("preprocessing", []) or []
        self.model = pipe.get("model", None)
        self._refresh_all()
        self.log_line(f"Loaded config: {path}")

    def train(self):
        try:
            cfg = self.config_dict()
        except Exception as exc:
            messagebox.showerror("Invalid config", str(exc))
            return

        output_path = Path(self.output_var.get().strip())
        config_path = output_path.with_suffix(".training_config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        self.log_line(f"Saved training config: {config_path}")
        self.log_line("Starting training in this process; no second GUI should open.")
        threading.Thread(
            target=self._run_training_direct,
            args=(str(config_path), str(output_path)),
            daemon=True,
        ).start()

    def validate_joblib(self):
        try:
            cfg = self.config_dict()
        except Exception as exc:
            messagebox.showerror("Invalid config", str(exc))
            return

        output_path = Path(self.output_var.get().strip())
        config_path = output_path.with_suffix(".training_config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        if not output_path.exists():
            messagebox.showerror("Joblib missing", f"No model exists at:\n{output_path}\n\nTrain first or choose an existing joblib.")
            return

        self.log_line(f"Validating joblib: {output_path}")
        threading.Thread(
            target=self._run_backend_action,
            args=("validate", str(output_path), str(config_path)),
            daemon=True,
        ).start()

    def compare_models(self):
        try:
            cfg = self.config_dict()
        except Exception as exc:
            messagebox.showerror("Invalid config", str(exc))
            return

        output_path = Path(self.output_var.get().strip())
        config_path = output_path.with_suffix(".training_config.yaml")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        self.log_line(f"Comparing models from config: {config_path}")
        threading.Thread(
            target=self._run_backend_action,
            args=("compare", str(config_path), None),
            daemon=True,
        ).start()

    def _run_backend_action(self, action, path, config_path=None):
        writer = _TkTextRedirector(self)
        try:
            script_dir = Path(__file__).resolve().parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))

            from train_nir_blocks import compare_models, validate_joblib

            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                if action == "validate":
                    validate_joblib(path, config_path)
                elif action == "compare":
                    compare_models(path)
                else:
                    raise ValueError(f"Unknown backend action: {action}")
            writer.flush()
            self.root.after(0, self.log_line, f"{action.title()} finished.")
            self.root.after(0, messagebox.showinfo, f"{action.title()} complete", f"{action.title()} finished.")
        except Exception as exc:
            writer.flush()
            tb = traceback.format_exc()
            self.root.after(0, self.log_line, f"{action.title()} failed:")
            for line in tb.rstrip().splitlines():
                self.root.after(0, self.log_line, line)
            self.root.after(0, messagebox.showerror, f"{action.title()} failed", str(exc))

    def _run_training_direct(self, config_path, output_path):
        """Run the training backend by importing train_from_blocks directly.

        The previous subprocess approach could accidentally launch another GUI
        if the wrong file was resolved as the backend in Spyder/working-dir
        setups.  Importing the backend function directly makes the Train button
        deterministic: it trains and writes the selected joblib path.
        """
        writer = _TkTextRedirector(self)
        try:
            script_dir = Path(__file__).resolve().parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))

            from train_nir_blocks import train_from_blocks

            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                train_from_blocks(config_path)
            writer.flush()
            self.root.after(0, self.log_line, f"Training finished. Expected model: {output_path}")
            self.root.after(0, messagebox.showinfo, "Training complete", f"Saved model:\n{output_path}")
        except Exception as exc:
            writer.flush()
            tb = traceback.format_exc()
            self.root.after(0, self.log_line, "Training failed:")
            for line in tb.rstrip().splitlines():
                self.root.after(0, self.log_line, line)
            self.root.after(0, messagebox.showerror, "Training failed", str(exc))


def main():
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    NIRBlockTrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

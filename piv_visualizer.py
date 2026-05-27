#!/usr/bin/env python3
"""Simple Tkinter visualizer for selecting and exporting PIV vectors."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import Normalize

import extract_uvw


MAX_RENDER_VECTORS = 1500


def format_coordinate_for_filename(value):
    """Convert a numeric coordinate into a filesystem-friendly token."""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


class PIVVisualizerApp:
    """Desktop app for previewing PIV vectors and exporting point data."""

    def __init__(self, root):
        self.root = root
        self.root.title("PIV Convergence Analysis Tool")
        self.root.geometry("1320x820")
        self.root.minsize(1100, 700)

        self.folder_path = None
        self.preview_file = None
        self.vectors = []
        self.selected_record = None
        self.colorbar = None
        self.auto_color_limits = None
        self.active_color_limits = None

        self.folder_var = tk.StringVar(value="Batch folder: -")
        self.preview_var = tk.StringVar(value="Preview file: -")
        self.vector_count_var = tk.StringVar(value="Valid vectors: -")
        self.bounds_var = tk.StringVar(value="Bounds: -")
        self.selection_var = tk.StringVar(value="Selected point: click a vector in the preview")
        self.velocity_var = tk.StringVar(value="Velocity: -")
        self.magnitude_var = tk.StringVar(value="Magnitude: -")
        self.color_min_var = tk.StringVar(value="")
        self.color_max_var = tk.StringVar(value="")
        self.color_range_var = tk.StringVar(value="Active scale: -")
        self.color_data_range_var = tk.StringVar(value="Data range: -")
        self.status_var = tk.StringVar(
            value="Choose a .dat file to preview, then click a vector for batch export."
        )

        self._build_layout()
        self._update_export_button_state()

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=4)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(1, weight=1)

        controls = ttk.Frame(container)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        controls.columnconfigure(1, weight=1)

        browse_button = ttk.Button(controls, text="Browse .dat File", command=self.browse_dat_file)
        browse_button.grid(row=0, column=0, sticky="w")

        folder_label = ttk.Label(controls, textvariable=self.folder_var)
        folder_label.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        plot_frame = ttk.Frame(container)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(9, 6), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title("Vector field preview")
        self.axes.set_xlabel("x [mm]")
        self.axes.set_ylabel("y [mm]")

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        details = ttk.LabelFrame(container, text="Selection", padding=12)
        details.grid(row=1, column=1, sticky="nsew")
        details.columnconfigure(1, weight=1)

        self._add_detail_row(details, 0, "Preview", self.preview_var)
        self._add_detail_row(details, 1, "Vectors", self.vector_count_var)
        self._add_detail_row(details, 2, "Bounds", self.bounds_var)
        self._add_detail_row(details, 3, "Point", self.selection_var)
        self._add_detail_row(details, 4, "Velocity", self.velocity_var)
        self._add_detail_row(details, 5, "Magnitude", self.magnitude_var)

        colour_frame = ttk.LabelFrame(details, text="Colour Scale", padding=8)
        colour_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        colour_frame.columnconfigure(1, weight=1)
        colour_frame.columnconfigure(3, weight=1)

        ttk.Label(colour_frame, text="Min").grid(row=0, column=0, sticky="w")
        colour_min_entry = ttk.Entry(colour_frame, textvariable=self.color_min_var, width=12)
        colour_min_entry.grid(row=0, column=1, sticky="ew", padx=(8, 12))
        colour_min_entry.bind("<Return>", self._apply_color_limits_from_event)

        ttk.Label(colour_frame, text="Max").grid(row=0, column=2, sticky="w")
        colour_max_entry = ttk.Entry(colour_frame, textvariable=self.color_max_var, width=12)
        colour_max_entry.grid(row=0, column=3, sticky="ew", padx=(8, 0))
        colour_max_entry.bind("<Return>", self._apply_color_limits_from_event)

        ttk.Button(colour_frame, text="Apply", command=self.apply_color_limits).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
        )
        ttk.Button(colour_frame, text="Auto", command=self.reset_color_limits).grid(
            row=1,
            column=2,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
            padx=(8, 0),
        )

        ttk.Label(
            colour_frame,
            textvariable=self.color_data_range_var,
            wraplength=260,
            justify="left",
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Label(
            colour_frame,
            textvariable=self.color_range_var,
            wraplength=260,
            justify="left",
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.export_button = ttk.Button(
            details,
            text="Extract And Export CSV",
            command=self.extract_and_export,
        )
        self.export_button.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(16, 0))

        status = ttk.Label(container, textvariable=self.status_var)
        status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))

    @staticmethod
    def _add_detail_row(parent, row_index, label_text, value_var):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row_index, column=0, sticky="nw", pady=(0, 10))
        value = ttk.Label(parent, textvariable=value_var, wraplength=260, justify="left")
        value.grid(row=row_index, column=1, sticky="nw", pady=(0, 10), padx=(12, 0))

    def browse_dat_file(self):
        selected_file = filedialog.askopenfilename(
            title="Select a .dat file to preview",
            filetypes=[("DAT files", "*.dat")],
        )

        if not selected_file:
            return

        self.load_preview_source(Path(selected_file))

    def load_preview_source(self, source_path):
        try:
            source_info = extract_uvw.resolve_preview_source(source_path)
            folder = source_info["folder"]
            preview_file = source_info["preview_file"]
            dat_files = source_info["dat_files"]
            vectors = extract_uvw.load_valid_vectors(preview_file)
        except Exception as exc:
            messagebox.showerror("Unable to load preview file", str(exc))
            self._set_status(str(exc))
            return

        if not vectors:
            message = f"No valid vectors were found in {preview_file.name}."
            messagebox.showerror("No valid vectors", message)
            self._set_status(message)
            return

        self.folder_path = Path(folder)
        self.preview_file = preview_file
        self.vectors = vectors
        self.selected_record = None

        self.folder_var.set(f"Batch folder: {self.folder_path}")
        self.preview_var.set(f"{preview_file.name} ({len(dat_files)} .dat files in batch)")
        self.vector_count_var.set(
            f"{len(vectors)} valid vectors, displaying {len(self._get_display_vectors())}"
        )

        bounds = extract_uvw.get_valid_coordinate_bounds(preview_file)
        if bounds is None:
            self.bounds_var.set("Bounds: unavailable")
        else:
            min_x, max_x, min_y, max_y = bounds
            self.bounds_var.set(
                f"x={min_x:.3f} to {max_x:.3f}, y={min_y:.3f} to {max_y:.3f}"
            )

        self._reset_color_limits(re_render=False)
        self._reset_selection_labels()
        self._render_plot()
        self._update_export_button_state()
        self._set_status(
            f"Loaded {preview_file.name}. Click a vector in the plot to choose the batch export point."
        )

    def _get_display_vectors(self):
        if len(self.vectors) <= MAX_RENDER_VECTORS:
            return self.vectors

        step = max(1, len(self.vectors) // MAX_RENDER_VECTORS)
        return self.vectors[::step]

    def _render_plot(self):
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        self.axes.clear()
        self.axes.set_title("Vector field preview")
        self.axes.set_xlabel("x [mm]")
        self.axes.set_ylabel("y [mm]")

        display_vectors = self._get_display_vectors()
        x_values = [vector["x"] for vector in display_vectors]
        y_values = [vector["y"] for vector in display_vectors]
        u_values = [vector["u"] for vector in display_vectors]
        v_values = [vector["v"] for vector in display_vectors]
        velmag_values = [vector["velmag"] for vector in display_vectors]
        norm = self._build_color_normalization()

        quiver = self.axes.quiver(
            x_values,
            y_values,
            u_values,
            v_values,
            velmag_values,
            cmap="rainbow",
            norm=norm,
            pivot="mid",
        )
        self.colorbar = self.figure.colorbar(quiver, ax=self.axes)
        self.colorbar.set_label("Velocity magnitude [m/s]")
        self.axes.set_aspect("equal", adjustable="box")

        if self.selected_record is not None:
            self.axes.scatter(
                [self.selected_record["x"]],
                [self.selected_record["y"]],
                s=110,
                facecolors="none",
                edgecolors="black",
                linewidths=1.8,
                zorder=5,
            )
            self.axes.scatter(
                [self.selected_record["x"]],
                [self.selected_record["y"]],
                s=30,
                c="white",
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
            )

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _build_color_normalization(self):
        if self.active_color_limits is None:
            return None

        color_min, color_max = self.active_color_limits
        return Normalize(vmin=color_min, vmax=color_max, clip=True)

    def _on_plot_click(self, event):
        if event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            return

        if not self.vectors:
            return

        result = extract_uvw.extract_uvw_from_vectors(
            self.vectors,
            target_x=event.xdata,
            target_y=event.ydata,
            tol=0.0,
            fallback="nearest",
            max_distance=None,
        )
        record = result["record"]

        if record is None:
            self._set_status("No valid vector could be selected from this click.")
            return

        self.selected_record = record
        self.selection_var.set(
            f"x={record['x']:.3f}, y={record['y']:.3f} (click distance {result['distance']:.3f})"
        )
        self.velocity_var.set(
            f"u={record['u']:.5f}, v={record['v']:.5f}, w={record['w']:.5f}"
        )
        self.magnitude_var.set(f"velmag={record['velmag']:.5f} m/s")
        self._render_plot()
        self._update_export_button_state()
        self._set_status(
            f"Selected vector at x={record['x']:.3f}, y={record['y']:.3f}."
        )

    def extract_and_export(self):
        if self.folder_path is None or self.selected_record is None:
            messagebox.showwarning(
                "Selection required",
                "Choose a .dat preview file and click a vector before exporting.",
            )
            return

        initial_name = (
            "uvw_"
            f"{format_coordinate_for_filename(self.selected_record['x'])}_"
            f"{format_coordinate_for_filename(self.selected_record['y'])}.csv"
        )
        output_path = filedialog.asksaveasfilename(
            title="Save extracted CSV",
            initialdir=str(self.folder_path),
            initialfile=initial_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )

        if not output_path:
            return

        try:
            summary = extract_uvw.run_extraction(
                self.folder_path,
                target_x=self.selected_record["x"],
                target_y=self.selected_record["y"],
                output_csv=Path(output_path),
            )
        except Exception as exc:
            messagebox.showerror("Extraction failed", str(exc))
            self._set_status(str(exc))
            return

        message = (
            f"Processed {len(summary['dat_files'])} .dat files.\n"
            f"Exact matches: {summary['exact_count']}\n"
            f"Nearest matches: {summary['nearest_count']}\n"
            f"Missing matches: {summary['missing_count']}\n"
            f"CSV written to: {summary['output_csv']}"
        )
        messagebox.showinfo("Extraction complete", message)
        self._set_status(f"CSV written to {summary['output_csv']}")

    def _reset_selection_labels(self):
        self.selection_var.set("Selected point: click a vector in the preview")
        self.velocity_var.set("Velocity: -")
        self.magnitude_var.set("Magnitude: -")

    def _apply_color_limits_from_event(self, _event):
        self.apply_color_limits()

    def _calculate_auto_color_limits(self):
        if not self.vectors:
            return None

        color_min = min(vector["velmag"] for vector in self.vectors)
        color_max = max(vector["velmag"] for vector in self.vectors)

        if color_max <= color_min:
            color_max = color_min + 1e-9

        return color_min, color_max

    def _reset_color_limits(self, re_render=True):
        self.auto_color_limits = self._calculate_auto_color_limits()

        if self.auto_color_limits is None:
            self.active_color_limits = None
            self.color_min_var.set("")
            self.color_max_var.set("")
            self.color_data_range_var.set("Data range: -")
            self.color_range_var.set("Active scale: -")
            return

        color_min, color_max = self.auto_color_limits
        self.active_color_limits = (color_min, color_max)
        self.color_min_var.set(f"{color_min:.5f}")
        self.color_max_var.set(f"{color_max:.5f}")
        self.color_data_range_var.set(
            f"Data range: {color_min:.5f} to {color_max:.5f}"
        )
        self.color_range_var.set(
            f"Active scale: auto ({color_min:.5f} to {color_max:.5f})"
        )

        if re_render and self.vectors:
            self._render_plot()

    def reset_color_limits(self):
        if not self.vectors:
            self._set_status("Load a .dat preview file before adjusting the colour scale.")
            return

        self._reset_color_limits(re_render=True)
        color_min, color_max = self.active_color_limits
        self._set_status(
            f"Colour scale reset to auto range {color_min:.5f} to {color_max:.5f}."
        )

    def apply_color_limits(self):
        if not self.vectors:
            self._set_status("Load a .dat preview file before adjusting the colour scale.")
            return

        try:
            color_min = float(self.color_min_var.get())
            color_max = float(self.color_max_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid colour scale",
                "Enter numeric values for both minimum and maximum colour limits.",
            )
            return

        if color_max <= color_min:
            messagebox.showerror(
                "Invalid colour scale",
                "The maximum colour limit must be greater than the minimum.",
            )
            return

        self.active_color_limits = (color_min, color_max)
        self.color_range_var.set(
            f"Active scale: manual ({color_min:.5f} to {color_max:.5f})"
        )
        self._render_plot()
        self._set_status(
            "Manual colour scale applied. Values outside the selected range are saturated."
        )

    def _update_export_button_state(self):
        if self.folder_path is None or self.selected_record is None:
            self.export_button.state(["disabled"])
            return

        self.export_button.state(["!disabled"])

    def _set_status(self, message):
        self.status_var.set(message)


def main(initial_folder=None):
    root = tk.Tk()
    app = PIVVisualizerApp(root)

    if initial_folder is not None:
        app.load_preview_source(initial_folder)

    root.mainloop()


if __name__ == "__main__":
    main()
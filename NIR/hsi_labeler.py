
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QLineEdit, QLabel, QMessageBox, QComboBox,
    QCheckBox, QSpinBox, QSizePolicy, QScrollArea, QFrame
)
from PySide6.QtGui import QColor, QBrush
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse


DEFAULT_CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# The left control panel is intentionally width-limited so long file names,
# cube shapes, or loaded image metadata cannot resize the buttons/controls and
# steal space from the spectral preview on a 1920 x 1080 display.
CONTROL_PANEL_WIDTH = 300
CONTROL_WIDGET_WIDTH = 272


def default_class_color(index):
    return DEFAULT_CLASS_COLORS[int(index) % len(DEFAULT_CLASS_COLORS)]


def normalise_hex_color(value, fallback="#1f77b4"):
    try:
        text = str(value).strip()
        if not text:
            return fallback
        if not text.startswith("#"):
            text = "#" + text
        if len(text) != 7:
            return fallback
        int(text[1:], 16)
        return text.lower()
    except Exception:
        return fallback


def load_hsi_file(path):
    """
    Load an HSI cube.

    .npy files are memory-mapped, so huge raw recordings are not fully loaded
    into RAM. .npz and .mat files still need normal loading because they are
    compressed/container formats.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        cube = np.load(path, mmap_mode="r")
    elif suffix == ".npz":
        data = np.load(path)
        key = list(data.keys())[0]
        cube = data[key]
    elif suffix == ".mat":
        mat = loadmat(path)
        candidates = [
            v for k, v in mat.items()
            if not k.startswith("__") and isinstance(v, np.ndarray) and v.ndim >= 3
        ]
        if not candidates:
            raise ValueError("No 3D array found in .mat file.")
        cube = candidates[0]
    else:
        raise ValueError("Unsupported file type.")

    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got shape {cube.shape}")

    return cube


def spectral_axis_for(cube):
    for i, s in enumerate(cube.shape):
        if s in (212, 220):
            return i
    return 1


def cube_to_yx_lambda(cube):
    spec_ax = spectral_axis_for(cube)
    return np.moveaxis(cube, spec_ax, -1)


def grayscale_image(cube, max_display_pixels=1_500_000):
    """
    Create a mean-over-spectrum grayscale image.

    For large cubes, this returns a downsampled preview and the y/x scale
    factors. All annotations/export still use original cube coordinates.
    """
    cube2 = cube_to_yx_lambda(cube)
    h, w, bands = cube2.shape

    total_pixels = h * w
    if total_pixels > max_display_pixels:
        scale = int(np.ceil(np.sqrt(total_pixels / max_display_pixels)))
        scale = max(1, scale)
    else:
        scale = 1

    view = cube2[::scale, ::scale, :]
    img = np.nanmean(view, axis=-1, dtype=np.float32)

    minv = np.nanmin(img)
    maxv = np.nanmax(img)
    img = img - minv
    denom = maxv - minv
    if denom > 0:
        img = img / denom

    return img.astype(np.float32, copy=False), scale, scale


def rotate_image_for_display(img, rotation_deg):
    k = (rotation_deg // 90) % 4
    return np.rot90(img, k=k)


def display_to_original_yx(display_y, display_x, original_h, original_w, rotation_deg):
    """
    Map coordinates clicked in the rotated display image back to original cube coordinates.

    rotation_deg is interpreted like np.rot90(img, k=rotation_deg/90):
    0: original
    90: counter-clockwise
    180
    270: counter-clockwise, equivalent to clockwise 90
    """
    rot = rotation_deg % 360

    yd = int(display_y)
    xd = int(display_x)

    if rot == 0:
        yo, xo = yd, xd
    elif rot == 90:
        yo = xd
        xo = original_w - 1 - yd
    elif rot == 180:
        yo = original_h - 1 - yd
        xo = original_w - 1 - xd
    elif rot == 270:
        yo = original_h - 1 - xd
        xo = yd
    else:
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees.")

    return int(yo), int(xo)


def original_to_display_yx(original_y, original_x, original_h, original_w, rotation_deg):
    """
    Map original cube coordinates to current rotated display coordinates.
    """
    rot = rotation_deg % 360

    yo = int(original_y)
    xo = int(original_x)

    if rot == 0:
        yd, xd = yo, xo
    elif rot == 90:
        yd = original_w - 1 - xo
        xd = yo
    elif rot == 180:
        yd = original_h - 1 - yo
        xd = original_w - 1 - xo
    elif rot == 270:
        yd = xo
        xd = original_h - 1 - yo
    else:
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees.")

    return int(yd), int(xd)


def preview_shape_from_original(original_h, original_w, sy, sx):
    """Return the unrotated downsampled preview shape for an original image."""
    base_h = int(np.ceil(original_h / sy))
    base_w = int(np.ceil(original_w / sx))
    return base_h, base_w


def base_preview_to_display_yx(base_y, base_x, base_h, base_w, rotation_deg):
    """Map unrotated preview coordinates to rotated display coordinates."""
    return original_to_display_yx(base_y, base_x, base_h, base_w, rotation_deg)


def display_to_base_preview_yx(display_y, display_x, base_h, base_w, rotation_deg):
    """Map rotated display coordinates back to unrotated preview coordinates."""
    return display_to_original_yx(display_y, display_x, base_h, base_w, rotation_deg)


def make_window_odd(value):
    value = int(value)
    if value < 3:
        value = 3
    if value % 2 == 0:
        value += 1
    return value


def preprocess_spectra(
    spectra,
    smoothing=False,
    derivative=False,
    zscore=False,
    window=11,
    polyorder=2
):
    if spectra.size == 0:
        return spectra

    y = np.asarray(spectra, dtype=np.float32).copy()

    window = make_window_odd(window)
    if window >= y.shape[1]:
        window = y.shape[1] - 1 if y.shape[1] % 2 == 0 else y.shape[1]
    window = max(3, window)

    polyorder = min(polyorder, window - 1)

    if smoothing:
        y = savgol_filter(y, window_length=window, polyorder=polyorder, deriv=0, axis=1)

    if derivative:
        y = savgol_filter(y, window_length=window, polyorder=polyorder, deriv=1, axis=1)

    if zscore:
        mu = np.nanmean(y, axis=1, keepdims=True)
        sigma = np.nanstd(y, axis=1, keepdims=True)
        sigma[sigma == 0] = 1
        y = (y - mu) / sigma

    return y




def yaml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "/")
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]", ","]) or text.strip() != text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def write_export_settings_yaml(path, settings):
    lines = []

    def write_dict(d, indent=0):
        for key, value in d.items():
            prefix = "  " * indent + f"{key}:"
            if isinstance(value, dict):
                lines.append(prefix)
                write_dict(value, indent + 1)
            elif isinstance(value, list):
                lines.append(prefix)
                for item in value:
                    if isinstance(item, dict):
                        lines.append("  " * (indent + 1) + "-")
                        write_dict(item, indent + 2)
                    else:
                        lines.append("  " * (indent + 1) + f"- {yaml_scalar(item)}")
            else:
                lines.append(prefix + f" {yaml_scalar(value)}")

    write_dict(settings)
    Path(path).write_text("\\n".join(lines) + "\\n", encoding="utf-8")


def spectra_from_pixels(cube, pixels):
    cube2 = cube_to_yx_lambda(cube)
    h, w = cube2.shape[:2]

    rows = []
    valid_pixels = []

    for y, x in pixels:
        if 0 <= y < h and 0 <= x < w:
            rows.append(cube2[y, x, :])
            valid_pixels.append((y, x))

    if not rows:
        return np.empty((0, cube2.shape[-1])), []

    return np.vstack(rows), valid_pixels


class HSIAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIR HSI Pixel Class Labeler")
        self.resize(1500, 900)
        self.setMinimumSize(1180, 720)

        self.files = []
        self.cubes = []
        self.index = 0

        self.classes = {}
        self.selections = {}
        self.rotations = {}

        # Large-cube display cache. Selections stay in original cube coordinates;
        # only the mean preview image is downsampled.
        self.display_image_cache = {}
        self.display_scales = {}
        self.max_display_pixels = 1_500_000

        self.mode = "pixel"
        self.action = "add"
        self.drag_start = None
        self.temp_patch = None
        self.hover_line = None
        self._hover_counter = 0
        self.hover_update_every = 4
        self.last_image_key = None
        self.last_image_rotation = None
        self.image_views = {}
        self._restoring_view = False

        self.image_fig = Figure(figsize=(7.0, 3.6), constrained_layout=True)
        self.image_canvas = FigureCanvas(self.image_fig)
        self.image_toolbar = NavigationToolbar(self.image_canvas, self)
        self.ax = self.image_fig.add_subplot(111)

        self.spectrum_fig = Figure(figsize=(7.0, 2.25), constrained_layout=True)
        self.spectrum_canvas = FigureCanvas(self.spectrum_fig)
        self.spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, self)
        self.spectrum_ax = self.spectrum_fig.add_subplot(111)

        self.project_fig = Figure(figsize=(7.0, 2.25), constrained_layout=True)
        self.project_canvas = FigureCanvas(self.project_fig)
        self.project_toolbar = NavigationToolbar(self.project_canvas, self)
        self.project_ax = self.project_fig.add_subplot(111)

        self.image_canvas.mpl_connect("button_press_event", self.on_press)
        self.image_canvas.mpl_connect("button_release_event", self.on_release)
        self.image_canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.image_canvas.mpl_connect("scroll_event", self.on_scroll_zoom)
        self.ax.callbacks.connect("xlim_changed", self.remember_image_view)
        self.ax.callbacks.connect("ylim_changed", self.remember_image_view)

        for canvas in [self.image_canvas, self.spectrum_canvas, self.project_canvas]:
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Keep enough vertical room for useful spectra while still fitting below
        # the window decoration/taskbar on 1920 x 1080 screens.
        self.image_canvas.setMinimumHeight(285)
        self.spectrum_canvas.setMinimumHeight(170)
        self.project_canvas.setMinimumHeight(170)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(6, 6, 6, 6)
        left.setSpacing(4)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(3)

        self.load_btn = QPushButton("Load HSI files")
        self.prev_btn = QPushButton("← Previous cube")
        self.next_btn = QPushButton("Next cube →")
        self.remove_current_btn = QPushButton("Remove current HSI image")

        self.file_label = QLabel("No file loaded")
        self.shape_label = QLabel("Shape: -")
        self.rotation_label = QLabel("Rotation: 0°")

        self.rotate_left_btn = QPushButton("Rotate image 90° left")
        self.rotate_right_btn = QPushButton("Rotate image 90° right")
        self.reset_rotation_btn = QPushButton("Reset rotation")
        self.full_view_btn = QPushButton("Fit full image")

        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Class name, e.g. PE, PP, PET")

        self.add_class_btn = QPushButton("Add class")
        self.delete_class_btn = QPushButton("Delete selected class")
        self.class_color_btn = QPushButton("Choose class colour")
        self.class_list = QListWidget()

        self.mode_box = QComboBox()
        self.mode_box.addItems(["pixel", "rectangle", "ellipse"])

        self.action_box = QComboBox()
        self.action_box.addItems(["add spectra", "remove spectra"])

        self.smooth_check = QCheckBox("Savitzky-Golay smoothing")
        self.deriv_check = QCheckBox("Savitzky-Golay 1st derivative")
        self.zscore_check = QCheckBox("Z-score normalisation")

        self.window_spin = QSpinBox()
        self.window_spin.setRange(3, 99)
        self.window_spin.setValue(11)
        self.window_spin.setSingleStep(2)

        self.poly_spin = QSpinBox()
        self.poly_spin.setRange(1, 5)
        self.poly_spin.setValue(2)

        self.export_kind_box = QComboBox()
        self.export_kind_box.addItems(["raw spectra", "preprocessed spectra"])

        self.export_btn = QPushButton("Export selected spectra to XLSX")
        self.project_view_btn = QPushButton("Update PROJECT SPECTRAL VIEW")

        self.count_label = QLabel("No class selected")
        self.hover_label = QLabel("Hover: -")

        self.configure_compact_controls()

        left.addWidget(self.load_btn)
        left.addWidget(self.prev_btn)
        left.addWidget(self.next_btn)
        left.addWidget(self.remove_current_btn)
        left.addSpacing(10)
        left.addWidget(self.file_label)
        left.addWidget(self.shape_label)
        left.addWidget(self.rotation_label)
        left.addWidget(self.rotate_left_btn)
        left.addWidget(self.rotate_right_btn)
        left.addWidget(self.reset_rotation_btn)
        left.addWidget(self.full_view_btn)
        left.addSpacing(15)

        left.addWidget(QLabel("Classes"))
        left.addWidget(self.class_input)
        left.addWidget(self.add_class_btn)
        left.addWidget(self.delete_class_btn)
        left.addWidget(self.class_color_btn)
        left.addWidget(self.class_list)
        left.addSpacing(15)

        left.addWidget(QLabel("Selection shape"))
        left.addWidget(self.mode_box)

        left.addWidget(QLabel("Selection action"))
        left.addWidget(self.action_box)
        left.addSpacing(15)

        left.addWidget(QLabel("Spectrum preprocessing preview/export"))
        left.addWidget(self.smooth_check)
        left.addWidget(self.deriv_check)
        left.addWidget(self.zscore_check)
        left.addWidget(QLabel("SavGol window length"))
        left.addWidget(self.window_spin)
        left.addWidget(QLabel("SavGol polynomial order"))
        left.addWidget(self.poly_spin)
        left.addSpacing(10)

        left.addWidget(QLabel("Export mode"))
        left.addWidget(self.export_kind_box)
        left.addWidget(self.export_btn)
        left.addWidget(self.project_view_btn)
        left.addSpacing(10)

        left.addWidget(self.count_label)
        left.addWidget(self.hover_label)
        left.addStretch()

        right.addWidget(self.image_toolbar)
        right.addWidget(self.image_canvas, 3)
        right.addWidget(self.spectrum_toolbar)
        right.addWidget(self.spectrum_canvas, 2)
        right.addWidget(QLabel("PROJECT SPECTRAL VIEW"))
        right.addWidget(self.project_toolbar)
        right.addWidget(self.project_canvas, 2)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_widget)
        left_scroll.setFixedWidth(CONTROL_PANEL_WIDTH)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        right_panel = QWidget()
        right_panel.setLayout(right)
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(left_scroll)
        layout.addWidget(right_panel, 1)

        self.setCentralWidget(root)

        self.load_btn.clicked.connect(self.load_files)
        self.prev_btn.clicked.connect(self.prev_file)
        self.next_btn.clicked.connect(self.next_file)
        self.remove_current_btn.clicked.connect(self.remove_current_file)
        self.rotate_left_btn.clicked.connect(lambda: self.rotate_current(90))
        self.rotate_right_btn.clicked.connect(lambda: self.rotate_current(-90))
        self.reset_rotation_btn.clicked.connect(self.reset_current_rotation)
        self.full_view_btn.clicked.connect(self.fit_full_image)
        self.add_class_btn.clicked.connect(self.add_class)
        self.class_input.returnPressed.connect(self.add_class)
        self.delete_class_btn.clicked.connect(self.delete_current_class)
        self.class_color_btn.clicked.connect(self.choose_class_color)
        self.export_btn.clicked.connect(self.export_xlsx)
        self.project_view_btn.clicked.connect(self.refresh_project_spectral_view)
        self.mode_box.currentTextChanged.connect(self.set_mode)
        self.action_box.currentTextChanged.connect(self.set_action)
        self.class_list.currentTextChanged.connect(self.refresh_display)

        for widget in [
            self.smooth_check, self.deriv_check, self.zscore_check,
            self.window_spin, self.poly_spin
        ]:
            if hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self.refresh_spectrum_plot)
                widget.stateChanged.connect(self.refresh_project_spectral_view)
            else:
                widget.valueChanged.connect(self.refresh_spectrum_plot)
                widget.valueChanged.connect(self.refresh_project_spectral_view)

    def configure_compact_controls(self):
        """Make the control column stable and usable on 1920 x 1080 screens.

        The important part is the horizontal size policy: file names and cube
        shapes loaded later must not change button/control width or shrink the
        matplotlib preview area.
        """
        buttons = [
            self.load_btn, self.prev_btn, self.next_btn, self.remove_current_btn,
            self.rotate_left_btn, self.rotate_right_btn, self.reset_rotation_btn,
            self.full_view_btn, self.add_class_btn, self.delete_class_btn,
            self.class_color_btn, self.export_btn, self.project_view_btn,
        ]

        for button in buttons:
            button.setFixedWidth(CONTROL_WIDGET_WIDTH)
            button.setMinimumHeight(25)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        for widget in [
            self.class_input, self.class_list, self.mode_box, self.action_box,
            self.window_spin, self.poly_spin, self.export_kind_box,
        ]:
            widget.setFixedWidth(CONTROL_WIDGET_WIDTH)
            widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.class_list.setMinimumHeight(90)
        self.class_list.setMaximumHeight(155)

        for label in [
            self.file_label, self.shape_label, self.rotation_label,
            self.count_label, self.hover_label,
        ]:
            label.setWordWrap(True)
            label.setFixedWidth(CONTROL_WIDGET_WIDTH)
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        for checkbox in [self.smooth_check, self.deriv_check, self.zscore_check]:
            checkbox.setFixedWidth(CONTROL_WIDGET_WIDTH)
            checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def current_class(self):
        item = self.class_list.currentItem()
        return item.text() if item else None

    def current_key(self):
        if not self.files:
            return None
        return str(self.files[self.index])

    def current_rotation(self):
        key = self.current_key()
        if key is None:
            return 0
        return self.rotations.get(key, 0)

    def preprocessing_settings(self):
        return {
            "smoothing": self.smooth_check.isChecked(),
            "derivative": self.deriv_check.isChecked(),
            "zscore": self.zscore_check.isChecked(),
            "window": self.window_spin.value(),
            "polyorder": self.poly_spin.value()
        }

    def class_color(self, cls):
        value = self.classes.get(cls)

        if isinstance(value, dict):
            color = value.get("color")
        else:
            try:
                idx = list(self.classes.keys()).index(cls)
            except ValueError:
                idx = 0
            color = default_class_color(idx)

        return normalise_hex_color(color, default_class_color(0))

    def set_class_color(self, cls, color):
        color = normalise_hex_color(color, self.class_color(cls))
        current = self.classes.get(cls, True)

        if isinstance(current, dict):
            current["color"] = color
            self.classes[cls] = current
        else:
            self.classes[cls] = {"enabled": bool(current), "color": color}

        self.update_class_list_colours()
        self.refresh_display()
        self.refresh_project_spectral_view()

    def update_class_list_colours(self):
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            cls = item.text()
            color = self.class_color(cls)
            item.setForeground(QBrush(QColor(color)))
            item.setToolTip(f"{cls} colour: {color}")

    def set_mode(self, mode):
        self.mode = mode

    def set_action(self, text):
        self.action = "remove" if "remove" in text else "add"

    def rotate_current(self, delta_deg):
        key = self.current_key()
        if key is None:
            return
        self.rotations[key] = (self.rotations.get(key, 0) + delta_deg) % 360
        self.last_image_rotation = None
        self.fit_full_image()
        self.refresh_display()

    def reset_current_rotation(self):
        key = self.current_key()
        if key is None:
            return
        self.rotations[key] = 0
        self.last_image_rotation = None
        self.refresh_display()

    def load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load HSI cubes",
            "",
            "HSI files (*.mat *.npy *.npz)"
        )

        if not paths:
            return

        for p in paths:
            try:
                path = Path(p)
                cube = load_hsi_file(path)
                self.files.append(path)
                self.cubes.append(cube)
                self.selections[str(path)] = {}
                self.rotations[str(path)] = 0
            except Exception as e:
                QMessageBox.warning(self, "Load error", f"{p}\n\n{e}")

        self.index = 0
        self.display_image_cache.clear()
        self.display_scales.clear()
        self.image_views.clear()
        self.refresh_display()

    def current_image_selected_pixel_count(self):
        key = self.current_key()
        if key is None:
            return 0

        return sum(
            len(pixels)
            for pixels in self.selections.get(key, {}).values()
        )

    def remove_current_file(self):
        if not self.files:
            QMessageBox.information(self, "No HSI image loaded", "There is no HSI image to remove.")
            return

        key = self.current_key()
        path = self.files[self.index]
        selected_count = self.current_image_selected_pixel_count()

        if selected_count > 0:
            reply = QMessageBox.warning(
                self,
                "Remove HSI image?",
                (
                    f"Remove the currently displayed HSI image from the loaded list?\n\n"
                    f"{path.name}\n\n"
                    f"This will also remove {selected_count} selected spectra/pixels "
                    f"from this image. The file on disk will not be deleted."
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply != QMessageBox.Yes:
                return

        if self.temp_patch:
            self.temp_patch.remove()
            self.temp_patch = None
        self.drag_start = None

        self.files.pop(self.index)
        self.cubes.pop(self.index)

        self.selections.pop(key, None)
        self.rotations.pop(key, None)
        self.display_image_cache.pop(key, None)
        self.display_scales.pop(key, None)
        self.image_views = {
            view_key: view
            for view_key, view in self.image_views.items()
            if view_key[0] != key
        }

        if self.files:
            self.index = min(self.index, len(self.files) - 1)
            self.last_image_key = None
            self.last_image_rotation = None
        else:
            self.index = 0
            self.last_image_key = None
            self.last_image_rotation = None

        self.fit_full_image()
        self.refresh_display()
        self.refresh_project_spectral_view()

    def add_class(self):
        name = self.class_input.text().strip()
        if not name:
            return

        if name not in self.classes:
            self.classes[name] = {
                "enabled": True,
                "color": default_class_color(len(self.classes)),
            }

        existing = [
            self.class_list.item(i).text()
            for i in range(self.class_list.count())
        ]

        class_item = None

        if name not in existing:
            class_item = QListWidgetItem(name)
            self.class_list.addItem(class_item)
        else:
            for i in range(self.class_list.count()):
                item = self.class_list.item(i)
                if item.text() == name:
                    class_item = item
                    break

        if class_item is not None:
            self.class_list.setCurrentItem(class_item)

        for key in self.selections:
            self.selections[key].setdefault(name, set())

        self.update_class_list_colours()
        self.class_input.clear()
        self.refresh_display()
        self.refresh_project_spectral_view()

    def delete_current_class(self):
        cls = self.current_class()
        if cls is None:
            QMessageBox.information(self, "No class selected", "Select a class to delete first.")
            return

        reply = QMessageBox.question(
            self,
            "Delete class",
            f"Delete class '{cls}' and all selected pixels assigned to it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        if cls in self.classes:
            del self.classes[cls]

        for key in self.selections:
            self.selections[key].pop(cls, None)

        for i in range(self.class_list.count()):
            if self.class_list.item(i).text() == cls:
                self.class_list.takeItem(i)
                break

        self.refresh_display()
        self.refresh_project_spectral_view()

    def choose_class_color(self):
        cls = self.current_class()
        if cls is None:
            QMessageBox.information(self, "No class selected", "Select a class first.")
            return

        try:
            import tkinter as tk
            from tkinter.colorchooser import askcolor

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            _, hex_color = askcolor(
                color=self.class_color(cls),
                title=f"Choose colour for class {cls}",
                parent=root,
            )
            root.destroy()

            if hex_color:
                self.set_class_color(cls, hex_color)

        except Exception as exc:
            QMessageBox.warning(self, "Colour selector error", str(exc))

    def prev_file(self):
        if not self.files:
            return
        self.index = (self.index - 1) % len(self.files)
        self.last_image_key = None
        self.fit_full_image()
        self.refresh_display()

    def next_file(self):
        if not self.files:
            return
        self.index = (self.index + 1) % len(self.files)
        self.last_image_key = None
        self.fit_full_image()
        self.refresh_display()

    def refresh_display(self):
        self.refresh_image()
        self.refresh_spectrum_plot()
        # Project spectral view can be expensive for huge cubes; update it with
        # the button when needed.

    def image_view_key(self):
        key = self.current_key()
        if key is None:
            return None
        return (key, self.current_rotation())

    def remember_image_view(self, ax=None):
        if self._restoring_view:
            return

        view_key = self.image_view_key()
        if view_key is None or not self.files or len(self.ax.images) == 0:
            return

        self.image_views[view_key] = {
            "xlim": tuple(float(v) for v in self.ax.get_xlim()),
            "ylim": tuple(float(v) for v in self.ax.get_ylim())
        }

    def restore_image_view(self):
        view_key = self.image_view_key()
        if view_key is None:
            return False

        view = self.image_views.get(view_key)
        if not view:
            return False

        self._restoring_view = True
        try:
            self.ax.set_xlim(view["xlim"])
            self.ax.set_ylim(view["ylim"])
        finally:
            self._restoring_view = False

        return True

    def fit_full_image(self):
        if not self.files:
            return

        view_key = self.image_view_key()
        if view_key in self.image_views:
            del self.image_views[view_key]

        self.refresh_image(preserve_view=False)

    def get_base_display_image(self, cube, key):
        if key in self.display_image_cache:
            return self.display_image_cache[key], self.display_scales[key]

        img, sy, sx = grayscale_image(cube, max_display_pixels=self.max_display_pixels)
        self.display_image_cache[key] = img
        self.display_scales[key] = (sy, sx)
        return img, (sy, sx)

    def original_shape_yx(self, cube):
        cube2 = cube_to_yx_lambda(cube)
        return cube2.shape[:2]

    def original_to_display_scaled_yx(self, original_y, original_x, original_h, original_w, rotation_deg, sy, sx):
        # Convert via the unrotated preview grid. Dividing a rotated full-size
        # coordinate by sy/sx is wrong after 90/270 degree rotations when the
        # preview y/x scale factors differ.
        base_h, base_w = preview_shape_from_original(original_h, original_w, sy, sx)
        base_y = int(original_y) // sy
        base_x = int(original_x) // sx
        yd, xd = base_preview_to_display_yx(base_y, base_x, base_h, base_w, rotation_deg)
        return int(yd), int(xd)

    def display_scaled_to_original_yx(self, display_y, display_x, original_h, original_w, rotation_deg, sy, sx):
        # Map the rotated display-preview coordinate back to the unrotated
        # downsampled preview grid first, then expand to original coordinates.
        # This is the same path used by rectangle/ellipse selection.
        base_h, base_w = preview_shape_from_original(original_h, original_w, sy, sx)
        base_y, base_x = display_to_base_preview_yx(
            display_y, display_x, base_h, base_w, rotation_deg
        )
        yo = int(base_y) * sy
        xo = int(base_x) * sx
        return int(yo), int(xo)

    def compact_filename(self, name, max_chars=36):
        text = str(name)
        if len(text) <= max_chars:
            return text
        keep = max_chars - 3
        left = keep // 2
        right = keep - left
        return f"{text[:left]}...{text[-right:]}"

    def refresh_image(self, preserve_view=True):
        self.remember_image_view()

        self.ax.clear()

        if not self.files:
            self.file_label.setText("No file loaded")
            self.file_label.setToolTip("")
            self.shape_label.setText("Shape: -")
            self.rotation_label.setText("Rotation: 0°")
            self.count_label.setText("No class selected")
            self.hover_label.setText("Hover: -")
            self.image_canvas.draw()
            return

        cube = self.cubes[self.index]
        key = self.current_key()
        base_img, (sy, sx) = self.get_base_display_image(cube, key)
        rotation = self.current_rotation()
        img = rotate_image_for_display(base_img, rotation)

        original_h, original_w = self.original_shape_yx(cube)

        self.ax.imshow(img, cmap="gray", origin="upper", interpolation="nearest")
        display_name = self.compact_filename(self.files[self.index].name, max_chars=48)
        self.ax.set_title(
            f"{display_name} | rotation {rotation}° | scale y={sy}, x={sx}"
        )
        self.ax.set_axis_off()

        self.file_label.setText(
            f"{self.index + 1}/{len(self.files)}: {self.compact_filename(self.files[self.index].name)}"
        )
        self.file_label.setToolTip(str(self.files[self.index]))
        self.shape_label.setText(f"Shape: {cube.shape}\nPreview scale: y={sy}, x={sx}")
        self.rotation_label.setText(f"Rotation: {rotation}°")

        any_marks = False
        for cls, pixels in self.selections.get(key, {}).items():
            if not pixels:
                continue

            dys = []
            dxs = []
            for yo, xo in pixels:
                yd, xd = self.original_to_display_scaled_yx(
                    yo, xo, original_h, original_w, rotation, sy, sx
                )
                dys.append(yd)
                dxs.append(xd)

            self.ax.scatter(dxs, dys, s=10, label=cls, color=self.class_color(cls))
            any_marks = True

        if any_marks:
            self.ax.legend(loc="upper right")

        cls = self.current_class()
        if cls:
            n = len(self.selections.get(key, {}).get(cls, set()))
            self.count_label.setText(f"{cls}: {n} pixels in this cube")
        else:
            self.count_label.setText("No class selected")

        if preserve_view:
            self.restore_image_view()

        self.last_image_key = key
        self.last_image_rotation = rotation

        self.image_canvas.draw()

    def on_scroll_zoom(self, event):
        if not self.files or event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Mouse-wheel zoom. This does not rely on the Matplotlib toolbar.
        base_scale = 1.25

        if event.button == "up":
            scale_factor = 1 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([
            xdata - new_width * (1 - relx),
            xdata + new_width * relx
        ])
        self.ax.set_ylim([
            ydata - new_height * (1 - rely),
            ydata + new_height * rely
        ])

        self.remember_image_view()
        self.image_canvas.draw_idle()

    def refresh_spectrum_plot(self):
        self.spectrum_ax.clear()
        self.hover_line = None
        self._hover_counter = 0
        self.hover_update_every = 4

        if not self.files:
            self.spectrum_canvas.draw()
            return

        cube = self.cubes[self.index]
        key = self.current_key()
        settings = self.preprocessing_settings()

        any_data = False

        for cls, pixels in self.selections.get(key, {}).items():
            if not pixels:
                continue

            spectra, _ = spectra_from_pixels(cube, sorted(pixels))
            spectra = preprocess_spectra(spectra, **settings)

            if spectra.size == 0:
                continue

            mean_spec = np.nanmean(spectra, axis=0)
            std_spec = np.nanstd(spectra, axis=0)
            x = np.arange(mean_spec.size)

            class_color = self.class_color(cls)
            self.spectrum_ax.plot(
                x,
                mean_spec,
                label=f"{cls} mean, n={spectra.shape[0]}",
                color=class_color,
            )
            self.spectrum_ax.fill_between(
                x,
                mean_spec - std_spec,
                mean_spec + std_spec,
                color=class_color,
                alpha=0.15,
            )
            any_data = True

        self.spectrum_ax.set_title("Selected class spectra preview + current hover spectrum")
        self.spectrum_ax.set_xlabel("Spectral band")
        self.spectrum_ax.set_ylabel("Intensity / processed value")
        self.spectrum_ax.grid(True, alpha=0.25)

        if any_data:
            self.spectrum_ax.legend(loc="best")
        else:
            self.spectrum_ax.text(
                0.5, 0.5,
                "Select pixels to preview class spectra\nMove mouse over image for live spectrum",
                ha="center", va="center",
                transform=self.spectrum_ax.transAxes
            )

        self.spectrum_canvas.draw()

    def collect_project_spectra_for_class(self, cls):
        all_spectra = []

        for path, cube in zip(self.files, self.cubes):
            key = str(path)
            pixels = sorted(self.selections.get(key, {}).get(cls, set()))
            if not pixels:
                continue

            spectra, _ = spectra_from_pixels(cube, pixels)
            if spectra.size:
                all_spectra.append(spectra)

        if not all_spectra:
            if self.cubes:
                bands = cube_to_yx_lambda(self.cubes[0]).shape[-1]
                return np.empty((0, bands))
            return np.empty((0, 0))

        spectra = np.vstack(all_spectra)
        spectra = preprocess_spectra(spectra, **self.preprocessing_settings())
        return spectra

    def refresh_project_spectral_view(self):
        self.project_ax.clear()

        if not self.files:
            self.project_canvas.draw()
            return

        any_data = False

        for cls in self.classes:
            spectra = self.collect_project_spectra_for_class(cls)

            if spectra.size == 0:
                continue

            class_color = self.class_color(cls)
            x = np.arange(spectra.shape[1])

            # Draw individual spectra lightly using the same class color.
            # For very large selections, sample them to keep the GUI responsive.
            max_individual = 250
            if spectra.shape[0] > max_individual:
                idx = np.linspace(0, spectra.shape[0] - 1, max_individual).astype(int)
                spectra_to_draw = spectra[idx]
            else:
                spectra_to_draw = spectra

            for spec in spectra_to_draw:
                self.project_ax.plot(
                    x,
                    spec,
                    color=class_color,
                    alpha=0.08,
                    linewidth=0.7
                )

            mean_spec = np.nanmean(spectra, axis=0)
            std_spec = np.nanstd(spectra, axis=0)

            self.project_ax.fill_between(
                x,
                mean_spec - std_spec,
                mean_spec + std_spec,
                color=class_color,
                alpha=0.12
            )
            self.project_ax.plot(
                x,
                mean_spec,
                color=class_color,
                linewidth=2.2,
                label=f"{cls} mean, n={spectra.shape[0]}"
            )

            any_data = True

        self.project_ax.set_title("PROJECT SPECTRAL VIEW: all selected spectra across all loaded cubes")
        self.project_ax.set_xlabel("Spectral band")
        self.project_ax.set_ylabel("Intensity / processed value")
        self.project_ax.grid(True, alpha=0.25)

        if any_data:
            self.project_ax.legend(loc="best")
        else:
            self.project_ax.text(
                0.5, 0.5,
                "No project spectra selected yet",
                ha="center", va="center",
                transform=self.project_ax.transAxes
            )

        self.project_canvas.draw()


    def display_xy_to_original_yx_safe(self, xd, yd):
        cube = self.cubes[self.index]
        key = self.current_key()
        _, (sy, sx) = self.get_base_display_image(cube, key)
        original_h, original_w = self.original_shape_yx(cube)
        rotation = self.current_rotation()

        yo, xo = self.display_scaled_to_original_yx(yd, xd, original_h, original_w, rotation, sy, sx)

        if 0 <= yo < original_h and 0 <= xo < original_w:
            return yo, xo
        return None

    def update_hover_spectrum(self, xd, yd):
        if not self.files:
            return

        mapped = self.display_xy_to_original_yx_safe(xd, yd)
        if mapped is None:
            return

        yo, xo = mapped

        cube2 = cube_to_yx_lambda(self.cubes[self.index])
        bands = cube2.shape[-1]

        raw = cube2[yo, xo, :][None, :]
        spec = preprocess_spectra(raw, **self.preprocessing_settings())[0]

        x_axis = np.arange(bands)

        if self.hover_line is None:
            self.hover_line, = self.spectrum_ax.plot(
                x_axis, spec,
                linestyle="--",
                linewidth=1.5,
                label=f"hover original y={yo}, x={xo}"
            )
        else:
            self.hover_line.set_ydata(spec)
            self.hover_line.set_label(f"hover original y={yo}, x={xo}")

        self.hover_label.setText(f"Hover original: y={yo}, x={xo}")

        self.spectrum_ax.relim()
        self.spectrum_ax.autoscale_view()
        self.spectrum_ax.legend(loc="best")
        self.spectrum_canvas.draw_idle()

    def image_toolbar_navigation_active(self):
        mode = getattr(self.image_toolbar, "mode", None)
        if mode is None:
            return False

        name = str(getattr(mode, "name", "")).strip().lower()
        text = str(mode).strip().lower()

        if name == "none" or text in ("", "none", "_mode.none"):
            return False

        return True

    def on_press(self, event):
        if self.image_toolbar_navigation_active():
            return

        if event.inaxes != self.ax or not self.files:
            return

        if self.current_class() is None:
            QMessageBox.information(self, "No class selected", "Create and select a class first.")
            return

        if event.xdata is None or event.ydata is None:
            return

        xd = int(round(event.xdata))
        yd = int(round(event.ydata))

        if self.mode == "pixel":
            mapped = self.display_xy_to_original_yx_safe(xd, yd)
            if mapped is not None:
                self.apply_pixels([mapped])
                self.refresh_display()
        else:
            self.drag_start = (xd, yd)

    def on_motion(self, event):
        if not self.files or event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        xd = int(round(event.xdata))
        yd = int(round(event.ydata))

        # Throttle hover spectra for large, memory-mapped cubes.
        self._hover_counter += 1
        if self._hover_counter % self.hover_update_every == 0:
            self.update_hover_spectrum(xd, yd)

        if self.drag_start is None:
            return

        x0, y0 = self.drag_start
        x1, y1 = event.xdata, event.ydata

        if self.temp_patch:
            self.temp_patch.remove()
        
        selection_colour = "yellow"   # or: "cyan", "lime", "white"
        if self.mode == "rectangle":
            self.temp_patch = Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0),
                abs(y1 - y0),
                fill=False,
                edgecolor=selection_colour,
                linewidth=1
            )
        else:
            self.temp_patch = Ellipse(
                ((x0 + x1) / 2, (y0 + y1) / 2),
                abs(x1 - x0),
                abs(y1 - y0),
                fill=False,
                edgecolor=selection_colour,
                linewidth=1
            )

        self.ax.add_patch(self.temp_patch)
        self.image_canvas.draw_idle()

    def on_release(self, event):
        if self.drag_start is None or event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            self.drag_start = None
            return

        x0, y0 = self.drag_start
        x1, y1 = int(round(event.xdata)), int(round(event.ydata))

        pixels = self.pixels_in_shape_display_then_original(x0, y0, x1, y1)
        self.apply_pixels(pixels)

        self.drag_start = None

        if self.temp_patch:
            self.temp_patch.remove()
            self.temp_patch = None

        self.refresh_display()

    def pixels_in_shape_display_then_original(self, x0, y0, x1, y1):
        cube = self.cubes[self.index]
        key = self.current_key()
        base_img, (sy, sx) = self.get_base_display_image(cube, key)
        display_img = rotate_image_for_display(base_img, self.current_rotation())
        dh, dw = display_img.shape

        xmin, xmax = sorted([int(round(x0)), int(round(x1))])
        ymin, ymax = sorted([int(round(y0)), int(round(y1))])

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(dw - 1, xmax)
        ymax = min(dh - 1, ymax)

        display_pixels = []

        if self.mode == "rectangle":
            for yd in range(ymin, ymax + 1):
                for xd in range(xmin, xmax + 1):
                    display_pixels.append((yd, xd))

        elif self.mode == "ellipse":
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            rx = max((xmax - xmin) / 2, 1)
            ry = max((ymax - ymin) / 2, 1)

            for yd in range(ymin, ymax + 1):
                for xd in range(xmin, xmax + 1):
                    if ((xd - cx) ** 2 / rx ** 2) + ((yd - cy) ** 2 / ry ** 2) <= 1:
                        display_pixels.append((yd, xd))

        original_pixels = set()
        original_h, original_w = self.original_shape_yx(cube)
        rotation = self.current_rotation()
        base_h, base_w = base_img.shape

        # Expand each selected rotated-display preview pixel back to the
        # corresponding unrotated preview cell first. This fixes rectangle and
        # ellipse selections after rotation and after changing the displayed view.
        for yd, xd in display_pixels:
            base_y, base_x = display_to_base_preview_yx(yd, xd, base_h, base_w, rotation)

            if not (0 <= base_y < base_h and 0 <= base_x < base_w):
                continue

            y0_full = int(base_y) * sy
            x0_full = int(base_x) * sx
            y1_full = min(y0_full + sy, original_h)
            x1_full = min(x0_full + sx, original_w)

            for yo in range(y0_full, y1_full):
                for xo in range(x0_full, x1_full):
                    original_pixels.add((yo, xo))

        return list(original_pixels)

    def apply_pixels(self, pixels):
        cls = self.current_class()
        key = self.current_key()

        if cls is None or key is None:
            return

        self.selections[key].setdefault(cls, set())

        if self.action == "add":
            self.selections[key][cls].update(pixels)
        else:
            self.selections[key][cls].difference_update(pixels)

    def export_xlsx(self):
        if not self.files:
            return

        processed = "preprocessed" in self.export_kind_box.currentText()

        out_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not out_dir:
            return

        out_dir = Path(out_dir)
        settings = self.preprocessing_settings()

        yaml_settings = {
            "export": {
                "type": "preprocessed" if processed else "raw",
                "xlsx_contains": "spectra_values_only",
                "xlsx_filename_pattern": "<class_name>.xlsx"
            },
            "preprocessing": {
                "savgol_smoothing": self.smooth_check.isChecked() if processed else False,
                "savgol_derivative": self.deriv_check.isChecked() if processed else False,
                "zscore": self.zscore_check.isChecked() if processed else False,
                "savgol_window": self.window_spin.value() if processed else None,
                "savgol_polyorder": self.poly_spin.value() if processed else None
            },
            "files": []
        }

        for cls in self.classes:
            all_spectra = []

            for path, cube in zip(self.files, self.cubes):
                key = str(path)
                pixels = sorted(self.selections.get(key, {}).get(cls, set()))

                spectra, valid_pixels = spectra_from_pixels(cube, pixels)

                if processed:
                    spectra = preprocess_spectra(spectra, **settings)

                if spectra.size:
                    all_spectra.append(spectra)

                yaml_settings["files"].append({
                    "source_file": path.name,
                    "class": cls,
                    "n_selected_pixels": len(valid_pixels),
                    "remembered_display_rotation_deg": self.rotations.get(key, 0),
                    "selected_original_coordinates_yx": [
                        [int(y), int(x)] for y, x in valid_pixels
                    ]
                })

            if all_spectra:
                matrix = np.vstack(all_spectra)
                df = pd.DataFrame(matrix)
                df.to_excel(out_dir / f"{cls}.xlsx", index=False, header=False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        write_export_settings_yaml(out_dir / f"export_settings_{timestamp}.yaml", yaml_settings)

        QMessageBox.information(
            self,
            "Export complete",
            f"Exported XLSX spectra and timestamped export settings YAML to:\n{out_dir}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = HSIAnnotator()
    win.show()
    sys.exit(app.exec())

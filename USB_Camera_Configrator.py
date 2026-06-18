import platform
import re
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import yaml
from PIL import Image, ImageTk


CAMERA_INDEX = 0
DEVICE_LINUX = "/dev/video0"

DEFAULT_FOURCC = "MJPG"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 60


def run_cmd(args):
    return subprocess.run(args, text=True, capture_output=True, check=True)


def v4l2_get_formats(device):
    try:
        result = run_cmd(["v4l2-ctl", "-d", device, "--list-formats-ext"])
    except Exception:
        return [(DEFAULT_FOURCC, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS)]

    formats = set()
    current_fourcc = None
    current_size = None

    fourcc_pattern = re.compile(r"\[\d+\]:\s+'(\w+)'")
    size_pattern = re.compile(r"Size:\s+Discrete\s+(\d+)x(\d+)")
    fps_pattern = re.compile(r"Interval:\s+Discrete\s+.*\(([\d.]+)\s+fps\)")

    for line in result.stdout.splitlines():
        fourcc_match = fourcc_pattern.search(line)
        if fourcc_match:
            current_fourcc = fourcc_match.group(1)
            continue

        size_match = size_pattern.search(line)
        if size_match:
            current_size = (int(size_match.group(1)), int(size_match.group(2)))
            continue

        fps_match = fps_pattern.search(line)
        if fps_match and current_fourcc and current_size:
            formats.add((
                current_fourcc,
                current_size[0],
                current_size[1],
                float(fps_match.group(1)),
            ))

    return sorted(formats, key=lambda x: (x[1] * x[2], x[3], x[0]), reverse=True)


def v4l2_get_controls(device):
    try:
        result = run_cmd(["v4l2-ctl", "-d", device, "--list-ctrls-menus"])
    except Exception:
        return []

    controls = []
    current = None

    ctrl_pattern = re.compile(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$")
    menu_pattern = re.compile(r"^\s+(-?\d+):\s*(.+)$")

    for line in result.stdout.splitlines():
        ctrl_match = ctrl_pattern.match(line)

        if ctrl_match:
            name, ctrl_type, rest = ctrl_match.groups()
            data = dict(re.findall(r"(\w+)=(-?\d+)", rest))
            flags = re.search(r"flags=([a-zA-Z_,]+)", rest)

            if flags and "inactive" in flags.group(1):
                current = None
                continue

            if ctrl_type not in ("int", "bool", "menu", "integer64"):
                current = None
                continue

            if "value" not in data:
                current = None
                continue

            current = {
                "backend": "v4l2",
                "name": name,
                "type": ctrl_type,
                "min": int(data.get("min", 0)),
                "max": int(data.get("max", 1)),
                "step": max(1, int(data.get("step", 1))),
                "value": int(data["value"]),
                "default": int(data.get("default", data["value"])),
                "menu_items": {},
            }

            controls.append(current)
            continue

        menu_match = menu_pattern.match(line)
        if menu_match and current is not None:
            value, label = menu_match.groups()
            current["menu_items"][int(value)] = label.strip()

    return controls


def v4l2_set_control(device, name, value):
    subprocess.run(
        ["v4l2-ctl", "-d", device, "-c", f"{name}={int(value)}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


COMMON_FORMATS = [
    (640, 480),
    (800, 600),
    (1280, 720),
    (1280, 960),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]

COMMON_FPS = [5, 10, 15, 20, 25, 30, 50, 60, 90, 120]
COMMON_FOURCCS = ["MJPG", "YUYV", "H264", "NV12"]


CV_PROPS = {
    "Brightness": cv2.CAP_PROP_BRIGHTNESS,
    "Contrast": cv2.CAP_PROP_CONTRAST,
    "Saturation": cv2.CAP_PROP_SATURATION,
    "Hue": cv2.CAP_PROP_HUE,
    "Gain": cv2.CAP_PROP_GAIN,
    "Exposure": cv2.CAP_PROP_EXPOSURE,
    "Focus": cv2.CAP_PROP_FOCUS,
    "Sharpness": cv2.CAP_PROP_SHARPNESS,
    "Gamma": cv2.CAP_PROP_GAMMA,
    "White balance temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    "Zoom": cv2.CAP_PROP_ZOOM,
    "Pan": cv2.CAP_PROP_PAN,
    "Tilt": cv2.CAP_PROP_TILT,
    "Roll": cv2.CAP_PROP_ROLL,
    "Auto exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "Auto white balance": cv2.CAP_PROP_AUTO_WB,
}

PROBE_VALUES = [
    -10000, -1000, -500, -255, -128, -64, -32, -16, -8, -4, -2, -1,
    0, 1, 2, 4, 8, 16, 32, 64, 128, 255, 500, 1000, 10000,
]


def opencv_probe_formats(cap):
    found = set()

    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ofps = cap.get(cv2.CAP_PROP_FPS)
    ofourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    for fourcc in COMMON_FOURCCS:
        if len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        for w, h in COMMON_FORMATS:
            for fps in COMMON_FPS:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, fps)

                rw = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                rh = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                rfps = round(cap.get(cv2.CAP_PROP_FPS) or 0)

                if abs(rw - w) <= 32 and abs(rh - h) <= 32 and rfps > 0:
                    found.add((fourcc, rw, rh, rfps))

    cap.set(cv2.CAP_PROP_FOURCC, ofourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ow)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, oh)
    cap.set(cv2.CAP_PROP_FPS, ofps)

    return sorted(found, key=lambda x: (x[1] * x[2], x[3], x[0]), reverse=True)


def opencv_get_controls(cap):
    controls = []

    for name, prop in CV_PROPS.items():
        original = cap.get(prop)
        observed = []

        if original == -1:
            continue

        for value in PROBE_VALUES:
            ok = cap.set(prop, float(value))
            readback = cap.get(prop)

            if ok and readback != -1:
                observed.append(round(readback, 4))

        cap.set(prop, original)

        unique = sorted(set(observed))
        if len(unique) < 2:
            continue

        controls.append({
            "backend": "opencv",
            "name": name,
            "prop": prop,
            "type": "int",
            "min": min(unique),
            "max": max(unique),
            "step": 1,
            "value": original,
            "default": original,
            "menu_items": {},
        })

    return controls


def fourcc_to_string(fourcc_int):
    try:
        return "".join(chr((int(fourcc_int) >> 8 * i) & 0xFF) for i in range(4))
    except Exception:
        return "????"


class CameraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Universal USB Camera Control")
        self.root.geometry("1600x900")
        self.show_center_crop = tk.BooleanVar(value=False)
        self.crop_size = 640
        self.crop_offset_x = tk.IntVar(value=0)
        self.crop_offset_y = tk.IntVar(value=0)
        self.current_frame_size = None
        self.display_frame_size = None
        self.dragging_reticule = False
        
        
        self.os_name = platform.system()
        self.widgets_by_control = {}

        self.cap = self.open_camera()
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        self.set_camera_format(DEFAULT_FOURCC, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS)

        if self.os_name == "Linux":
            self.formats = v4l2_get_formats(DEVICE_LINUX)
            self.controls = v4l2_get_controls(DEVICE_LINUX)
        else:
            self.formats = opencv_probe_formats(self.cap)
            self.controls = opencv_get_controls(self.cap)

        self.build_ui()
        self.update_actual_format_label()
        self.update_frame()

    def open_camera(self):
        if self.os_name == "Windows":
            return cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if self.os_name == "Linux":
            return cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        return cv2.VideoCapture(CAMERA_INDEX)

    def build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        left_frame = ttk.Frame(main)
        left_frame.pack(side="left", fill="both", expand=True)

        self.image_label = ttk.Label(left_frame)
        self.image_label.pack(anchor="nw", padx=8, pady=8)
        self.image_label.bind("<Button-1>", self.on_reticule_mouse_down)
        self.image_label.bind("<B1-Motion>", self.on_reticule_drag)
        self.image_label.bind("<ButtonRelease-1>", self.on_reticule_mouse_up)

        right_outer = ttk.Frame(main, width=460)
        right_outer.pack(side="right", fill="y")
        right_outer.pack_propagate(False)

        canvas = tk.Canvas(right_outer, width=440, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_outer, orient="vertical", command=canvas.yview)

        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        panel = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=panel, anchor="nw")

        panel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

        def on_mousewheel(event):
            if self.os_name == "Windows":
                canvas.yview_scroll(int(-event.delta / 120), "units")
            elif self.os_name == "Darwin":
                canvas.yview_scroll(int(-event.delta), "units")
            else:
                if event.num == 4:
                    canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    canvas.yview_scroll(1, "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.bind_all("<Button-4>", on_mousewheel)
        canvas.bind_all("<Button-5>", on_mousewheel)

        ttk.Label(panel, text=f"OS: {self.os_name}").pack(anchor="w", padx=6, pady=(6, 2))

        if self.os_name == "Linux":
            ttk.Label(panel, text=f"Device: {DEVICE_LINUX}").pack(anchor="w", padx=6, pady=(0, 8))

        self.add_format_controls(panel)
        
        reticule_frame = ttk.LabelFrame(panel, text="640x640 Center Crop Reticule")
        reticule_frame.pack(fill="x", padx=6, pady=6)
        ttk.Checkbutton(
            reticule_frame,
            text="Show reticule",
            variable=self.show_center_crop,
        ).pack(anchor="w", padx=4, pady=4)
        ttk.Label(reticule_frame, text="Offset X").pack(anchor="w", padx=4)
        ttk.Spinbox(
            reticule_frame,
            from_=-10000,
            to=10000,
            textvariable=self.crop_offset_x,
            command=self.clamp_crop_offsets,
            width=10,
        ).pack(fill="x", padx=4, pady=2)
        
        ttk.Label(reticule_frame, text="Offset Y").pack(anchor="w", padx=4)
        ttk.Spinbox(
            reticule_frame,
            from_=-10000,
            to=10000,
            textvariable=self.crop_offset_y,
            command=self.clamp_crop_offsets,
            width=10,
        ).pack(fill="x", padx=4, pady=2)
        
        ttk.Button(
            reticule_frame,
            text="Reset crop offset",
            command=lambda: self.set_crop_offsets(0, 0),
        ).pack(fill="x", padx=4, pady=4)
        
        ttk.Label(
            reticule_frame,
            text="You can also drag the reticule center in the image.",
            wraplength=400,
        ).pack(anchor="w", padx=4, pady=(0, 4))
        
        ttk.Label(
            reticule_frame,
            text="Reticule is disabled at low Resolutions [min(Resolutions)<640]",
            wraplength=400,
        ).pack(anchor="w", padx=4, pady=(0, 4))


        ttk.Separator(panel).pack(fill="x", padx=6, pady=8)

        ttk.Button(
            panel,
            text="Load YAML and Apply Settings",
            command=self.load_yaml,
        ).pack(fill="x", padx=6, pady=6)

        ttk.Button(
            panel,
            text="Export YAML for SenSoRTC",
            command=self.export_yaml,
        ).pack(fill="x", padx=6, pady=6)

        ttk.Label(panel, text=f"Detected usable controls: {len(self.controls)}").pack(
            anchor="w", padx=6, pady=(0, 8)
        )

        ttk.Button(panel, text="Restore All Defaults", command=self.restore_all_defaults).pack(
            fill="x", padx=6, pady=6
        )

        if self.os_name == "Windows":
            ttk.Button(
                panel,
                text="Open native camera settings",
                command=self.open_windows_camera_settings,
            ).pack(fill="x", padx=6, pady=6)

        for control in self.controls:
            self.add_control(panel, control)

        ttk.Button(panel, text="Quit", command=self.close).pack(fill="x", padx=6, pady=12)

    def add_format_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Pixel Format / Resolution / FPS")
        frame.pack(fill="x", padx=6, pady=6)

        self.format_labels = [
            f"{fourcc} | {w} x {h} @ {fps:g} FPS"
            for fourcc, w, h, fps in self.formats
        ]

        self.format_var = tk.StringVar(value=self.format_labels[0] if self.format_labels else "")

        combo = ttk.Combobox(
            frame,
            textvariable=self.format_var,
            values=self.format_labels,
            state="readonly",
        )
        combo.pack(fill="x", padx=4, pady=4)

        ttk.Button(frame, text="Apply format", command=self.apply_selected_format).pack(
            fill="x", padx=4, pady=4
        )

        self.actual_format_label = ttk.Label(frame, text="")
        self.actual_format_label.pack(anchor="w", padx=4, pady=4)

        ttk.Label(
            frame,
            text="Tip: choose MJPG for high FPS. YUYV is often USB-bandwidth-limited.",
            wraplength=400,
        ).pack(anchor="w", padx=4, pady=(0, 4))

    def apply_selected_format(self):
        text = self.format_var.get()

        match = re.search(r"(\w+)\s+\|\s+(\d+)\s+x\s+(\d+)\s+@\s+([\d.]+)", text)
        if not match:
            return

        fourcc = match.group(1)
        width = int(match.group(2))
        height = int(match.group(3))
        fps = float(match.group(4))

        self.set_camera_format(fourcc, width, height, fps)
        self.update_actual_format_label()

    def set_camera_format(self, fourcc, width, height, fps):
        if self.os_name == "Linux":
            self.cap.release()

            subprocess.run(
                [
                    "v4l2-ctl",
                    "-d",
                    DEVICE_LINUX,
                    f"--set-fmt-video=width={width},height={height},pixelformat={fourcc}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            subprocess.run(
                ["v4l2-ctl", "-d", DEVICE_LINUX, f"--set-parm={fps}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self.cap = self.open_camera()

        if len(fourcc) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def update_actual_format_label(self):
        if not hasattr(self, "actual_format_label"):
            return

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        fourcc = fourcc_to_string(self.cap.get(cv2.CAP_PROP_FOURCC))

        self.actual_format_label.configure(
            text=f"Current: {fourcc} | {actual_w} x {actual_h} @ {actual_fps:.2f} FPS"
        )

    def get_selected_format(self):
        text = self.format_var.get()
        match = re.search(r"(\w+)\s+\|\s+(\d+)\s+x\s+(\d+)\s+@\s+([\d.]+)", text)

        if match:
            return {
                "fourcc": match.group(1),
                "width": int(match.group(2)),
                "height": int(match.group(3)),
                "fps": float(match.group(4)),
            }

        return {
            "fourcc": fourcc_to_string(self.cap.get(cv2.CAP_PROP_FOURCC)).strip(),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self.cap.get(cv2.CAP_PROP_FPS) or 0),
        }

    def add_control(self, parent, control):
        frame = ttk.LabelFrame(parent, text=control["name"])
        frame.pack(fill="x", padx=6, pady=4)

        value_label = ttk.Label(frame, text=str(control["value"]))
        value_label.pack(anchor="e", padx=4)

        self.widgets_by_control[control["name"]] = {
            "label": value_label,
            "slider": None,
            "var": None,
            "combo": None,
        }

        if control["type"] == "bool":
            var = tk.IntVar(value=int(control["value"]))
            self.widgets_by_control[control["name"]]["var"] = var

            ttk.Checkbutton(
                frame,
                text="Enabled",
                variable=var,
                command=lambda c=control, v=var, lbl=value_label:
                    self.set_control(c, v.get(), lbl),
            ).pack(anchor="w", padx=4, pady=4)

        elif control["type"] == "menu" and control["menu_items"]:
            values = [
                f"{value}: {label}"
                for value, label in sorted(control["menu_items"].items())
            ]

            current = f"{control['value']}: {control['menu_items'].get(control['value'], '')}"
            var = tk.StringVar(value=current)

            combo = ttk.Combobox(frame, textvariable=var, values=values, state="readonly")
            combo.pack(fill="x", padx=4, pady=4)

            self.widgets_by_control[control["name"]]["combo"] = combo

            ttk.Button(
                frame,
                text="Apply",
                command=lambda c=control, v=var, lbl=value_label:
                    self.set_control(c, int(v.get().split(":")[0]), lbl),
            ).pack(fill="x", padx=4, pady=2)

        else:
            slider = ttk.Scale(
                frame,
                from_=control["min"],
                to=control["max"],
                orient="horizontal",
                command=lambda v, c=control, lbl=value_label:
                    self.set_control(c, v, lbl),
            )
            slider.set(control["value"])
            slider.pack(fill="x", padx=4, pady=4)

            self.widgets_by_control[control["name"]]["slider"] = slider

            ttk.Label(
                frame,
                text=(
                    f"min {control['min']} / "
                    f"max {control['max']} / "
                    f"step {control['step']} / "
                    f"default {control['default']}"
                ),
            ).pack(anchor="w", padx=4)

        ttk.Button(
            frame,
            text="Reset this control",
            command=lambda c=control: self.restore_one_default(c),
        ).pack(anchor="e", padx=4, pady=2)

    def set_control(self, control, value, label=None):
        step = control.get("step", 1)
        value = round(float(value) / step) * step

        if control["backend"] == "v4l2":
            v4l2_set_control(DEVICE_LINUX, control["name"], value)
            actual = int(value)
        else:
            self.cap.set(control["prop"], float(value))
            actual = self.cap.get(control["prop"])

        if label is not None:
            label.configure(text=str(round(actual, 2)))

        return actual

    def get_control_value(self, control):
        if control["backend"] == "v4l2":
            try:
                result = run_cmd(["v4l2-ctl", "-d", DEVICE_LINUX, "-C", control["name"]])
                match = re.search(r":\s*(-?\d+)", result.stdout)
                if match:
                    return int(match.group(1))
            except Exception:
                return control["value"]

        return self.cap.get(control["prop"])

    def collect_settings(self):
        fmt = self.get_selected_format()

        control_values = {}
        for control in self.controls:
            control_values[control["name"]] = {
                "value": self.get_control_value(control),
                "type": control["type"],
                "backend": control["backend"],
            }

        return {
            "camera": {
                "index": CAMERA_INDEX,
                "device_linux": DEVICE_LINUX if self.os_name == "Linux" else None,
                "backend": self.os_name,
                "format": fmt,
                "controls": control_values,
                "reticule": {
                    "enabled": bool(self.show_center_crop.get()),
                    "crop_size": int(self.crop_size),
                    "offset_x": int(self.crop_offset_x.get()),
                    "offset_y": int(self.crop_offset_y.get()),
                },
            }
        }

    def export_yaml(self):
        settings = self.collect_settings()

        path = filedialog.asksaveasfilename(
            title="Export SenSoRTC camera settings",
            defaultextension=".yaml",
            filetypes=[
                ("YAML files", "*.yaml"),
                ("YAML files", "*.yml"),
                ("All files", "*.*"),
            ],
            initialfile="sensor_tc_camera_settings.yaml",
        )

        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(settings, f, sort_keys=False)

        messagebox.showinfo("Export complete", f"Saved settings to:\n{path}")

    def load_yaml(self):
        path = filedialog.askopenfilename(
            title="Load SenSoRTC camera settings",
            filetypes=[
                ("YAML files", "*.yaml *.yml"),
                ("All files", "*.*"),
            ],
        )
    
        if not path:
            return
    
        try:
            with open(path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
    
            self.apply_loaded_settings(settings)
            messagebox.showinfo("Load complete", f"Loaded settings from:\n{path}")
    
        except Exception as e:
            messagebox.showerror("Load failed", str(e))


    def apply_loaded_settings(self, settings):
        cam = settings.get("camera", {})
    
        fmt = cam.get("format", {})
        fourcc = fmt.get("fourcc")
        width = fmt.get("width")
        height = fmt.get("height")
        fps = fmt.get("fps")
    
        if fourcc and width and height and fps:
            self.set_camera_format(
                str(fourcc),
                int(width),
                int(height),
                float(fps),
            )
    
            label = f"{fourcc} | {int(width)} x {int(height)} @ {float(fps):g} FPS"
            if label in self.format_labels:
                self.format_var.set(label)
    
            self.update_actual_format_label()
    
        reticule = cam.get("reticule", {})
        self.show_center_crop.set(bool(reticule.get("enabled", False)))
        self.crop_size = int(reticule.get("crop_size", 640))
        self.set_crop_offsets(
            int(reticule.get("offset_x", 0)),
            int(reticule.get("offset_y", 0)),
        )
    
        controls = cam.get("controls", {})
    
        for control in self.controls:
            name = control["name"]
    
            if name not in controls:
                continue
    
            loaded = controls[name]
            value = loaded.get("value") if isinstance(loaded, dict) else loaded
    
            if value is None:
                continue
    
            actual = self.set_control(control, value, self.widgets_by_control[name].get("label"))
            widgets = self.widgets_by_control.get(name, {})
    
            if widgets.get("slider") is not None:
                widgets["slider"].set(actual)
    
            if widgets.get("var") is not None:
                widgets["var"].set(int(actual))
    
            if widgets.get("combo") is not None and control.get("menu_items"):
                label = control["menu_items"].get(int(actual), "")
                widgets["combo"].set(f"{int(actual)}: {label}")

    def restore_one_default(self, control):
        default = control["default"]
        widgets = self.widgets_by_control.get(control["name"], {})

        actual = self.set_control(control, default, widgets.get("label"))

        if widgets.get("slider") is not None:
            widgets["slider"].set(actual)

        if widgets.get("var") is not None:
            widgets["var"].set(int(actual))

        if widgets.get("combo") is not None and control.get("menu_items"):
            label = control["menu_items"].get(int(default), "")
            widgets["combo"].set(f"{int(default)}: {label}")

    def restore_all_defaults(self):
        for control in self.controls:
            self.restore_one_default(control)

    def open_windows_camera_settings(self):
        self.cap.set(cv2.CAP_PROP_SETTINGS, 0)

    def set_crop_offsets(self, x, y):
        self.crop_offset_x.set(int(x))
        self.crop_offset_y.set(int(y))
        self.clamp_crop_offsets()
    
    def clamp_crop_offsets(self):
        if not self.current_frame_size:
            return
    
        w, h = self.current_frame_size
        crop = self.crop_size
    
        if w <= crop or h <= crop:
            self.crop_offset_x.set(0)
            self.crop_offset_y.set(0)
            return
    
        max_x = (w - crop) // 2
        max_y = (h - crop) // 2
    
        x = max(-max_x, min(max_x, int(self.crop_offset_x.get())))
        y = max(-max_y, min(max_y, int(self.crop_offset_y.get())))
    
        self.crop_offset_x.set(x)
        self.crop_offset_y.set(y)
    
    def set_crop_center_from_display_xy(self, display_x, display_y):
        if not self.current_frame_size or not self.display_frame_size:
            return
    
        frame_w, frame_h = self.current_frame_size
        display_w, display_h = self.display_frame_size
    
        if display_w <= 0 or display_h <= 0:
            return
    
        scale_x = frame_w / display_w
        scale_y = frame_h / display_h
    
        image_x = int(display_x * scale_x)
        image_y = int(display_y * scale_y)
    
        offset_x = image_x - frame_w // 2
        offset_y = image_y - frame_h // 2
    
        self.set_crop_offsets(offset_x, offset_y)
    
    def on_reticule_mouse_down(self, event):
        if not self.show_center_crop.get():
            return
    
        self.dragging_reticule = True
        self.set_crop_center_from_display_xy(event.x, event.y)
    
    def on_reticule_drag(self, event):
        if not self.dragging_reticule:
            return
    
        self.set_crop_center_from_display_xy(event.x, event.y)
    
    def on_reticule_mouse_up(self, event):
        self.dragging_reticule = False

    def update_frame(self):
        ok, frame = self.cap.read()
    
        if ok:
            h, w = frame.shape[:2]
            self.current_frame_size = (w, h)
    
            crop = self.crop_size
    
            if self.show_center_crop.get() and w > crop and h > crop:
                self.clamp_crop_offsets()
    
                cx = w // 2 + int(self.crop_offset_x.get())
                cy = h // 2 + int(self.crop_offset_y.get())
    
                half = crop // 2
    
                x1 = max(0, min(w - crop, cx - half))
                y1 = max(0, min(h - crop, cy - half))
                x2 = x1 + crop
                y2 = y1 + crop
    
                cx = x1 + half
                cy = y1 + half
    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 1)
                cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 1)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    
                cv2.putText(
                    frame,
                    f"crop 640x640 offset x={self.crop_offset_x.get()} y={self.crop_offset_y.get()}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image.thumbnail((1150, 850))
    
            self.display_frame_size = image.size
    
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
    
        self.root.after(15, self.update_frame)

    def close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()

    try:
        app = CameraGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.close)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Camera error", str(e))    
        try:
            app.cap.release()
        except Exception as e:
            messagebox.showerror("Ending Error", str(e))
        root.destroy()
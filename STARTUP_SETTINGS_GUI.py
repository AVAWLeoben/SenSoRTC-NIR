# -*- coding: utf-8 -*-
"""
Startup config GUI with YAML load/save
"""
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import os

import importlib.util
from importlib.metadata import version, PackageNotFoundError

try:
    MODBUS_VERSION = version("pymodbus")
    MODBUS_AVAILABLE = MODBUS_VERSION == "3.5.4"
except PackageNotFoundError:
    MODBUS_VERSION = None
    MODBUS_AVAILABLE = False


CONFIG_FILE = "startup_config.yaml"


def load_yaml_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to load config YAML: {e}")
    return {}


def save_yaml_config(config):
    try:
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        print(f"Failed to save config YAML: {e}")



def startup_config_gui():
    root = tk.Tk()
    root.title("AI Classifier Control - Startup Settings")
    root.geometry("700x500")
    root.resizable(False, False)

    # 🔹 Load defaults from YAML
    yaml_config = load_yaml_config()

    config = {}

    camera_var = tk.StringVar(value=yaml_config.get("CAMERA_TYPE", "Basler"))
    connection_var = tk.StringVar(value=yaml_config.get("CONNECTION_TYPE", "MODBUS"))
    model_path_var = tk.StringVar(value=yaml_config.get("MODEL_PATH", "Modells/yolo26n.onnx"))
    pfs_path_var = tk.StringVar(value=yaml_config.get("PFS_PATH", "Camera_Settings/Kamerasettings_Kiramet.pfs"))
    verbose_var = tk.BooleanVar(value=yaml_config.get("MODEL_VERBOSE", False))
    scaleable_ui = tk.BooleanVar(value=False)
    video_path_var = tk.StringVar(value=yaml_config.get("VIDEO_PATH", ""))
    fps_var = tk.StringVar(value=str(yaml_config.get("FPS", 30)))
    usb_config_path_var = tk.StringVar(value=yaml_config.get("USB_CAMERA_SETTINGS_PATH", "sensor_tc_camera_settings.yaml"))
    mvimpact_config_path_var = tk.StringVar(value=yaml_config.get("MVIMPACT_NIR_SETTINGS_PATH", "mvimpact_nir_camera_settings.yaml"))
    nir_classifier_path_var = tk.StringVar(value=yaml_config.get("NIR_CLASSIFIER_PATH", ""))
    nir_classifier_kind_var = tk.StringVar(value=yaml_config.get("NIR_CLASSIFIER_KIND", "SAM_PLACEHOLDER"))
    
            
    def open_usb_camera_configurator():
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "USB_Camera_Configrator.py"
        )
    
        if not os.path.exists(script_path):
            messagebox.showerror(
                "Missing file",
                f"Could not find:\n{script_path}"
            )
            return
        subprocess.Popen([sys.executable, script_path])
        
    def browse_model():
        path = filedialog.askopenfilename(
            title="Select YOLO model",
            filetypes=[("Model files", "*.pt *.onnx *.engine"), ("All files", "*.*")]
        )
        if path:
            model_path_var.set(path)

    def browse_pfs():
        path = filedialog.askopenfilename(
            title="Select Basler PFS file",
            filetypes=[("PFS files", "*.pfs"), ("All files", "*.*")]
        )
        if path:
            pfs_path_var.set(path)
    
    def browse_usb_config():
        path = filedialog.askopenfilename(
            title="Select USB camera settings YAML",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if path:
            usb_config_path_var.set(path)        
    
    def browse_video():
        path = filedialog.askopenfilename(
            title="Select simulation video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            video_path_var.set(path)

    def browse_mvimpact_config():
        path = filedialog.askopenfilename(
            title="Select mvImpact NIR camera settings YAML",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if path:
            mvimpact_config_path_var.set(path)

    def browse_nir_classifier():
        path = filedialog.askopenfilename(
            title="Select NIR scikit-learn classifier pipeline",
            filetypes=[
                ("scikit-learn pipeline files", "*.pkl *.pickle *.joblib"),
                ("All files", "*.*"),
            ]
        )
        if path:
            nir_classifier_path_var.set(path)

    def on_camera_change(*args):
        if camera_var.get() == "Basler":
            pfs_entry.configure(state="normal")
            pfs_button.configure(state="normal")            
        else:
            pfs_entry.configure(state="disabled")
            pfs_button.configure(state="disabled")
    
        if camera_var.get() == "SIMULATED":
            video_entry.configure(state="normal")
            video_button.configure(state="normal")
        else:
            video_entry.configure(state="disabled")
            video_button.configure(state="disabled")

        if camera_var.get() == "USB":
            usb_config_entry.configure(state="normal")
            usb_config_button.configure(state="normal")
            usb_camera_config_button.configure(state="normal")
        else:
            usb_config_entry.configure(state="disabled")
            usb_config_button.configure(state="disabled")
            usb_camera_config_button.configure(state="disabled")

        if camera_var.get() == "MVIMPACT_NIR":
            # Check if mvImpact is actually installed and tell the user that it will fall back on synthetic NIR if mvImpact is NOT installed in the current environment
            if importlib.util.find_spec("mvIMPACT") is None:
                messagebox.showwarning(
                    "mvIMPACT not installed",
                    "The 'mvIMPACT' package is not installed in this environment.\n\n"
                    "The camera will fall back to the synthetic NIR placeholder.\n\n"
                    "Install mvIMPACT Acquire and its Python bindings to use the real NIR camera."
                )
            mvimpact_config_entry.configure(state="normal")
            mvimpact_config_button.configure(state="normal")
        else:
            mvimpact_config_entry.configure(state="disabled")
            mvimpact_config_button.configure(state="disabled")
            
        if camera_var.get() in ("Basler"):
            fps_entry.configure(state="normal")
        else:
            fps_var.set("0")
            fps_entry.configure(state="disabled")

        if camera_var.get() == "MVIMPACT_NIR":
            model_entry.configure(state="disabled")
            model_button.configure(state="disabled")
            verbose_check.configure(state="disabled")
            nir_classifier_entry.configure(state="normal")
            nir_classifier_button.configure(state="normal")
            nir_classifier_kind.configure(state="readonly")
        else:
            model_entry.configure(state="normal")
            model_button.configure(state="normal")
            verbose_check.configure(state="normal")
            nir_classifier_entry.configure(state="disabled")
            nir_classifier_button.configure(state="disabled")
            nir_classifier_kind.configure(state="disabled")

    def on_ok():
        camera_type = camera_var.get()
        connection_type = connection_var.get()
        model_path = model_path_var.get().strip()
        pfs_path = pfs_path_var.get().strip()
        usb_config_path = usb_config_path_var.get().strip()
        mvimpact_config_path = mvimpact_config_path_var.get().strip()
        nir_classifier_path = nir_classifier_path_var.get().strip()
        nir_classifier_kind = nir_classifier_kind_var.get().strip()
        
        if camera_type == "USB" and not usb_config_path:
            messagebox.showerror("Invalid input", "Please select a USB camera settings YAML file.")
            return

        if camera_type.upper() not in ("USB", "BASLER", "SIMULATED", "MVIMPACT_NIR"):
            messagebox.showerror("Invalid input", "CAMERA_TYPE must be USB, Basler, SIMULATED, or MVIMPACT_NIR.")
            return

        if connection_type.upper() not in ("UDP", "SERIAL", "MODBUS", "SIMULATED"):
            messagebox.showerror("Invalid input", "CONNECTION_TYPE must be UDP, SERIAL, or MODBUS.")
            return

        if camera_type != "MVIMPACT_NIR" and not model_path:
            messagebox.showerror("Invalid input", "Please select a model file.")
            return

        if camera_type == "Basler" and not pfs_path:
            messagebox.showerror("Invalid input", "Please select a Basler PFS file.")
            return
        
        video_path = video_path_var.get().strip()

        if camera_type == "SIMULATED" and not video_path:
            messagebox.showerror("Invalid input", "Please select a simulation video.")
            return

        if camera_type == "MVIMPACT_NIR" and not mvimpact_config_path:
            messagebox.showerror("Invalid input", "Please select an mvImpact NIR camera settings YAML file.")
            return

        # The NIR classifier is optional at startup because smart-camera/classified
        # mode can already deliver width x 1 class lines. Spectral mode can later
        # require this path when classification is enabled in the NIR pipeline.

        if camera_type in ("Basler"):
            try:
                fps_value = int(float(fps_var.get().strip()))
            except ValueError:
                messagebox.showerror("Invalid input", "Camera FPS must be a number.")
                return
            if fps_value <= 0:
                messagebox.showerror("Invalid input", "Camera FPS must be greater than 0.")
                return
        else:
            fps_value = 30

        config["CAMERA_TYPE"] = camera_type
        config["CONNECTION_TYPE"] = connection_type
        config["MODEL_PATH"] = model_path
        config["PFS_PATH"] = pfs_path
        config["MODEL_VERBOSE"] = verbose_var.get()
        config["VIDEO_PATH"] = video_path
        config["SCALEABLE_UI"] = scaleable_ui.get()
        config["FPS"] = fps_value
        config["USB_CAMERA_SETTINGS_PATH"] = usb_config_path
        config["MVIMPACT_NIR_SETTINGS_PATH"] = mvimpact_config_path
        config["NIR_CLASSIFIER_PATH"] = nir_classifier_path
        config["NIR_CLASSIFIER_KIND"] = nir_classifier_kind

        # 🔹 Save config to YAML
        save_yaml_config(config)

        root.destroy()

    def on_cancel():
        root.destroy()
        raise SystemExit("Startup cancelled by user.")

    root.columnconfigure(1, weight=1)

    ttk.Label(root, text="Camera Type:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    ttk.Combobox(
        root,
        textvariable=camera_var,
        values=["USB", "Basler", "SIMULATED", "MVIMPACT_NIR"],
        state="readonly",
        width=20
    ).grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    usb_camera_config_button = ttk.Button(
        root,
        text="USB Camera Config",
        command=open_usb_camera_configurator
    )
    usb_camera_config_button.grid(row=0, column=2, padx=10, pady=10, sticky="ew")
    ttk.Label(root, text="Connection Type:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    connection_combo = ttk.Combobox(
        root,
        textvariable=connection_var,
        values=["UDP", "SERIAL", "MODBUS", "SIMULATED"],
        state="readonly",
        width=20
    )
    connection_combo.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
    
    if not MODBUS_AVAILABLE:
        messagebox.showwarning(
            "pymodbus not installed",
            "The required 'pymodbus' package is not installed or has the wrong version.\n\n"
            "MODBUS nozzle control has been disabled.\n\n"
            "Stop the program, install the correct version with:\n\n"
            "pip install pymodbus==3.5.4\n\n"
            "Then restart the program."
        )
    
        connection_combo.configure(values=["UDP", "SERIAL", "SIMULATED"])
    
        if connection_var.get().upper() == "MODBUS":
            connection_var.set("SIMULATED")
    

    ttk.Label(root, text="Model Path:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
    model_entry = ttk.Entry(root, textvariable=model_path_var)
    model_entry.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
    model_button = ttk.Button(root, text="Browse...", command=browse_model)
    model_button.grid(row=2, column=2, padx=10, pady=10)

    ttk.Label(root, text="Basler PFS Path:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
    pfs_entry = ttk.Entry(root, textvariable=pfs_path_var)
    pfs_entry.grid(row=3, column=1, padx=10, pady=10, sticky="ew")
    pfs_button = ttk.Button(root, text="Browse...", command=browse_pfs)
    pfs_button.grid(row=3, column=2, padx=10, pady=10)
    
    ttk.Label(root, text="USB Settings Path:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
    usb_config_entry = ttk.Entry(root, textvariable=usb_config_path_var)
    usb_config_entry.grid(row=4, column=1, padx=10, pady=10, sticky="ew")
    usb_config_button = ttk.Button(root, text="Browse...", command=browse_usb_config)
    usb_config_button.grid(row=4, column=2, padx=10, pady=10)
        
    ttk.Label(root, text="Simulation Video:").grid(row=5, column=0, padx=10, pady=10, sticky="w")
    video_entry = ttk.Entry(root, textvariable=video_path_var)
    video_entry.grid(row=5, column=1, padx=10, pady=10, sticky="ew")
    video_button = ttk.Button(root, text="Browse...", command=browse_video)
    video_button.grid(row=5, column=2, padx=10, pady=10)

    ttk.Label(root, text="mvImpact NIR Settings:").grid(row=6, column=0, padx=10, pady=8, sticky="w")
    mvimpact_config_entry = ttk.Entry(root, textvariable=mvimpact_config_path_var)
    mvimpact_config_entry.grid(row=6, column=1, padx=10, pady=8, sticky="ew")
    mvimpact_config_button = ttk.Button(root, text="Browse...", command=browse_mvimpact_config)
    mvimpact_config_button.grid(row=6, column=2, padx=10, pady=8)

    ttk.Label(root, text="NIR Classifier Type:").grid(row=7, column=0, padx=10, pady=8, sticky="w")
    nir_classifier_kind = ttk.Combobox(
        root,
        textvariable=nir_classifier_kind_var,
        values=["SAM_PLACEHOLDER", "SKLEARN_PIPELINE", "SMART_CAMERA_CLASSIFIED", "SYNTHETIC_SKLEARN"],
        state="readonly",
        width=24,
    )
    nir_classifier_kind.grid(row=7, column=1, padx=10, pady=8, sticky="ew")

    ttk.Label(root, text="NIR Classifier Path:").grid(row=8, column=0, padx=10, pady=8, sticky="w")
    nir_classifier_entry = ttk.Entry(root, textvariable=nir_classifier_path_var)
    nir_classifier_entry.grid(row=8, column=1, padx=10, pady=8, sticky="ew")
    nir_classifier_button = ttk.Button(root, text="Browse...", command=browse_nir_classifier)
    nir_classifier_button.grid(row=8, column=2, padx=10, pady=8)

    ttk.Label(root, text="Camera FPS:").grid(row=9, column=0, padx=10, pady=8, sticky="w")
    fps_entry = ttk.Entry(root, textvariable=fps_var)
    fps_entry.grid(row=9, column=1, padx=10, pady=8, sticky="ew")

    verbose_check = ttk.Checkbutton(root, text="Verbose YOLO output", variable=verbose_var)
    verbose_check.grid(row=10, column=1, padx=10, pady=8, sticky="w")
    
    ttk.Checkbutton(root, text="Scalable UI (experimental)", variable=scaleable_ui).grid(
        row=10, column=2, padx=10, pady=8, sticky="w"
    )

    button_frame = ttk.Frame(root)
    button_frame.grid(row=11, column=0, columnspan=3, pady=18)
    
    ttk.Button(button_frame, text="Start", command=on_ok).pack(side="left", padx=10)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="left", padx=10)
    
    camera_var.trace_add("write", on_camera_change)
    on_camera_change()

    root.mainloop()
    return config

if __name__ == "__main__":
    print("Hellow")
    startup_config_gui()
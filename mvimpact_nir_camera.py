# -*- coding: utf-8 -*-
"""
Matrix Vision / Balluff mvIMPACT Acquire NIR camera adapter with an automatic
synthetic placeholder fallback for SenSoRTC.

The adapter intentionally mirrors cv2.VideoCapture enough for SenSoRTC:
    camera = MvImpactNIRCamera(...).connect()
    ret, frame = camera.read()
    camera.release()

NIR concept:
    input_mode: spectral
        one acquired NIR sample is width x spectral_depth, e.g. 312 x 220.
        A scikit-learn/joblib classifier is used when available. If the selected
        classifier is missing, invalid, incompatible, or raises during predict,
        a deterministic placeholder SAM-like classifier is used instead.

    input_mode: classified
        one acquired NIR sample is already width x 1, e.g. 312 x 1.
        No classifier is used; the line is appended directly to the rolling
        buffer.

read() returns the rolling classified NIR buffer as a uint8 image with shape:
    (rolling_height, width, 3) when convert_to_bgr/synthetic_as_bgr is true
    (rolling_height, width)    otherwise

The most recent class-label line is also available as:
    camera.last_classified_line
"""

import ctypes
import os
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml
import threading
import tkinter as tk
from tkinter import messagebox

class PlaceholderSAMClassifier:
    """Deterministic SAM-style fallback classifier for NIR spectral data.

    Class 0 is always background. For classes 1..N this classifier can use real
    material reference spectra loaded from PE/PET/PP .mat files. Background is
    decided in raw ALU space:

        mean(spectrum) < background_threshold -> class 0

    SAM angle comparison is only applied to non-background spectra after vector
    normalisation.
    """

    def __init__(
        self,
        n_classes: int = 4,
        background_threshold: float = 300.0,
        reference_spectra: Optional[np.ndarray] = None,
        reference_labels: Optional[np.ndarray] = None,
    ):
        self.n_classes = max(1, int(n_classes))
        self.background_threshold = float(background_threshold)
        self.reference_spectra = None
        self.reference_labels = None
        self._references_by_depth: Dict[int, np.ndarray] = {}
        self._reference_labels_by_depth: Dict[int, np.ndarray] = {}

        if reference_spectra is not None:
            refs = np.asarray(reference_spectra, dtype=np.float32)
            if refs.ndim == 1:
                refs = refs.reshape(1, -1)
            refs = refs - np.min(refs, axis=1, keepdims=True)
            norms = np.linalg.norm(refs, axis=1, keepdims=True) + 1e-8
            self.reference_spectra = refs / norms
            if reference_labels is None:
                reference_labels = np.arange(1, refs.shape[0] + 1, dtype=np.uint8)
            self.reference_labels = np.asarray(reference_labels, dtype=np.uint8).reshape(-1)
            if self.reference_labels.size != refs.shape[0]:
                self.reference_labels = np.arange(1, refs.shape[0] + 1, dtype=np.uint8)
            self.n_classes = max(self.n_classes, int(self.reference_labels.max()) + 1)

    def _fallback_references(self, spectral_depth: int) -> Tuple[np.ndarray, np.ndarray]:
        spectral_depth = max(1, int(spectral_depth))
        if spectral_depth in self._references_by_depth:
            return self._references_by_depth[spectral_depth], self._reference_labels_by_depth[spectral_depth]

        bands = np.linspace(0.0, 1.0, spectral_depth, dtype=np.float32)
        refs = []
        labels = []
        for cls in range(1, max(2, self.n_classes)):
            centre = 0.20 + 0.65 * ((cls - 1) / max(1, self.n_classes - 2))
            width = 0.055 + 0.018 * cls
            peak = np.exp(-((bands - centre) ** 2) / (2.0 * width ** 2))
            shoulder = 0.45 * np.exp(-((bands - min(0.95, centre + 0.18)) ** 2) / (2.0 * (width * 1.8) ** 2))
            refs.append(0.35 + peak + shoulder)
            labels.append(cls)

        ref = np.asarray(refs, dtype=np.float32)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8
        label_arr = np.asarray(labels, dtype=np.uint8)
        self._references_by_depth[spectral_depth] = ref
        self._reference_labels_by_depth[spectral_depth] = label_arr
        return ref, label_arr

    def _references(self, spectral_depth: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.reference_spectra is None:
            return self._fallback_references(spectral_depth)

        refs = self.reference_spectra
        labels = self.reference_labels
        if refs.shape[1] == spectral_depth:
            return refs, labels

        # Interpolate references to the incoming spectral depth.
        x_old = np.linspace(0.0, 1.0, refs.shape[1])
        x_new = np.linspace(0.0, 1.0, spectral_depth)
        resized = np.vstack([np.interp(x_new, x_old, row) for row in refs]).astype(np.float32)
        resized = resized - np.min(resized, axis=1, keepdims=True)
        resized /= np.linalg.norm(resized, axis=1, keepdims=True) + 1e-8
        return resized, labels

    def predict(self, spectral_frame: np.ndarray) -> np.ndarray:
        s = np.asarray(spectral_frame, dtype=np.float32)
        if s.ndim != 2:
            s = s.reshape((s.shape[0], -1))

        mean_all = s.mean(axis=1)
        labels = np.zeros((s.shape[0],), dtype=np.uint8)
        foreground = mean_all >= self.background_threshold
        if not np.any(foreground):
            return labels

        refs, ref_labels = self._references(s.shape[1])
        s_fg = s[foreground]
        s_fg = s_fg - np.min(s_fg, axis=1, keepdims=True)
        s_norm = s_fg / (np.linalg.norm(s_fg, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.clip(s_norm @ refs.T, -1.0, 1.0)
        angles = np.arccos(cos_sim)
        labels[foreground] = ref_labels[np.argmin(angles, axis=1)]
        return labels.astype(np.uint8) % max(1, self.n_classes)

class MvImpactNIRCamera:
    def __init__(
        self,
        settings_path: Optional[str] = None,
        fps: float = 30.0,
        classifier_path: Optional[str] = None,
        classifier_kind: str = "SAM_PLACEHOLDER",
    ):
        self.settings_path = settings_path
        self.settings = self._load_settings(settings_path)

        self.source = str(self.settings.get("source", "auto")).lower()
        self.fallback_to_synthetic = bool(self.settings.get("fallback_to_synthetic", True))
        
        
        # Recording Settings and Flags and counters
        self.raw_recording_fast_mode = False
        self.line_counter = 0
        self.display_decimation = 4
        
        # NIR logical dimensions.
        self.input_mode = str(
            self.settings.get("input_mode", self.settings.get("output_mode", "spectral"))
        ).lower()
        self.width = int(self.settings.get("width", 312))
        self.spectral_depth = int(self.settings.get("spectral_depth", self.settings.get("height", 220)))
        self.rolling_height = int(self.settings.get("rolling_height", 220))
        self.classified_height = 1

        # Classifier settings.
        self.classifier_path = classifier_path or self.settings.get("classifier_path", "")
        self.classifier_kind = str(classifier_kind or self.settings.get("classifier_kind", "SAM_PLACEHOLDER"))
        self.confidence_threshold = 0.0
        self.background_label = 0
        self.not_classified_label = None
        self.class_names = []
        self.background_threshold = self._coerce_background_threshold(
            self.settings.get("background_threshold", self.settings.get("sam_background_threshold", 300.0))
        )
        self.synthetic_material_paths = self.settings.get("synthetic_material_paths", {}) or {}
        self.synthetic_material_min_mean = float(self.settings.get("synthetic_material_min_mean", self.background_threshold))
        self.synthetic_material_samples_per_class = int(self.settings.get("synthetic_material_samples_per_class", 256))
        self.synthetic_material_templates = {}
        self.synthetic_material_references = None
        self.synthetic_material_reference_labels = None
        self.synthetic_object_use_mean_spectra = bool(self.settings.get("synthetic_object_use_mean_spectra", True))

        # Synthetic placeholder behaviour.
        self.synthetic_speed_px = float(self.settings.get("synthetic_speed_px", 4.0))
        self.synthetic_noise = float(self.settings.get("synthetic_noise", 8.0))
        self.synthetic_seed = int(self.settings.get("synthetic_seed", 1234))
        self.synthetic_classes = int(self.settings.get("synthetic_classes", self.settings.get("classes", 4)))
        self.synthetic_period = int(self.settings.get("synthetic_period", 96))
        self.synthetic_as_bgr = bool(self.settings.get("synthetic_as_bgr", True))
        self.roll_direction = str(self.settings.get("roll_direction", "up")).lower()

        # Synthetic object generator.  Unlike the old periodic/check-pattern
        # placeholder, this creates finite PE/PET/PP objects that persist across
        # multiple line-scan rows, producing distinct blobs in the rolling buffer.
        self.synthetic_object_mode = bool(self.settings.get("synthetic_object_mode", True))
        self.synthetic_object_spawn_probability = float(self.settings.get("synthetic_object_spawn_probability", 0.18))
        self.synthetic_object_max_active = int(self.settings.get("synthetic_object_max_active", 7))
        self.synthetic_object_min_width = int(self.settings.get("synthetic_object_min_width", 12))
        self.synthetic_object_max_width = int(self.settings.get("synthetic_object_max_width", 55))
        self.synthetic_object_min_length = int(self.settings.get("synthetic_object_min_length", 28))
        self.synthetic_object_max_length = int(self.settings.get("synthetic_object_max_length", 95))
        self.synthetic_object_edge_margin = int(self.settings.get("synthetic_object_edge_margin", 10))
        self._synthetic_objects = []

        # Real mvIMPACT settings.
        self.fps = float(self.settings.get("fps", fps))
        self.device_index = int(self.settings.get("device_index", 0))
        self.serial = self.settings.get("serial", None)
        self.timeout_ms = int(self.settings.get("timeout_ms", 1000))
        self.crop_size = int(self.settings.get("crop_size", 0))
        self.crop_offset_x = int(self.settings.get("offset_x", 0))
        self.crop_offset_y = int(self.settings.get("offset_y", 0))
        self.convert_to_bgr = bool(self.settings.get("convert_to_bgr", True))
        self.request_count = int(self.settings.get("request_count", 4))
        self.feature_writes: Dict[str, Any] = self.settings.get("features", {}) or {}

        # Acquisition-stall recovery.  mvIMPACT can occasionally stop returning
        # valid request numbers without the device being physically gone.  A
        # request-queue restart is much cheaper than a full device reconnect and
        # does not reload the sklearn joblib.
        self.timeout_recovery_enabled = bool(self.settings.get("timeout_recovery_enabled", True))
        self.read_retry_count = int(self.settings.get("read_retry_count", 1))
        self.min_acquisition_restart_interval_s = float(self.settings.get("min_acquisition_restart_interval_s", 0.25))
        self._read_timeout_failures = 0
        self._request_not_ok_failures = 0
        self._acquisition_restart_count = 0
        self._last_acquisition_restart_time = 0.0
        self._last_acquisition_error = ""

        self.acquire = None
        self.dev_mgr = None
        self.device = None
        self.fi = None
        self._opened = False
        self._using_synthetic = False
            
        self._frame_idx = 0
        self._last_frame_time = 0.0

        # Live diagnostics exposed to SenSoRTC/UI.
        self.lps = 0.0
        self.line_period_ms = 0.0
        self.classification_ms = 0.0
        self._last_line_time = 0.0

        self._rng = np.random.default_rng(self.synthetic_seed)
        self._rolling_buffer = np.zeros((max(1, self.rolling_height), max(1, self.width)), dtype=np.uint8)
        self._rolling_class_buffer = np.zeros((max(1, self.rolling_height), max(1, self.width)), dtype=np.uint8)
        self._classifier = None
        self._classifier_name = "none"
        # Consecutive and lifetime classifier failures.  A predict failure is
        # not a camera acquisition failure; failed lines are emitted as background.
        self._classifier_predict_failures = 0
        self._classifier_predict_failures_total = 0
        self._last_classifier_error = ""
        self._last_classifier_error_line = -1
        self.last_classified_line = np.zeros((max(1, self.width),), dtype=np.uint8)
        self.line_counter = 0
        # Last native line/block kept for raw NIR recording.
        # spectral mode: shape (width, spectral_depth), raw ALU/dtype where possible
        # classified mode: shape (width,), class IDs
        self.last_native_sample = None
        self.last_spectral_sample = None
        self.last_raw_record_sample = None

    @staticmethod
    def _load_settings(path: Optional[str]) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("camera", data)
        except Exception as exc:
            print(f"[mvImpact NIR] Failed to load settings YAML: {exc}")
            return {}

    @staticmethod
    def _coerce_background_threshold(value) -> float:
        """Return background threshold in raw ALU units.

        Older configs used 0..1. If such a value is encountered, interpret it as
        a fraction of a 0..2000 ALU range to avoid silently disabling
        background filtering.
        """
        try:
            threshold = float(value)
        except Exception:
            return 300.0
        if 0.0 < threshold <= 1.0:
            return threshold * 2000.0
        return threshold

    def set_background_threshold(self, value) -> None:
        """Update SAM background threshold live in raw ALU units.

        The GUI control is now in ALU. For compatibility, values in 0..1 are
        still interpreted as a fraction of 2000 ALU.
        """
        self.background_threshold = self._coerce_background_threshold(value)
        clf = getattr(self, "_classifier", None)
        if hasattr(clf, "background_threshold"):
            clf.background_threshold = self.background_threshold


    def get_diagnostics(self) -> Dict[str, Any]:
        """Return lightweight live diagnostics for SenSoRTC/UI status pills."""
        return {
            "classifier_name": str(self._classifier_name),
            "classifier_predict_failures": int(self._classifier_predict_failures),
            "classifier_predict_failures_total": int(self._classifier_predict_failures_total),
            "last_classifier_error": str(self._last_classifier_error),
            "last_classifier_error_line": int(self._last_classifier_error_line),
            "line_counter": int(self.line_counter),
            "lps": float(self.lps),
            "line_period_ms": float(self.line_period_ms),
            "classification_ms": float(self.classification_ms),
            "read_timeout_failures": int(self._read_timeout_failures),
            "request_not_ok_failures": int(self._request_not_ok_failures),
            "acquisition_restart_count": int(self._acquisition_restart_count),
            "last_acquisition_error": str(self._last_acquisition_error),
        }


    @staticmethod
    def _coerce_confidence_threshold(value) -> float:
        try:
            value = float(value)
        except Exception:
            return 0.0
        if not np.isfinite(value):
            return 0.0
        value = float(np.clip(value, 0.0, 1.0))
        # Match the UI precision: anything displayed as 0.00 is OFF.
        return 0.0 if value <= 0.005 else value

    def set_confidence_threshold(self, value) -> None:
        """Update the loaded sklearn confidence rejector live.

        UI confidence 0.00 means OFF.  Internally we pass -1.0, not 0.0, so
        rejectors implemented as either `conf < threshold` or `conf <= threshold`
        cannot create Not-classified pixels.
        """
        self.confidence_threshold = self._coerce_confidence_threshold(value)
        threshold_to_apply = -1.0 if self.confidence_threshold <= 0.0 else self.confidence_threshold

        clf = getattr(self, "_classifier", None)
        if clf is None:
            return
        if hasattr(clf, "set_threshold"):
            try:
                clf.set_threshold(threshold_to_apply)
            except Exception:
                pass
        if hasattr(clf, "threshold"):
            try:
                clf.threshold = threshold_to_apply
            except Exception:
                pass
        if hasattr(clf, "confidence_threshold"):
            try:
                clf.confidence_threshold = threshold_to_apply
            except Exception:
                pass

    def _suppress_nc_when_confidence_off(self, labels: np.ndarray) -> np.ndarray:
        """Remove explicit Not-classified labels when confidence reject is OFF."""
        if self.not_classified_label is None:
            return labels
        if float(getattr(self, "confidence_threshold", 0.0)) > 0.0:
            return labels

        labels = np.asarray(labels).copy()
        labels[labels == int(self.not_classified_label)] = int(self.background_label)
        return labels


    def _load_synthetic_material_templates(self):
        """Load PE/PET/PP material spectra from .mat files, excluding background.

        Expected .mat variable: imnData with shape width x spectral_depth x samples.
        Background spectra are excluded by mean intensity before any templates or
        references are built, so the fake camera never draws from background
        spectra contained in the material files.
        """
        if self.synthetic_material_templates:
            return

        default_paths = {
            "PE": "PE.mat",
            "PET": "PET.mat",
            "PP": "PP.mat",
        }
        configured = dict(default_paths)
        configured.update(self.synthetic_material_paths or {})

        label_map = {"PE": 1, "PET": 2, "PP": 3}
        references = []
        reference_labels = []

        try:
            from scipy.io import loadmat
        except Exception as exc:
            print(f"[mvImpact NIR] scipy is unavailable, using fallback synthetic spectra: {exc}")
            return

        base_dir = os.path.dirname(os.path.abspath(self.settings_path)) if self.settings_path else os.getcwd()
        for name, rel_path in configured.items():
            label = label_map.get(str(name).upper())
            if label is None:
                continue

            path = str(rel_path)
            candidates = [path]
            if not os.path.isabs(path):
                candidates.append(os.path.join(base_dir, path))
                candidates.append(os.path.join(os.getcwd(), path))
                candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))

            mat_path = next((c for c in candidates if os.path.exists(c)), None)
            if mat_path is None:
                print(f"[mvImpact NIR] Synthetic material file not found for {name}: {rel_path}")
                continue

            try:
                data = loadmat(mat_path)
                arr = data.get("imnData")
                if arr is None:
                    keys = [k for k in data.keys() if not k.startswith("__")]
                    arr = data[keys[0]] if keys else None
                if arr is None:
                    raise ValueError("no array found in .mat file")

                spectra = self._mat_array_to_spectra(arr)
                mean_values = spectra.mean(axis=1)
                spectra = spectra[mean_values >= self.synthetic_material_min_mean]
                if spectra.size == 0:
                    print(
                        f"[mvImpact NIR] All {name} spectra were filtered as background "
                        f"at mean >= {self.synthetic_material_min_mean:.1f} ALU."
                    )
                    continue

                if spectra.shape[0] > self.synthetic_material_samples_per_class:
                    idx = self._rng.choice(spectra.shape[0], self.synthetic_material_samples_per_class, replace=False)
                    spectra = spectra[idx]

                self.synthetic_material_templates[label] = spectra.astype(np.float32, copy=False)
                ref = np.mean(spectra, axis=0).astype(np.float32)
                references.append(ref)
                reference_labels.append(label)
                print(
                    f"[mvImpact NIR] Loaded {spectra.shape[0]} foreground {name} spectra "
                    f"from {os.path.basename(mat_path)} as class {label}."
                )
            except Exception as exc:
                print(f"[mvImpact NIR] Failed to load {name} spectra from {mat_path}: {exc}")

        if references:
            self.synthetic_material_references = np.asarray(references, dtype=np.float32)
            self.synthetic_material_reference_labels = np.asarray(reference_labels, dtype=np.uint8)
            self.synthetic_classes = max(self.synthetic_classes, int(self.synthetic_material_reference_labels.max()) + 1)

    def _mat_array_to_spectra(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            # Common shape in the supplied files: width x spectral_depth x samples.
            if arr.shape[1] == self.spectral_depth:
                spectra = arr.transpose(0, 2, 1).reshape(-1, arr.shape[1])
            elif arr.shape[0] == self.spectral_depth:
                spectra = arr.transpose(1, 2, 0).reshape(-1, arr.shape[0])
            else:
                spectra = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 2:
            if arr.shape[1] == self.spectral_depth:
                spectra = arr
            elif arr.shape[0] == self.spectral_depth:
                spectra = arr.T
            else:
                spectra = arr.reshape(-1, arr.shape[-1])
        else:
            spectra = arr.reshape(-1, self.spectral_depth)

        spectra = np.asarray(spectra, dtype=np.float32)
        if spectra.shape[1] != self.spectral_depth:
            x_old = np.linspace(0.0, 1.0, spectra.shape[1])
            x_new = np.linspace(0.0, 1.0, self.spectral_depth)
            spectra = np.vstack([np.interp(x_new, x_old, row) for row in spectra]).astype(np.float32)
        return spectra

    
    
    @staticmethod
    def show_error_box(error_text: str):
        def worker():
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showerror("Import Error", error_text)
            root.destroy()
        worker()   # call directly so it's visible and blocks; see note below

    def connect(self):        
        self._prepare_classifier()
        
        if self.classifier_kind.upper() == "SYNTHETIC_SKLEARN":
            return self._connect_synthetic("configured synthetic source")

        if self.source in ("synthetic", "placeholder", "simulated"):
            return self._connect_synthetic("configured synthetic source")

        try:
            
            from mvIMPACT import acquire
                
            self.acquire = acquire
            self.dev_mgr = acquire.DeviceManager()
            device_count = int(self.dev_mgr.deviceCount()) if hasattr(self.dev_mgr, "deviceCount") else 0
            if device_count <= 0:
                raise RuntimeError("No mvIMPACT Acquire devices found.")

            self.device = self._select_device()
            self.device.open()
            self._apply_basic_settings()
            self.fi = acquire.FunctionInterface(self.device)
            self._queue_initial_requests()
            self._opened = True
            self._using_synthetic = False

            name = self._safe_read_attr(self.device, "product") or "mvIMPACT device"
            serial = self._safe_read_attr(self.device, "serial") or "unknown serial"
            print(f"[Process-1] mvImpact NIR camera connected: {name} ({serial})")
            print(f"[Process-1] NIR classifier active: {self._classifier_name}")
            return self

        except Exception as exc:
            if self.fallback_to_synthetic or self.source == "auto":
                return self._connect_synthetic(f"mvIMPACT unavailable: {exc}")
            raise

    def _prepare_classifier(self):
        if self.input_mode in ("classified", "classified_line", "classified_line_312x1", "smart"):
            self._classifier = None
            self._classifier_name = "not used; camera delivers classified lines"
            return

        self._classifier = None
        if self.classifier_path:
            try:
                import joblib
                loaded = joblib.load(self.classifier_path)
                if isinstance(loaded, dict):
                    self.class_names = loaded.get("class_names", []) or []
                    self.background_label = int(loaded.get("background_label", 0))
                    ncl = loaded.get("not_classified_label", loaded.get("reject_label", None))
                    if ncl is None and self.class_names:
                        ncl = len(self.class_names) - 1
                    self.not_classified_label = None if ncl is None else int(ncl)
                    loaded = loaded["pipeline"]
                if not hasattr(loaded, "predict"):
                    raise TypeError("loaded object has no predict(...) method")
                self._classifier = loaded
                self.set_confidence_threshold(self.confidence_threshold)
                self._classifier_name = f"joblib:{self.classifier_path}"
                self._classifier_predict_failures = 0
                self._last_classifier_error = ""
                self._last_classifier_error_line = -1
                return
            except Exception as exc:
                print(f"[mvImpact NIR] Could not load classifier {self.classifier_path!r}: {exc}")
                print("[mvImpact NIR] Falling back to PlaceholderSAMClassifier.")

        # Only load PE/PET/PP template files when the placeholder classifier is
        # actually needed.  A valid joblib should not do .mat IO on every camera
        # reconnect; that slows recovery and adds unnecessary failure points.
        self._load_synthetic_material_templates()
        self._classifier = PlaceholderSAMClassifier(
            n_classes=self.synthetic_classes,
            background_threshold=self.background_threshold,
            reference_spectra=self.synthetic_material_references,
            reference_labels=self.synthetic_material_reference_labels,
        )
        self._classifier_name = "PlaceholderSAMClassifier"
        self._classifier_predict_failures = 0
        self._last_classifier_error = ""
        self._last_classifier_error_line = -1

    def _connect_synthetic(self, reason: str):
        self._opened = True
        self._using_synthetic = True
        self._rolling_buffer.fill(0)
        self._rolling_class_buffer.fill(0)
        print(
            f"[Process-1] Synthetic NIR placeholder connected ({reason}). "
            f"input_mode={self.input_mode}, spectral={self.width}x{self.spectral_depth}, "
            f"rolling_buffer={self.rolling_height}x{self.width}"
        )
        print(f"[Process-1] NIR classifier active: {self._classifier_name}")
        return self

    def _select_device(self):
        if self.serial:
            for idx in range(int(self.dev_mgr.deviceCount())):
                dev = self.dev_mgr.getDevice(idx)
                if self._safe_read_attr(dev, "serial") == str(self.serial):
                    return dev
            raise RuntimeError(f"No mvIMPACT device with serial {self.serial!r} found.")
        return self.dev_mgr.getDevice(self.device_index)

    @staticmethod
    def _safe_read_attr(obj, attr_name: str):
        try:
            attr = getattr(obj, attr_name)
            return attr.read() if hasattr(attr, "read") else attr
        except Exception:
            return None

    def _write_property(self, owner, name: str, value: Any) -> bool:
        try:
            prop = getattr(owner, name)
            if hasattr(prop, "write"):
                prop.write(value)
            elif hasattr(prop, "SetValue"):
                prop.SetValue(value)
            else:
                return False
            return True
        except Exception as exc:
            print(f"[mvImpact NIR] Could not set {name}={value!r}: {exc}")
            return False

    def _apply_basic_settings(self):
        acquire = self.acquire
        try:
            ss = acquire.SystemSettings(self.device)
            if hasattr(ss, "requestCount"):
                ss.requestCount.write(self.request_count)
        except Exception as exc:
            print(f"[mvImpact NIR] Could not set requestCount: {exc}")

        
        for name, value in self.feature_writes.items():
            for owner_factory in (
                lambda: acquire.AcquisitionControl(self.device),
                lambda: acquire.ImageDestination(self.device),
                lambda: acquire.CameraSettingsBase(self.device),
            ):
                try:
                    if self._write_property(owner_factory(), name, value):
                        break
                except Exception:
                    continue

    def _queue_initial_requests(self):
        if self.fi is None:
            return
        acquire = self.acquire
        for _ in range(max(1, self.request_count)):
            result = self.fi.imageRequestSingle()
            if result != acquire.DMR_NO_ERROR:
                break
        try:
            self.fi.acquisitionStart()
        except Exception:
            pass

    def restart_acquisition_queue(self):
        if self.fi is None:
            return
    
        try:
            self.fi.acquisitionStop()
        except Exception:
            pass
    
        try:
            self.fi.imageRequestReset(0, 0)
        except Exception:
            pass
    
        for _ in range(max(1, self.request_count)):
            try:
                self.fi.imageRequestSingle()
            except Exception:
                pass
    
        try:
            self.fi.acquisitionStart()
        except Exception:
            pass

    def _soft_restart_acquisition_queue(self, reason: str) -> bool:
        """Restart mvIMPACT acquisition/request queue without reopening device."""
        if not self.timeout_recovery_enabled or self.fi is None:
            return False

        now = time.monotonic()
        if now - self._last_acquisition_restart_time < self.min_acquisition_restart_interval_s:
            return False

        self._last_acquisition_restart_time = now
        self._acquisition_restart_count += 1
        self._last_acquisition_error = str(reason)
        print(
            f"[mvImpact NIR] Soft acquisition queue restart "
            f"#{self._acquisition_restart_count}: {reason}"
        )
        self.restart_acquisition_queue()
        return True


    def _note_line_acquired(self) -> None:
        """Update line counter and live LPS estimate after one successful line."""
        now = time.perf_counter()

        if self._last_line_time > 0.0:
            dt = now - self._last_line_time
            if dt > 0.0:
                inst_lps = 1.0 / dt
                inst_period_ms = dt * 1000.0

                if self.lps <= 0.0:
                    self.lps = inst_lps
                    self.line_period_ms = inst_period_ms
                else:
                    # Smooth enough for UI readability, responsive enough for stalls.
                    alpha = 0.10
                    self.lps = (1.0 - alpha) * self.lps + alpha * inst_lps
                    self.line_period_ms = (1.0 - alpha) * self.line_period_ms + alpha * inst_period_ms

        self._last_line_time = now
        self.line_counter += 1

    def read(self):
        if not self._opened:
            return False, None

        if self._using_synthetic:
            self._pace_fps()
            native = self._next_synthetic_native_sample()
            self._note_line_acquired()
            frame = self._process_native_sample(native)
            return True, self._format_output(frame)
        # Function ends here if no real mvIMPACT camera is available

        if self.fi is None:
            return False, None

        acquire = self.acquire
        max_attempts = max(1, int(self.read_retry_count) + 1)

        for attempt in range(max_attempts):
            request_nr = self.fi.imageRequestWaitFor(self.timeout_ms)
            if not self.fi.isRequestNrValid(request_nr):
                self._read_timeout_failures += 1
                self._last_acquisition_error = "imageRequestWaitFor timeout/invalid request number"
                restarted = self._soft_restart_acquisition_queue(self._last_acquisition_error)
                if restarted and attempt + 1 < max_attempts:
                    continue
                return False, None

            request = self.fi.getRequest(request_nr)
            request_requeued = False

            def unlock_and_requeue_request():
                nonlocal request_requeued
                if request_requeued:
                    return
                try:
                    request.unlock()
                except Exception:
                    try:
                        self.fi.imageRequestUnlock(request_nr)
                    except Exception:
                        pass
                try:
                    self.fi.imageRequestSingle()
                except Exception:
                    pass
                request_requeued = True

            try:
                is_ok_attr = getattr(request, "isOK", False)
                is_ok = bool(is_ok_attr() if callable(is_ok_attr) else is_ok_attr)
                if not is_ok and hasattr(request, "requestResult"):
                    is_ok = request.requestResult.read() == acquire.rrOK
                if not is_ok:
                    self._request_not_ok_failures += 1
                    self._last_acquisition_error = "mvIMPACT request not OK"
                    unlock_and_requeue_request()
                    restarted = self._soft_restart_acquisition_queue(self._last_acquisition_error)
                    if restarted and attempt + 1 < max_attempts:
                        continue
                    return False, None

                native = self._request_to_numpy(request)
                native = self._normalise_native(native)

                # _request_to_numpy() copies the mvIMPACT buffer.  Requeue the
                # hardware request before classification/display work so Python,
                # sklearn, GC, or UI spikes do not starve the acquisition queue.
                unlock_and_requeue_request()

                self._read_timeout_failures = 0
                self._request_not_ok_failures = 0
                self._note_line_acquired()

                # Divert if camera is in recording mode to circumvent classification and its CPU load
                if getattr(self, "raw_recording_fast_mode", False):

                    raw = np.asarray(native)

                    self.last_native_sample = raw.copy()
                    self.last_raw_record_sample = raw.copy()

                    if raw.shape == (self.width, self.spectral_depth):
                        spectral = raw
                    elif raw.shape == (self.spectral_depth, self.width):
                        spectral = raw.T
                    else:
                        spectral = self._coerce_spectral_sample(raw)

                    self.last_spectral_sample = spectral.copy()

                    if self.line_counter % self.display_decimation == 0:
                        mean_line = spectral.mean(axis=1)
                        gray_line = np.clip(mean_line / 4095.0 * 255.0, 0, 255).astype(np.uint8)
                        empty_classes = np.zeros((self.width,), dtype=np.uint8)
                        frame = self._append_line_to_rolling_buffer(gray_line, empty_classes)
                    else:
                        frame = self._rolling_buffer

                    return True, self._format_output(frame)

                # If not recording proceed with classification route
                frame = self._process_native_sample(native)
                return True, self._format_output(frame)
            finally:
                unlock_and_requeue_request()

        return False, None

    def _pace_fps(self):
        return

    def _next_synthetic_native_sample(self) -> np.ndarray:
        self._frame_idx += 1
        if self.input_mode in ("classified", "classified_line", "classified_line_312x1", "smart"):
            return self._synthetic_smart_camera_labels()
        return self._synthetic_spectral_sample()

    def _synthetic_material_curve(self, label: int, count: int) -> np.ndarray:
        """Fallback foreground spectra when PE/PET/PP .mat files are absent."""
        bands = np.linspace(0.0, 1.0, self.spectral_depth, dtype=np.float32)
        centre = {1: 0.28, 2: 0.55, 3: 0.78}.get(int(label), 0.5)
        shoulder = min(0.95, centre + 0.16)
        curve = (
            260.0
            + 850.0 * np.exp(-((bands - centre) ** 2) / (2.0 * 0.055 ** 2))
            + 360.0 * np.exp(-((bands - shoulder) ** 2) / (2.0 * 0.095 ** 2))
        )
        return np.tile(curve[None, :], (int(count), 1)).astype(np.float32)

    def _spawn_synthetic_object(self):
        if self.synthetic_classes <= 1:
            return
        margin = max(0, min(self.synthetic_object_edge_margin, self.width // 3))
        min_w = max(1, min(self.synthetic_object_min_width, self.width))
        max_w = max(min_w, min(self.synthetic_object_max_width, self.width))
        min_l = max(1, self.synthetic_object_min_length)
        max_l = max(min_l, self.synthetic_object_max_length)
        half_width = int(self._rng.integers(min_w, max_w + 1)) / 2.0
        lo = int(max(margin + half_width, 0))
        hi = int(min(self.width - margin - half_width, self.width - 1))
        if hi <= lo:
            lo, hi = 0, self.width - 1
        obj = {
            "label": int(self._rng.integers(1, max(2, min(self.synthetic_classes, 4)))),
            "center": float(self._rng.uniform(lo, hi + 1)),
            "half_width": float(half_width),
            "length": int(self._rng.integers(min_l, max_l + 1)),
            "age": 0,
            "wiggle_phase": float(self._rng.uniform(0.0, 2.0 * np.pi)),
            "wiggle_amp": float(self._rng.uniform(-2.5, 2.5)),
        }
        self._synthetic_objects.append(obj)

    def _synthetic_object_labels_line(self) -> np.ndarray:
        """Create one classified line containing finite foreground objects."""
        labels = np.zeros((self.width,), dtype=np.uint8)

        if (
            self.synthetic_object_mode
            and len(self._synthetic_objects) < max(1, self.synthetic_object_max_active)
            and self._rng.random() < self.synthetic_object_spawn_probability
        ):
            self._spawn_synthetic_object()

        x = np.arange(self.width, dtype=np.float32)
        alive = []
        for obj in self._synthetic_objects:
            length = max(1, int(obj["length"]))
            age = int(obj["age"])
            if age >= length:
                continue

            # Elliptical object envelope in scan direction: narrow at the first
            # and last lines, widest near the object centre.
            t = (age + 0.5) / float(length)
            envelope = max(0.0, 1.0 - ((t - 0.5) / 0.5) ** 2) ** 0.5
            half_width = max(1.0, float(obj["half_width"]) * envelope)
            center = float(obj["center"]) + float(obj["wiggle_amp"]) * np.sin(2.0 * np.pi * t + float(obj["wiggle_phase"]))
            mask = np.abs(x - center) <= half_width
            labels[mask] = int(obj["label"])

            obj["age"] = age + 1
            if obj["age"] < length:
                alive.append(obj)
        self._synthetic_objects = alive
        return labels

    def _synthetic_spectral_sample(self) -> np.ndarray:
        """Return exact width x spectral_depth data using PE/PET/PP object spectra.

        Class 0 is generated as low-intensity background. Classes 1..3 are
        finite foreground objects sampled only from spectra whose mean ALU is
        above synthetic_material_min_mean, so background spectra from the .mat
        files are never used as PE/PET/PP examples.
        """
        self._load_synthetic_material_templates()

        # Low-intensity belt/background. Keep it below the background threshold.
        #bg_level = max(0.0, min(self.background_threshold * 0.15, self.background_threshold - 1000.0))
        bg_level = 300
        sample = self._rng.normal(
            loc=bg_level,
            scale=max(1.0, self.synthetic_noise),
            size=(self.width, self.spectral_depth),
        ).astype(np.float32)

        labels = self._synthetic_object_labels_line()
        for label in range(1, max(1, self.synthetic_classes)):
            indices = np.where(labels == label)[0]
            if indices.size == 0:
                continue

            templates = self.synthetic_material_templates.get(label)
            if templates is None or templates.size == 0:
                spectra = self._synthetic_material_curve(label, indices.size)
            elif self.synthetic_object_use_mean_spectra:
                # Use the foreground class mean for cleaner, object-like blobs.
                # This still derives the spectral shape from PE/PET/PP .mat
                # foreground spectra, but avoids pixel-wise template changes that
                # look like salt-and-pepper classifier noise in the rolling view.
                class_mean = np.mean(templates, axis=0).astype(np.float32)
                spectra = np.tile(class_mean[None, :], (indices.size, 1))
            else:
                pick = self._rng.integers(0, templates.shape[0], size=indices.size)
                spectra = templates[pick].astype(np.float32, copy=True)

            gain = self._rng.normal(1.0, 0.020, size=(indices.size, 1)).astype(np.float32)
            offset = self._rng.normal(0.0, max(1.0, self.synthetic_noise), size=(indices.size, self.spectral_depth)).astype(np.float32)
            sample[indices, :] = spectra * gain + offset

        return np.clip(sample, 0, 4095).astype(np.uint16)

    def _synthetic_smart_camera_labels(self) -> np.ndarray:
        # Direct 312 x 1 classified output, as if the smart NIR camera has
        # already classified the belt pixels. Uses the same finite object
        # generator as spectral mode, but skips spectra/SAM classification.
        return self._synthetic_object_labels_line() % max(1, self.synthetic_classes)

    def _process_native_sample(self, native: np.ndarray) -> np.ndarray:
        self.last_native_sample = np.asarray(native).copy()
        if self.input_mode in ("classified", "classified_line", "classified_line_312x1", "smart"):
            labels = self._coerce_classified_labels(native)
            self.last_spectral_sample = None
            self.last_raw_record_sample = np.asarray(labels, dtype=np.uint8).reshape(-1).copy()
        else:
            spectral = self._coerce_spectral_sample(native)
            labels = self._classify_spectral_sample(spectral)
            self.last_spectral_sample = np.asarray(spectral).copy()
            # Preserve the native raw ALU dtype for NIR raw recording whenever possible.
            raw_native = np.asarray(native)
            if raw_native.shape == (self.width, self.spectral_depth):
                self.last_raw_record_sample = raw_native.copy()
            elif raw_native.shape == (self.spectral_depth, self.width):
                self.last_raw_record_sample = raw_native.T.copy()
            else:
                self.last_raw_record_sample = self.last_spectral_sample.copy()

        self.last_classified_line = np.asarray(labels, dtype=np.uint8).reshape(-1)
        #self.line_counter += 1 # line counter increment moved to read()
        intensity_line = self._labels_to_intensity_line(self.last_classified_line)
        return self._append_line_to_rolling_buffer(intensity_line, self.last_classified_line)

    def _coerce_spectral_sample(self, sample: np.ndarray) -> np.ndarray:
        s = np.asarray(sample)
        if s.ndim == 3:
            s = s[:, :, 0]
        # Preserve raw ALU values for spectral classification/recording.
        if s.ndim == 1:
            s = np.tile(s.reshape(-1, 1), (1, self.spectral_depth))

        # Accept either width x spectral_depth or spectral_depth x width.
        if s.shape == (self.width, self.spectral_depth):
            return s.astype(np.float32, copy=False)
        if s.shape == (self.spectral_depth, self.width):
            return s.T.astype(np.float32, copy=False)

        resized = cv2.resize(s.astype(np.float32), (self.spectral_depth, self.width), interpolation=cv2.INTER_LINEAR)
        return resized.astype(np.float32, copy=False)

    def _coerce_classified_labels(self, sample: np.ndarray) -> np.ndarray:
        arr = np.asarray(sample)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        arr = arr.reshape(-1)
        if arr.size != self.width:
            arr = cv2.resize(arr[None, :].astype(np.uint8), (self.width, 1), interpolation=cv2.INTER_NEAREST)[0]

        # Smart cameras may return either class labels 0..N or intensity-coded
        # labels 0..255. If values exceed n_classes, convert from intensity.
        arr = arr.astype(np.uint8, copy=False)
        if arr.size and int(arr.max()) >= max(8, self.synthetic_classes + 1):
            denom = max(1, self.synthetic_classes - 1)
            arr = np.rint(arr.astype(np.float32) * denom / 255.0).astype(np.uint8)
        return arr % max(1, self.synthetic_classes)

    def _classify_spectral_sample(self, spectral: np.ndarray) -> np.ndarray:
        spectral = np.asarray(spectral, dtype=np.float32)

        # 1) Background pre-filter in raw ALU space
        mean_alu = spectral.mean(axis=1)
        foreground = mean_alu >= float(self.background_threshold)
    
        labels = np.zeros((spectral.shape[0],), dtype=np.uint8)
    
        # Nothing bright enough -> all background
        if not np.any(foreground):
            return labels
        
        # 2) Only send foreground spectra through sklearn pipeline
        if self._classifier is None:
            self._prepare_classifier()
        try:
            t0 = time.perf_counter()
            labels[foreground] = self._classifier.predict(spectral[foreground])
            dt_ms = (time.perf_counter() - t0) * 1000.0

            self._classifier_predict_failures = 0

            if self.classification_ms <= 0.0:
                self.classification_ms = dt_ms
            else:
                alpha = 0.10
                self.classification_ms = (1.0 - alpha) * self.classification_ms + alpha * dt_ms

            labels = np.asarray(labels).reshape(-1)
            if labels.size != self.width:
                labels = cv2.resize(labels[None, :].astype(np.uint8), (self.width, 1), interpolation=cv2.INTER_NEAREST)[0]
            labels = self._suppress_nc_when_confidence_off(labels)
            #return labels.astype(np.uint8) % max(1, self.synthetic_classes)
            return labels.astype(np.uint8)
        except Exception as exc:
            self._classifier_predict_failures += 1
            self._classifier_predict_failures_total += 1
            self._last_classifier_error = str(exc)
            self._last_classifier_error_line = int(self.line_counter)
            if self._classifier_predict_failures <= 3 or self._classifier_predict_failures % 100 == 0:
                print(
                    f"[mvImpact NIR] Classifier predict failed "
                    f"({self._classifier_predict_failures}x, total={self._classifier_predict_failures_total}): {exc}"
                )

            # Safe failure mode: this line is background.
            # Do not silently switch classifier type during live sorting.
            return labels.astype(np.uint8)
            

    def _labels_to_intensity_line(self, labels: np.ndarray) -> np.ndarray:
        denom = max(1, self.synthetic_classes - 1)
        return np.round(np.asarray(labels, dtype=np.float32) * (255.0 / denom)).astype(np.uint8)

    def _append_line_to_rolling_buffer(self, line: np.ndarray, class_line: Optional[np.ndarray] = None) -> np.ndarray:
        line = np.asarray(line, dtype=np.uint8).reshape(-1)
        if line.size != self.width:
            line = cv2.resize(line[None, :], (self.width, 1), interpolation=cv2.INTER_NEAREST)[0]

        if class_line is None:
            class_line = self.last_classified_line
        class_line = np.asarray(class_line, dtype=np.uint8).reshape(-1)
        if class_line.size != self.width:
            class_line = cv2.resize(class_line[None, :], (self.width, 1), interpolation=cv2.INTER_NEAREST)[0]

        if self.roll_direction in ("down", "bottom"):
            self._rolling_buffer[:-1, :] = self._rolling_buffer[1:, :]
            self._rolling_buffer[-1, :] = line
            self._rolling_class_buffer[:-1, :] = self._rolling_class_buffer[1:, :]
            self._rolling_class_buffer[-1, :] = class_line
        else:
            self._rolling_buffer[1:, :] = self._rolling_buffer[:-1, :]
            self._rolling_buffer[0, :] = line
            self._rolling_class_buffer[1:, :] = self._rolling_class_buffer[:-1, :]
            self._rolling_class_buffer[0, :] = class_line
        return self._rolling_buffer.copy()

    @property
    def rolling_class_buffer(self) -> np.ndarray:
        return self._rolling_class_buffer.copy()

    def _format_output(self, frame: np.ndarray) -> np.ndarray:
        if self.synthetic_as_bgr or self.convert_to_bgr:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def _request_to_numpy(self, request):
        width = int(request.imageWidth.read())
        height = int(request.imageHeight.read())
        channels = int(request.imageChannelCount.read())
        bit_depth = int(request.imageChannelBitDepth.read())
        image_size = int(request.imageSize.read())
        address = int(request.imageData.read())

        cbuf = (ctypes.c_char * image_size).from_address(address)
        dtype = np.uint16 if bit_depth > 8 else np.uint8
        arr = np.frombuffer(cbuf, dtype=dtype).copy()

        expected = width * height * max(1, channels)
        if arr.size < expected:
            raise RuntimeError(f"mvIMPACT image buffer too small: got {arr.size}, expected {expected}")
        arr = arr[:expected]

        if channels <= 1:
            return arr.reshape((height, width))
        return arr.reshape((height, width, channels))

    def _normalise_native(self, frame):
        # Preserve raw NIR ALU values for spectral classification and raw recording.
        return frame

    def release(self):
        self._opened = False
        try:
            if self.fi is not None:
                try:
                    self.fi.acquisitionStop()
                except Exception:
                    pass
                try:
                    self.fi.imageRequestReset(0, 0)
                except Exception:
                    pass
        finally:
            try:
                if self.device is not None:
                    self.device.close()
            except Exception:
                pass
            self.fi = None
            self.device = None
            self.dev_mgr = None
            self._using_synthetic = False

    def isOpened(self):
        return self._opened

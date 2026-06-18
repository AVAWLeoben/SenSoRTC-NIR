# -*- coding: utf-8 -*-
"""
AI Classifier Control Software - spawn-safe startup-config version.

Drop-in notes:
- No multiprocessing child process depends on settings created only inside __main__.
- The producer owns its local model, camera and mask history state.
- VERT_MOVEMENT is a multiprocessing.Value and can be changed from the UI.
- Ultralytics model.track() can be enabled from the UI to estimate vertical movement.
  The estimate is displayed but is NOT applied automatically.
"""

# %% Imports
import multiprocessing
import threading
import queue
from queue import Full, Empty
import os
import time
import cv2
import numpy as np

from pypylon import pylon

from ultralytics import YOLO

import yaml
import datetime 

from UI_LAYER import display

from MODEL_HOTSWAP import load_detection_model, names_to_dict

from NOZZLE_CONTROL_LAYER import (
    nozzle_control_UDP,
    nozzle_control_ARDUINO,
    nozzle_control_MODBUS,
    nozzle_control_SIMULATED
)

from collections import deque
movement_history = deque(maxlen=30)

import platform
import subprocess

# ---------------------------------------------------------------------------
# Persistent runtime settings helpers
# ---------------------------------------------------------------------------
RUNTIME_CONFIG_FILE = "runtime_settings.yaml"

def load_runtime_config():
    if os.path.exists(RUNTIME_CONFIG_FILE):
        try:
            with open(RUNTIME_CONFIG_FILE, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to load runtime settings: {e}")
    return {}


def _to_safe_yaml_data(value):
    """Convert multiprocessing/list-proxy/numpy-ish values into safe YAML types."""
    if isinstance(value, dict):
        return {str(k): _to_safe_yaml_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_safe_yaml_data(v) for v in value]
    try:
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def save_runtime_config(config):
    try:
        with open(RUNTIME_CONFIG_FILE, "w") as f:
            yaml.safe_dump(_to_safe_yaml_data(config), f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Failed to save runtime settings: {e}")



# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def reverseString(string):
    return string[::-1]


def connectCamera(camera_type, pfs_path, FPS, rerun=False, VIDEO_PATH=None, USB_SETTINGS_PATH=None, MVIMPACT_NIR_SETTINGS_PATH=None, NIR_CLASSIFIER_PATH=None, NIR_CLASSIFIER_KIND="SAM_PLACEHOLDER"):
    camera_type = camera_type.lower()

    if camera_type == "basler":
        return connectCamera_BASLER(pfs_path, FPS, rerun)

    elif camera_type == "usb":
        return connectCamera_USB(settings_path=USB_SETTINGS_PATH)

    elif camera_type == "simulated":
        return connectCamera_Simulated(VIDEO_PATH)

    elif camera_type in ("mvimpact_nir", "mvimpact-nir", "nir"):
        return connectCamera_MVIMPACT_NIR(
            settings_path=MVIMPACT_NIR_SETTINGS_PATH,
            fps=0,
            classifier_path=NIR_CLASSIFIER_PATH,
            classifier_kind=NIR_CLASSIFIER_KIND,
        )

    else:
        raise ValueError(
            f"Unsupported CAMERA_TYPE: {camera_type}. Must be USB, Basler, SIMULATED, or MVIMPACT_NIR."
        )


def connectCamera_BASLER(pfs_path, FPS, rerun=False):
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        print("[Process-1] No camera devices found.")
        return None

    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
    camera.Open()
    pylon.FeaturePersistence.Load(pfs_path, camera.GetNodeMap())
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera_fps = float(FPS.value if hasattr(FPS, "value") else FPS)
    camera.AcquisitionFrameRate.SetValue(camera_fps)
    try:
        FPS.value = int(round(camera.AcquisitionFrameRate.GetValue()))
    except Exception:
        pass
    print(f"[Process-1] Basler camera FPS set to {camera_fps:.1f}")
    try: # To fix Pixel_Format to BGR8 for downstream consistency with cv2, pygame
        camera.PixelFormat.SetValue("BGR8")
    except Exception as e:
        print(f"Error when trying to set PixelFormat=BGR8. {e}")
        print("Check correct Pixel Format in Pylon Viewer")
    return camera


def connectCamera_USB(rerun=False, camera_index=0, width=2560, height=1440, fps=30, settings_path=None):
    settings = load_usb_camera_settings(settings_path)
    cam_cfg = settings.get("camera", {})
    fmt = cam_cfg.get("format", {})
    controls = cam_cfg.get("controls", {})

    camera_index = int(cam_cfg.get("index", camera_index))
    width = int(fmt.get("width", width))
    height = int(fmt.get("height", height))
    fps = float(fmt.get("fps", fps))
    fourcc = fmt.get("fourcc", "MJPG")

    backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_DSHOW
    camera = cv2.VideoCapture(camera_index, backend)

    if not camera.isOpened():
        print(f"[Process-1] No USB camera found at index {camera_index}.")
        return None

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    time.sleep(0.3)

    if platform.system() == "Linux":
        device = cam_cfg.get("device_linux", f"/dev/video{camera_index}")
        apply_usb_settings_linux(device, controls)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print(f"[Process-1] USB camera connected: {actual_width}x{actual_height} @ {actual_fps:.1f} fps")
    print(f"[Process-1] USB settings loaded from: {settings_path}")

    return camera

def connectCamera_Simulated(VIDEO_PATH,fps=30):
    from simulated_camera import simulated_camera
    return simulated_camera(fps=fps,VIDEO_PATH=VIDEO_PATH).connect()

def connectCamera_MVIMPACT_NIR(settings_path=None, fps=30, classifier_path=None, classifier_kind="SAM_PLACEHOLDER"):
    from mvimpact_nir_camera import MvImpactNIRCamera
    return MvImpactNIRCamera(
        settings_path=settings_path,
        fps=fps,
        classifier_path=classifier_path,
        classifier_kind=classifier_kind,
    ).connect()

def load_usb_camera_settings(path):
    if not path or not os.path.exists(path):
        print(f"[USB] Settings YAML not found: {path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[USB] Failed to load USB settings YAML: {e}")
        return {}


def apply_usb_settings_linux(device, controls):
    for name, meta in controls.items():
        if not isinstance(meta, dict) or "value" not in meta:
            continue

        subprocess.run(
            ["v4l2-ctl", "-d", device, "-c", f"{name}={int(meta['value'])}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

def center_crop_square(img, size=640, offset_x=0, offset_y=0):
    if img is None:
        raise ValueError("Input image is None")

    h, w = img.shape[:2]
    if h < size or w < size:
        raise ValueError(f"Image too small for {size}x{size} crop: got {w}x{h}")

    cx = w // 2 + int(offset_x)
    cy = h // 2 + int(offset_y)
    half = size // 2

    x0 = max(0, min(w - size, cx - half))
    y0 = max(0, min(h - size, cy - half))

    return img[y0:y0 + size, x0:x0 + size]


def prepare_nir_frame_for_pipeline(img, output_size=640):
    """
    The placeholder NIR camera can intentionally produce native sensor-shaped
    data such as 312x220 frames or 312x1 classified lines.  The rest of this
    application expects image-like frames, so this helper creates a square
    visual/inference image while preserving the camera's native aspect/pattern.
    """
    if img is None:
        raise ValueError("Input NIR image is None")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    interpolation = cv2.INTER_NEAREST if h <= 4 else cv2.INTER_LINEAR

    if h <= 4:
        # Classified 312x1 lines are repeated vertically first so they become
        # visible and safe for model/display processing.
        img = np.repeat(img, 220, axis=0)
        h, w = img.shape[:2]
        interpolation = cv2.INTER_NEAREST

    scale = float(output_size) / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((output_size, output_size, 3), dtype=resized.dtype)
    y0 = (output_size - new_h) // 2
    x0 = (output_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Mask generation and vertical movement calibration
# ---------------------------------------------------------------------------

# Upper bound for hot-swappable models. TARGET_CLASSES is a fixed-size shared
# array, so it is allocated once at this size and only the first
# N_CLASSES.value entries are meaningful for the currently loaded model.
MAX_MODEL_CLASSES = 256


def publish_model_info(MODEL_INFO, names, kind, path, status):
    """Publish the active detection model to the UI process (manager dict)."""
    if MODEL_INFO is None:
        return
    try:
        MODEL_INFO["names"] = {int(k): str(v) for k, v in dict(names).items()}
        MODEL_INFO["kind"] = str(kind)
        MODEL_INFO["path"] = str(path)
        MODEL_INFO["status"] = str(status)
        MODEL_INFO["generation"] = int(MODEL_INFO.get("generation", 0)) + 1
    except Exception as e:
        print(f"[Process-1] Could not publish model info: {e}")



def is_nir_camera_type(camera_type):
    return str(camera_type).lower() in ("mvimpact_nir", "mvimpact-nir", "nir")


def load_nir_camera_settings(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("camera", data)
    except Exception as e:
        print(f"[NIR] Failed to load NIR settings YAML: {e}")
        return {}


def make_empty_nozzle_mask(n_nozzles):
    return np.zeros((640, int(n_nozzles)), dtype=np.uint8)


NIR_RAW_CHUNK_LINES = 6000


def _nir_raw_mode(camera):
    mode = str(getattr(camera, "input_mode", "nir")).lower()
    if mode in ("classified", "classified_line", "classified_line_312x1", "smart"):
        return "classified"
    return "spectral"


def _get_nir_raw_record_sample(camera):
    sample = getattr(camera, "last_raw_record_sample", None)
    if sample is None:
        sample = getattr(camera, "last_spectral_sample", None)
    if sample is None:
        sample = getattr(camera, "last_classified_line", None)
    if sample is None:
        return None
    return np.asarray(sample).copy()


def _flush_nir_raw_chunk(state, output_dir, reason="chunk"):
    if not state["buffer"]:
        return

    os.makedirs(output_dir, exist_ok=True)
    data = np.stack(state["buffer"], axis=0)
    
    
    # Convert from:
    # (lines, width, bands)
    # to:
    # (width, bands, lines)
    
    if data.ndim == 3:
        data = np.transpose(data, (1, 2, 0))
    
    first_ts = state["wall_start"] or datetime.datetime.now()
    last_ts = datetime.datetime.now()
    first_str = first_ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    last_str = last_ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    mode = state.get("mode") or "nir"
    chunk_idx = int(state.get("chunk_index", 0))

    base = f"nir_raw_{mode}_{first_str}_to_{last_str}_chunk{chunk_idx:05d}_{reason}"
    data_path = os.path.join(output_dir, base + ".npy")
    np.save(data_path, data)
    print(f"Saved NIR raw chunk: {data_path} shape={data.shape} dtype={data.dtype}")

    state["buffer"].clear()
    state["wall_start"] = None
    state["chunk_index"] = chunk_idx + 1


def _append_nir_raw_line(state, camera, output_dir, chunk_lines=NIR_RAW_CHUNK_LINES):
    sample = _get_nir_raw_record_sample(camera)
    if sample is None:
        return

    mode = _nir_raw_mode(camera)
    if state.get("mode") != mode:
        _flush_nir_raw_chunk(state, output_dir, reason="modechange")
        state["mode"] = mode

    if state["wall_start"] is None:
        state["wall_start"] = datetime.datetime.now()

    state["buffer"].append(sample)
    if len(state["buffer"]) >= int(chunk_lines):
        _flush_nir_raw_chunk(state, output_dir, reason="full")


DEFAULT_NIR_CLASS_COLORS = [
    (0, 0, 0),        # Background
    (255, 64, 64),    # Class 1
    (64, 220, 64),    # Class 2
    (64, 128, 255),   # Class 3
    (255, 220, 64),
    (220, 64, 255),
    (64, 220, 220),
    (255, 140, 64),
]


def normalise_nir_class_colors(raw_colors, n_classes):
    colors = []
    if isinstance(raw_colors, (list, tuple)):
        for item in raw_colors:
            try:
                if isinstance(item, str):
                    item = item.strip().lstrip('#')
                    if len(item) == 6:
                        rgb = tuple(int(item[i:i+2], 16) for i in (0, 2, 4))
                    else:
                        continue
                else:
                    rgb = tuple(int(v) for v in item[:3])
                colors.append(tuple(max(0, min(255, v)) for v in rgb))
            except Exception:
                continue

    while len(colors) < int(n_classes):
        colors.append(DEFAULT_NIR_CLASS_COLORS[len(colors) % len(DEFAULT_NIR_CLASS_COLORS)])
    return colors[:int(n_classes)]



# UI displays confidence with two decimals. Treat anything that displays as
# 0.00 as a real OFF state, not as a tiny positive reject threshold.
NIR_CONF_OFF_EPS = 0.005


def _normalise_confidence_threshold(value):
    try:
        value = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(value):
        return 0.0
    value = float(np.clip(value, 0.0, 1.0))
    return 0.0 if value <= NIR_CONF_OFF_EPS else value


def set_nir_classifier_confidence_threshold(camera, threshold):
    """Apply the live NIR confidence threshold wherever the adapter stores it.

    The mvImpact NIR adapter has changed shape during development: sometimes
    camera._classifier is the ConfidenceRejectingClassifier, sometimes it is a
    small wrapper/dict that contains the real sklearn pipeline.  A direct
    `camera._classifier.threshold = ...` therefore silently misses some loaded
    joblibs.  This helper walks the common wrapper attributes and updates every
    object that exposes either `set_threshold`, `threshold`,
    `set_confidence_threshold`, or `confidence_threshold`.
    """
    threshold = _normalise_confidence_threshold(threshold)

    # GUI 0.00 means OFF. Use -1 internally so even classifiers/adapters that
    # implement `conf <= threshold` cannot create reject-label pixels.
    threshold_to_apply = -1.0 if threshold <= 0.0 else threshold

    seen = set()
    stack = []
    applied = False

    def push(obj):
        if obj is None:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        stack.append(obj)

    # Do not apply a generic `.threshold` to the camera object itself; that
    # could collide with camera/background settings.  Only use explicit
    # confidence APIs on the adapter, then walk the classifier/pipeline objects.
    if hasattr(camera, "set_confidence_threshold"):
        try:
            camera.set_confidence_threshold(threshold_to_apply)
            applied = True
        except Exception:
            pass
    if hasattr(camera, "confidence_threshold"):
        try:
            camera.confidence_threshold = threshold_to_apply
            applied = True
        except Exception:
            pass

    for attr in (
        "_classifier", "classifier", "_pipeline", "pipeline",
        "_model", "model", "pipe", "estimator",
    ):
        push(getattr(camera, attr, None))

    while stack:
        obj = stack.pop()

        if isinstance(obj, dict):
            for key in (
                "pipeline", "classifier", "model", "estimator",
                "wrapped", "pipe", "sklearn_pipeline",
            ):
                push(obj.get(key))
            continue

        if hasattr(obj, "set_threshold"):
            try:
                obj.set_threshold(threshold_to_apply)
                applied = True
            except Exception:
                pass
        if hasattr(obj, "threshold"):
            try:
                obj.threshold = threshold_to_apply
                applied = True
            except Exception:
                pass
        if hasattr(obj, "set_confidence_threshold"):
            try:
                obj.set_confidence_threshold(threshold_to_apply)
                applied = True
            except Exception:
                pass
        if hasattr(obj, "confidence_threshold"):
            try:
                obj.confidence_threshold = threshold_to_apply
                applied = True
            except Exception:
                pass

        # sklearn Pipeline support: also inspect nested/final steps.
        try:
            if hasattr(obj, "steps"):
                for _, step in obj.steps:
                    push(step)
        except Exception:
            pass
        try:
            if hasattr(obj, "named_steps"):
                for step in obj.named_steps.values():
                    push(step)
        except Exception:
            pass

        for attr in (
            "_classifier", "classifier", "_pipeline", "pipeline",
            "_model", "model", "pipe", "estimator", "wrapped",
            "sklearn_pipeline",
        ):
            try:
                push(getattr(obj, attr, None))
            except Exception:
                pass

    return applied


def set_nir_background_threshold(camera, threshold):
    """Apply the NIR intensity/background threshold to the camera/classifier stack.

    This is separate from confidence threshold handling.  The GUI THRESHOLD
    value in NIR mode means raw spectrum intensity cutoff:
        mean(spectrum) < threshold -> Background class 0

    Reconnecting the mvImpact camera creates a fresh adapter/classifier object,
    so this value must be pushed again immediately after every connect/reconnect
    and classifier hotswap, even when the GUI value has not changed.
    """
    try:
        threshold = float(threshold)
    except Exception:
        return False
    if not np.isfinite(threshold):
        return False

    seen = set()
    stack = []
    applied = False

    def push(obj):
        if obj is None:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        stack.append(obj)

    push(camera)

    while stack:
        obj = stack.pop()

        if isinstance(obj, dict):
            for key in (
                "pipeline", "classifier", "model", "estimator",
                "wrapped", "pipe", "sklearn_pipeline",
            ):
                push(obj.get(key))
            continue

        if hasattr(obj, "set_background_threshold"):
            try:
                obj.set_background_threshold(threshold)
                applied = True
            except Exception:
                pass

        if hasattr(obj, "background_threshold"):
            try:
                obj.background_threshold = threshold
                applied = True
            except Exception:
                pass

        # sklearn Pipeline support: inspect nested/final steps as well.
        try:
            if hasattr(obj, "steps"):
                for _, step in obj.steps:
                    push(step)
        except Exception:
            pass
        try:
            if hasattr(obj, "named_steps"):
                for step in obj.named_steps.values():
                    push(step)
        except Exception:
            pass

        for attr in (
            "_classifier", "classifier", "_pipeline", "pipeline",
            "_model", "model", "pipe", "estimator", "wrapped",
            "sklearn_pipeline",
        ):
            try:
                push(getattr(obj, attr, None))
            except Exception:
                pass

    return applied


def clear_nir_classification_history(camera):
    """Clear rolling classified display/history after threshold changes.

    Otherwise old Not-classified pixels remain visible until they scroll out of
    the rolling line buffer, which makes the live threshold control look broken.
    """
    for attr in (
        "rolling_class_buffer", "class_buffer", "classified_buffer",
        "_rolling_class_buffer", "_class_buffer", "last_classified_line",
    ):
        try:
            arr = getattr(camera, attr, None)
            if isinstance(arr, np.ndarray):
                arr.fill(0)
        except Exception:
            pass


def colorize_nir_class_buffer(class_buffer, class_colors):
    """Colorize class IDs without converting invalid IDs to Not classified.

    The old implementation used np.clip(..., 0, n_colors - 1).  That makes any
    invalid value such as 255, -1, or NaN become the last class color.  In a
    normal NIR bundle the last class is "Not classified", so uninitialised or
    adapter-side ignore pixels looked like real NC pixels even when the
    confidence threshold was 0.0.

    Only exact valid class IDs are colorized as classes.  Everything outside
    0..n_classes-1 is drawn as Background.
    """
    buf = np.asarray(class_buffer)
    if buf.ndim == 3:
        buf = buf[:, :, 0]

    colors = np.asarray(list(class_colors), dtype=np.uint8)
    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] == 0:
        colors = np.asarray(DEFAULT_NIR_CLASS_COLORS, dtype=np.uint8)

    buf_f = np.asarray(buf, dtype=np.float32)
    valid = np.isfinite(buf_f) & (buf_f >= 0) & (buf_f < colors.shape[0])

    # Default invalid/uninitialised/ignore pixels to Background (class 0),
    # never to the last class / Not classified.
    idx = np.zeros(buf_f.shape, dtype=np.intp)
    idx[valid] = buf_f[valid].astype(np.intp)

    rgb = colors[idx]
    # Downstream display code assumes OpenCV BGR and converts to RGB.
    return rgb[:, :, ::-1].copy()


def nir_class_id_counts(class_buffer, max_items=8):
    """Small debug summary for checking whether NC is real label 2 or invalid 255."""
    try:
        buf = np.asarray(class_buffer)
        if buf.ndim == 3:
            buf = buf[:, :, 0]
        buf = buf[np.isfinite(buf)]
        if buf.size == 0:
            return []
        values, counts = np.unique(buf.astype(np.int32, copy=False), return_counts=True)
        order = np.argsort(counts)[::-1]
        return [(int(values[i]), int(counts[i])) for i in order[:int(max_items)]]
    except Exception:
        return []


def classified_line_to_nozzle_mask(classified_line, target_classes, n_nozzles, height=1, beischuss=0):
    """Map one NIR class-id line to a nozzle activation mask.

    In NIR line-scan mode the executable ejection event is a single timestamped
    line, not a crop from the rolling display buffer.  The returned default
    shape is therefore 1 x N_NOZZLES.  A larger height is only for display.
    """
    line = np.asarray(classified_line, dtype=np.uint8).reshape(-1)
    if line.size == 0:
        return np.zeros((int(height), int(n_nozzles)), dtype=np.uint8)

    targets = list(target_classes)
    active_classes = [idx for idx, enabled in enumerate(targets) if int(enabled) == 1]
    if not active_classes:
        return np.zeros((int(height), int(n_nozzles)), dtype=np.uint8)

    active_line = np.isin(line, active_classes).astype(np.uint8) * 255
    nozzle_line = cv2.resize(
        active_line[None, :],
        (int(n_nozzles), 1),
        interpolation=cv2.INTER_NEAREST,
    )[0]

    beischuss = int(max(0, beischuss))
    if beischuss > 0 and np.any(nozzle_line):
        kernel = np.ones((1, beischuss * 2 + 1), dtype=np.uint8)
        nozzle_line = cv2.dilate(nozzle_line[None, :], kernel, iterations=1)[0]

    return np.repeat(nozzle_line[None, :], int(height), axis=0).astype(np.uint8)


def classified_rolling_buffer_to_nozzle_mask(classified_buffer, target_classes, n_nozzles, beischuss=0):
    """Map the NIR rolling class history to a display-only nozzle history.

    Rows are time/line history, columns are belt width/nozzles.  This fixes the
    90-degree mismatch caused by showing a repeated single-line mask beside the
    rolling NIR image.
    """
    buf = np.asarray(classified_buffer, dtype=np.uint8)
    if buf.ndim == 1:
        buf = buf[None, :]
    h = max(1, int(buf.shape[0]))

    targets = list(target_classes)
    active_classes = [idx for idx, enabled in enumerate(targets) if int(enabled) == 1]
    if not active_classes:
        return np.zeros((h, int(n_nozzles)), dtype=np.uint8)

    active = np.isin(buf, active_classes).astype(np.uint8) * 255
    mapped = cv2.resize(
        active,
        (int(n_nozzles), h),
        interpolation=cv2.INTER_NEAREST,
    )

    beischuss = int(max(0, beischuss))
    if beischuss > 0 and np.any(mapped):
        kernel = np.ones((1, beischuss * 2 + 1), dtype=np.uint8)
        mapped = cv2.dilate(mapped, kernel, iterations=1)
    return mapped.astype(np.uint8)

def drawNozzleMask(
    boxes,
    target_classes,
    VORSCHUSS,
    NACHSCHUSS,
    BEISCHUSS,
    THRESHOLD,
    N_NOZZLES,
    last_mask,
    vert_movement,
):
    """
    Build the nozzle activation mask and shift the previous mask downward by
    vert_movement pixels. No globals are used; last_mask is owned by produce().
    """
    nozzle_mask = np.zeros((640, N_NOZZLES), dtype=np.uint8)
    height, width = nozzle_mask.shape

    for box in boxes:
        if int(box.cls) not in target_classes:
            continue
        
        x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()
        y_min = max(int(y1 * height) - NACHSCHUSS.value, 0)
        y_max = min(int(y2 * height) + VORSCHUSS.value, height)
        if N_NOZZLES <= 16:
            # Low nozzle/flipper count: activate every flipper whose physical bin
            # overlaps the detected object. This prevents objects between two flippers
            # from disappearing due to rounding.
            obj_x_min = x1 * width
            obj_x_max = x2 * width
        
            # Apply Beischuss in nozzle/flipper units converted to pixel width.
            nozzle_pitch = width / float(N_NOZZLES)
            obj_x_min -= BEISCHUSS.value * nozzle_pitch
            obj_x_max += BEISCHUSS.value * nozzle_pitch
        
            for nozzle_idx in range(N_NOZZLES):
                nozzle_x_min = nozzle_idx * nozzle_pitch
                nozzle_x_max = (nozzle_idx + 1) * nozzle_pitch
        
                overlaps = (
                    obj_x_max >= nozzle_x_min and
                    obj_x_min <= nozzle_x_max
                )
        
                if overlaps:
                    nozzle_mask[y_min:y_max, nozzle_idx] = 255
        
        # High nozzle count: original pixel-style mask is fine.
        else:
            x_min = max(int(x1 * width) - BEISCHUSS.value, 0)
            x_max = min(int(x2 * width) + BEISCHUSS.value, width)
            nozzle_mask[y_min:y_max, x_min:x_max] = 255

    # Preserve active region from the previous frame.
    np.maximum(last_mask, nozzle_mask, out=nozzle_mask)

    vert_movement = int(max(0, min(639, vert_movement)))
    if vert_movement > 0:
        vert_padding = np.zeros((vert_movement, N_NOZZLES), dtype=np.uint8)
        new_last_mask = np.vstack((vert_padding, nozzle_mask[:-vert_movement, :]))
    else:
        new_last_mask = nozzle_mask.copy()

    return nozzle_mask, new_last_mask


def update_vertical_movement_estimate(track_results, previous_track_centers, CALIBRATED_VERT_MOVEMENT):
    """
    Estimate vertical object displacement between consecutive frames using
    Ultralytics tracking IDs. This only writes the suggested value; it does not
    modify the active VERT_MOVEMENT setting.
    """
    try:
        boxes = track_results[0].boxes
        if boxes is None or boxes.id is None:
            return previous_track_centers

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        current_centers = {}

        for obj_id, box in zip(ids, xyxy):
            x1, y1, x2, y2 = box
            cy = float((y1 + y2) / 2.0)
            current_centers[int(obj_id)] = cy

        movements = []

        for obj_id, cy in current_centers.items():
            if obj_id in previous_track_centers:
                dy = cy - previous_track_centers[obj_id]
                if -200.0 <= dy <= 200.0:
                    movements.append(dy)

        if movements:
            frame_median = float(np.median(movements))
            movement_history.append(frame_median)

            CALIBRATED_VERT_MOVEMENT.value = float(
                np.median(movement_history)
            )

        return current_centers

    except Exception as e:
        print(f"[Process-1] Error in vertical movement calibration: {e}")
        return previous_track_centers


# ---------------------------------------------------------------------------
# Producer / consumer pipeline
# ---------------------------------------------------------------------------

def produce(
    DISPLAY_QUEUE,
    MASK_QUEUE,
    TARGET_CLASSES,
    STOP_FLAG,
    CONF,
    IOU,
    VORSCHUSS,
    NACHSCHUSS,
    BEISCHUSS,
    THRESHOLD,
    RECORD_RAW,
    DRAW_BBOXES,
    CAMERA_TYPE,
    PFS_PATH,
    MODEL_PATH,
    MODEL_VERBOSE,
    N_NOZZLES,
    VERT_MOVEMENT,
    CALIBRATED_VERT_MOVEMENT,
    RUN_VERT_CALIBRATION,
    ROTATE,
    FLIP_H,
    FLIP_V,
    VIDEO_PATH,
    RAW_RECORDING_FPS,
    RECORDING_PATHS,
    FPS,
    USB_SETTINGS_PATH,
    MVIMPACT_NIR_SETTINGS_PATH,
    NIR_CLASSIFIER_PATH="",
    NIR_CLASSIFIER_KIND="SAM_PLACEHOLDER",
    NIR_CLASS_COLORS=None,
    NIR_RAW_CHUNK_LINES_VALUE=NIR_RAW_CHUNK_LINES,
    MODEL_INFO=None,
    MODEL_SWAP_QUEUE=None,
    N_CLASSES=None,
):
    backup_image = cv2.imread("preheat_image.png")
    if backup_image is None:
        backup_image = np.zeros((640, 640, 3), dtype=np.uint8)
    is_nir_camera = is_nir_camera_type(CAMERA_TYPE)

    last_save_time = 0
    save_name = "aufnahme"
    nir_raw_state = {"buffer": [], "mode": None, "chunk_index": 0, "wall_start": None}
    nir_recording_was_active = False
    last_nir_mask_line_counter = None
    last_nir_conf_value = None
    last_nir_bg_threshold_value = None
    last_nir_read_stall_log_time = 0.0

    # Cammera Reconnection
    camera_connected = False
    camera = None
    camera_fail_count = 0
    MAX_CAMERA_FAILS = 5
    last_reconnect_attempt = 0.0
    RECONNECT_COOLDOWN = 2.0
    
    # USB Camera Settings
    usb_settings = load_usb_camera_settings(USB_SETTINGS_PATH)
    usb_reticule = usb_settings.get("camera", {}).get("reticule", {})
    usb_crop_size = int(usb_reticule.get("crop_size", 640))
    usb_crop_offset_x = int(usb_reticule.get("offset_x", 0))
    usb_crop_offset_y = int(usb_reticule.get("offset_y", 0))
    
    
    last_calibration_state = None
    
    
    # Producer-local mask state. This replaces global LAST_MASK / vert_padding.
    last_mask = np.zeros((640, N_NOZZLES), dtype=np.uint8)
    previous_track_centers = {}

    def apply_nir_runtime_settings_after_camera_open(reason="connect"):
        """Push current GUI/runtime NIR settings into a fresh camera object.

        The per-line loop deliberately applies background/confidence settings
        only when their GUI values change.  After connect/reconnect/hotswap the
        object is new, but the GUI value may be unchanged, so the change detector
        would otherwise skip reapplying the current threshold.
        """
        nonlocal last_nir_bg_threshold_value, last_nir_conf_value, last_nir_mask_line_counter

        if not is_nir_camera or not camera_connected or camera is None:
            return

        try:
            current_bg_threshold = float(THRESHOLD.value)
        except Exception:
            current_bg_threshold = 0.0

        bg_applied = set_nir_background_threshold(camera, current_bg_threshold)

        try:
            current_conf = _normalise_confidence_threshold(CONF.value)
        except Exception:
            current_conf = 0.0
        conf_applied = set_nir_classifier_confidence_threshold(camera, current_conf)

        clear_nir_classification_history(camera)
        last_nir_bg_threshold_value = current_bg_threshold
        last_nir_conf_value = current_conf
        last_nir_mask_line_counter = None

        print(
            f"[Process-1] Applied NIR runtime settings after {reason}: "
            f"intensity_threshold={current_bg_threshold}, "
            f"confidence={current_conf}, "
            f"bg_applied={bg_applied}, conf_applied={conf_applied}"
        )

    try:
        camera = connectCamera(
            CAMERA_TYPE,
            PFS_PATH,
            FPS,
            VIDEO_PATH=VIDEO_PATH,
            USB_SETTINGS_PATH=USB_SETTINGS_PATH,
            MVIMPACT_NIR_SETTINGS_PATH=MVIMPACT_NIR_SETTINGS_PATH,
            NIR_CLASSIFIER_PATH=NIR_CLASSIFIER_PATH,
            NIR_CLASSIFIER_KIND=NIR_CLASSIFIER_KIND,
        )
        camera_connected = camera is not None
        if camera_connected and CAMERA_TYPE.lower() == "basler":
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        # Push current NIR intensity/confidence settings immediately after
        # opening the camera.  Do not wait for a GUI value change.
        if camera_connected and is_nir_camera:
            apply_nir_runtime_settings_after_camera_open("connect")
    except Exception as e:
        print(f"[Process-1] Error in Camera Connection: {e}")
        camera_connected = False
        camera = None

    model = None
    model_kind = "YOLO"
    all_classes = []
    if is_nir_camera:
        print("[Process-1] NIR camera selected: skipping YOLO model load/preheat.")
    else:
        try:
            model, model_kind, model_names = load_detection_model(MODEL_PATH)
            all_classes = list(range(len(model_names)))
        except Exception as e:
            print(f"Error in Loading Model: {e}")
            print("Downloading yolov8n base model")
            model = YOLO("yolov8n.pt", task="detect")
            model_kind = "YOLO"
            model_names = names_to_dict(model.names)
            all_classes = list(range(len(model_names)))

        if N_CLASSES is not None:
            N_CLASSES.value = len(all_classes)
        publish_model_info(MODEL_INFO, model_names, model_kind, MODEL_PATH, "ready")

        try:
            _ = model("preheat_image.png", conf=CONF.value, iou=IOU.value, verbose=MODEL_VERBOSE)
            print("Preheat Complete")
        except Exception as e:
            print(f"[Process-1] Error in Model Preheating: {e}")

    def try_nir_classifier_hotswap():
        if MODEL_SWAP_QUEUE is None:
            return
    
        try:
            new_path = MODEL_SWAP_QUEUE.get_nowait()
        except Empty:
            return
    
        if not new_path:
            return
    
        base = os.path.basename(str(new_path))
        print(f"[Process-1] NIR classifier hotswap requested: {new_path}")
    
        if MODEL_INFO is not None:
            MODEL_INFO["status"] = f"loading {base} ..."
    
        try:
            # Update camera classifier path and reload classifier
            camera.classifier_path = str(new_path)
            camera.classifier_kind = "SKLEARN_PIPELINE"
    
            if hasattr(camera, "_prepare_classifier"):
                camera._prepare_classifier()
                apply_nir_runtime_settings_after_camera_open("classifier hotswap")
            else:
                raise RuntimeError("NIR camera does not support classifier reload")
    
            meta = load_sklearn_bundle_metadata(new_path)
            class_names = meta.get("class_names", [])
    
            if not class_names:
                n_classes = int(getattr(camera, "synthetic_classes", 4))
                class_names = ["Background"] + [
                    f"NIR Class {i}" for i in range(1, n_classes)
                ]
    
            names = {i: str(name) for i, name in enumerate(class_names)}
    
            # Reset class toggles because class indices may have changed
            for i in range(len(TARGET_CLASSES)):
                TARGET_CLASSES[i] = 0
    
            if N_CLASSES is not None:
                N_CLASSES.value = len(names)
    
            # Resize NIR colors if needed
            if NIR_CLASS_COLORS is not None:
                new_colors = normalise_nir_class_colors(
                    list(NIR_CLASS_COLORS),
                    len(names),
                )
    
                while len(NIR_CLASS_COLORS) > len(new_colors):
                    NIR_CLASS_COLORS.pop()
    
                while len(NIR_CLASS_COLORS) < len(new_colors):
                    NIR_CLASS_COLORS.append(tuple(new_colors[len(NIR_CLASS_COLORS)]))
    
                for i, color in enumerate(new_colors):
                    NIR_CLASS_COLORS[i] = tuple(color)
    
            publish_model_info(
                MODEL_INFO,
                names,
                meta.get("kind", "SKLEARN_PIPELINE"),
                new_path,
                "ready",
            )
            # Save selected Model Path to startup config yaml
            try:
                cfg = load_runtime_config()
                cfg["NIR_CLASSIFIER_PATH"] = str(new_path)
                cfg["NIR_CLASSIFIER_KIND"] = "SKLEARN_PIPELINE"
                save_runtime_config(cfg)
            except Exception as e:
                print(f"[Process-1] NIR classifier hotswap error in runtime config update: {e}")
            print(f"[Process-1] NIR classifier hotswap complete: {base}")
    
        except Exception as exc:
            print(f"[Process-1] NIR classifier hotswap failed: {exc}")
            if MODEL_INFO is not None:
                MODEL_INFO["status"] = f"load failed: {exc}"

    def try_model_hotswap():
        """RGB-mode only: swap the detection model when the UI requests it."""
        nonlocal model, model_kind, all_classes
        if MODEL_SWAP_QUEUE is None:
            return
        try:
            new_path = MODEL_SWAP_QUEUE.get_nowait()
        except Empty:
            return
        if not new_path:
            return

        base = os.path.basename(str(new_path))
        print(f"[Process-1] Model hotswap requested: {new_path}")
        if MODEL_INFO is not None:
            try:
                MODEL_INFO["status"] = f"loading {base} ..."
            except Exception:
                pass

        try:
            new_model, new_kind, new_names = load_detection_model(new_path)
            # Preheat before swapping so the live pipeline never sees the
            # first-inference latency spike.
            try:
                _ = new_model("preheat_image.png", conf=CONF.value, iou=IOU.value, verbose=MODEL_VERBOSE)
            except Exception as preheat_exc:
                print(f"[Process-1] Hotswap preheat warning: {preheat_exc}")

            if len(new_names) > MAX_MODEL_CLASSES:
                raise ValueError(
                    f"Model has {len(new_names)} classes; maximum supported is {MAX_MODEL_CLASSES}."
                )

            model = new_model
            model_kind = new_kind
            all_classes = list(range(len(new_names)))

            # Class indices are model-specific: reset the shared selection.
            for i in range(len(TARGET_CLASSES)):
                TARGET_CLASSES[i] = 0
            if N_CLASSES is not None:
                N_CLASSES.value = len(all_classes)

            # Reset producer-local detection state.
            last_mask.fill(0)
            previous_track_centers.clear()

            publish_model_info(MODEL_INFO, new_names, new_kind, new_path, "ready")
            print(f"[Process-1] Model hotswap complete: {new_kind} / {base}")
        except Exception as exc:
            print(f"[Process-1] Model hotswap failed, keeping previous model: {exc}")
            if MODEL_INFO is not None:
                try:
                    MODEL_INFO["status"] = f"load failed: {exc}"
                except Exception:
                    pass

    def reconnect_camera():
        nonlocal camera, camera_connected, last_reconnect_attempt, nir_recording_was_active
    
        now = time.monotonic()
        if now - last_reconnect_attempt < RECONNECT_COOLDOWN:
            return
    
        last_reconnect_attempt = now
        print("[Process-1] Attempting camera reconnect...")
    
        try:
            if camera is not None:
                try:
                    if CAMERA_TYPE.lower() == "basler":
                        if camera.IsGrabbing():
                            camera.StopGrabbing()
                        if camera.IsOpen():
                            camera.Close()
                    else:
                        camera.release()
                except Exception:
                    pass
    
            camera = connectCamera(
            CAMERA_TYPE,
            PFS_PATH,
            FPS,
            VIDEO_PATH=VIDEO_PATH,
            USB_SETTINGS_PATH=USB_SETTINGS_PATH,
            MVIMPACT_NIR_SETTINGS_PATH=MVIMPACT_NIR_SETTINGS_PATH,
            NIR_CLASSIFIER_PATH=NIR_CLASSIFIER_PATH,
            NIR_CLASSIFIER_KIND=NIR_CLASSIFIER_KIND,
        )
            camera_connected = camera is not None
            nir_recording_was_active = False
            
            if camera_connected and CAMERA_TYPE.lower() == "basler":
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                print("[Process-1] Basler reconnect successful.")
            elif camera_connected:
                print("[Process-1] Camera reconnect successful.")
            else:
                print("[Process-1] Camera reconnect failed.")

            if camera_connected and is_nir_camera:
                apply_nir_runtime_settings_after_camera_open("reconnect")
    
        except Exception as e:
            camera_connected = False
            camera = None
            print(f"[Process-1] Camera reconnect error: {e}")

    while not STOP_FLAG.is_set():
        # Defaults for this loop iteration.  In NIR mode, read_ms may otherwise
        # be stale/undefined if camera.read() raises before timing is computed.
        read_ms = 0.0
        nir_acquisition_failed = False

        if is_nir_camera:
            try_nir_classifier_hotswap()
        else:
            try_model_hotswap()

        try:
            if camera_connected and CAMERA_TYPE.lower() == "basler":
                if camera.IsGrabbing():
                    grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    if grab.GrabSucceeded():
                        img = grab.GetArray()
                        camera_fail_count = 0
                    grab.Release()
                else:
                    raise RuntimeError("Basler camera is not grabbing")
            
            elif CAMERA_TYPE.lower() == "simulated" and camera_connected:
                ret, img = camera.get_frame()
                if not ret or img is None:
                    img = backup_image
                    
            elif camera_connected and CAMERA_TYPE.lower() == "usb":
                ret, img = camera.read()
                if ret and img is not None:
                    img = center_crop_square(img,size=usb_crop_size,offset_x=usb_crop_offset_x,offset_y=usb_crop_offset_y,)
                else:
                    raise RuntimeError("USB camera frame grab failed")
            
            # NIR CAMERA READ BLOCK
            # NIR CAMERA READ BLOCK
            # NIR CAMERA READ BLOCK
            # NIR CAMERA READ BLOCK
            elif camera_connected and is_nir_camera:
                # Check if camera is recording lines:
                camera.raw_recording_fast_mode = bool(RECORD_RAW.value)
                
                # if it is NOT recording lines, prepare classification
                if not camera.raw_recording_fast_mode:
                    # THRESHOLD: mean spectrum intensity below this -> Background
                    try:
                        current_bg_threshold = float(THRESHOLD.value)
                        bg_changed = (
                            last_nir_bg_threshold_value is None or
                            abs(current_bg_threshold - last_nir_bg_threshold_value) > 1e-9
                        )

                        # This setter can touch the classifier/adapter. Do it only
                        # when the GUI value actually changes, not once per line.
                        if bg_changed:
                            if hasattr(camera, "set_background_threshold"):
                                camera.set_background_threshold(current_bg_threshold)
                            else:
                                camera.background_threshold = current_bg_threshold
                                clf = getattr(camera, "_classifier", None)
                                if hasattr(clf, "background_threshold"):
                                    clf.background_threshold = current_bg_threshold

                            clear_nir_classification_history(camera)
                            last_nir_bg_threshold_value = current_bg_threshold
                    except Exception:
                        pass
                    
                    # Runtime-adjustable sklearn confidence threshold.
                    # Use a recursive setter because the loaded joblib can be
                    # either the rejector itself or a wrapper/dict containing it.
                    try:
                        current_conf = _normalise_confidence_threshold(CONF.value)
                        if current_conf <= 0.0 and float(CONF.value) != 0.0:
                            CONF.value = 0.0
                        # Recursive threshold setting walks the loaded joblib/pipeline.
                        # Do it only on changes; doing it per line creates avoidable
                        # Python/object-walk overhead at hundreds of lines per second.
                        if last_nir_conf_value is None or abs(current_conf - last_nir_conf_value) > 1e-12:
                            set_nir_classifier_confidence_threshold(camera, current_conf)
                            clear_nir_classification_history(camera)
                            last_nir_conf_value = current_conf
                    except Exception:
                        pass
                                
                t_read0 = time.perf_counter()
                ret, img = camera.read()
                read_ms = (time.perf_counter() - t_read0) * 1000.0

                # A successful read can still hide a stall if the adapter blocks
                # inside imageRequestWaitFor() and recovers internally.  Log that
                # explicitly so UI "No New Lines" events have a matching console
                # breadcrumb.  Throttle to avoid flooding the console.
                if read_ms > 100.0:
                    now_log = time.monotonic()
                    if now_log - last_nir_read_stall_log_time > 1.0:
                        last_nir_read_stall_log_time = now_log
                        diag = {}
                        try:
                            if hasattr(camera, "get_diagnostics"):
                                diag = camera.get_diagnostics() or {}
                        except Exception:
                            diag = {}
                        print(
                            f"[Process-1] NIR read stall: read_ms={read_ms:.1f}, "
                            f"line_counter={int(getattr(camera, 'line_counter', 0))}, "
                            f"acq_restarts={diag.get('acquisition_restart_count', 'n/a')}, "
                            f"timeouts={diag.get('read_timeout_failures', 'n/a')}, "
                            f"not_ok={diag.get('request_not_ok_failures', 'n/a')}, "
                            f"last_error={diag.get('last_acquisition_error', '')}"
                        )
                
                if not ret or img is None:
                    raise RuntimeError("mvImpact NIR camera frame grab failed")
            else:
                img = backup_image

        except Exception as e:
            if is_nir_camera:
                # Do not use the RGB preheat image as a semantic NIR fallback.
                # NIR handling below creates an empty class line/display instead.
                img = None
                nir_acquisition_failed = True
            else:
                img = backup_image
            camera_fail_count += 1
        
            print(f"[Process-1] Error in Camera Acquisition ({camera_fail_count}/{MAX_CAMERA_FAILS}): {e}")
            print("Using Empty NIR Fallback" if is_nir_camera else "Using Backup Image")
        
            if (CAMERA_TYPE.lower() == "basler" or is_nir_camera) and camera_fail_count >= MAX_CAMERA_FAILS:
                reconnect_camera()
                camera_fail_count = 0
        
        try:
            if img is not None:
                if ROTATE.value > 0:
                    for i in range(ROTATE.value):
                        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
                if FLIP_H.value:
                    img = cv2.flip(img,1,0)
                if FLIP_V.value:
                    img = cv2.flip(img,0,1)
        except Exception as e:
            img = None if is_nir_camera else backup_image
            print(f"[Process-1] Error in Image Rotation: {e}")
            print("Using Empty NIR Fallback" if is_nir_camera else "Using Backup Image")
        
        if is_nir_camera:
            # NIR frames are rolling classified buffers from the camera adapter.
            # They must not be sent through YOLO. The latest classified line is
            # mapped directly to the existing mask/nozzle pipeline using the GUI
            # class selection toggles.
            fallback_width = int(getattr(camera, "width", 312) or 312)
            empty_nir_line = np.zeros((max(1, fallback_width),), dtype=np.uint8)

            if nir_acquisition_failed:
                # A failed NIR read must not reuse/eject the previous physical line.
                latest_line = empty_nir_line
                line_counter = int(getattr(camera, "line_counter", 0))
                is_new_line = False
            else:
                latest_line = getattr(camera, "last_classified_line", empty_nir_line)
                line_counter = int(getattr(camera, "line_counter", 0))

                # A failed read() leaves the previous line in place; the same
                # physical line must not be ejected twice.
                is_new_line = line_counter != last_nir_mask_line_counter
                last_nir_mask_line_counter = line_counter

            # Executable NIR ejection is one timestamped line.  Do not create a
            # 640-high mask and do not later crop it at DETECTION_POS.
            nozzle_line_mask = classified_line_to_nozzle_mask(
                latest_line,
                TARGET_CLASSES,
                N_NOZZLES,
                height=1,
                beischuss=BEISCHUSS.value,
            )

            try:
                # If no class is armed, all masks are guaranteed empty; do not
                # churn the inter-process delay/nozzle queues.  When at least
                # one class is armed, keep sending zero masks as before in case
                # the nozzle layer expects explicit off/empty updates.
                active_class_armed = any(int(TARGET_CLASSES[i]) == 1 for i in range(min(len(TARGET_CLASSES), int(N_CLASSES.value) if N_CLASSES is not None else len(TARGET_CLASSES))))
                if is_new_line and not RECORD_RAW.value and active_class_armed:
                    MASK_QUEUE.put_nowait((nozzle_line_mask, time.monotonic()))
            except Full:
                pass

            try:
                nir_display_decimation = 24  # UI only. Ejection/MASK_QUEUE still runs every line.

                if line_counter % nir_display_decimation == 0:
                    # All rolling-buffer copies, resizing and colorization are
                    # display-only. Keep them inside the decimated branch so the
                    # per-line ejection path stays as lean as possible.
                    rolling_classes = getattr(camera, "rolling_class_buffer", None)
                    if rolling_classes is None and nir_acquisition_failed:
                        rolling_classes = empty_nir_line[None, :]

                    if rolling_classes is not None:
                        display_nozzle_mask = classified_rolling_buffer_to_nozzle_mask(
                            rolling_classes,
                            TARGET_CLASSES,
                            N_NOZZLES,
                            beischuss=BEISCHUSS.value,
                        )
                        if not RECORD_RAW.value:
                            img = colorize_nir_class_buffer(rolling_classes, NIR_CLASS_COLORS)
                    else:
                        display_nozzle_mask = make_empty_nozzle_mask(N_NOZZLES)
                        if img is None:
                            img = np.zeros((640, 640, 3), dtype=np.uint8)

                    # Keep only newest UI frame on display lines. This avoids a
                    # stale full queue preventing newer frames from being shown.
                    try:
                        DISPLAY_QUEUE.get_nowait()
                    except Empty:
                        pass

                    nir_diag = {}
                    try:
                        if hasattr(camera, "get_diagnostics"):
                            nir_diag = camera.get_diagnostics() or {}
                    except Exception:
                        nir_diag = {}

                    nir_stats = {
                        "nir_lps": float(getattr(camera, "lps", 0.0)),
                        "nir_line_ms": float(getattr(camera, "line_period_ms", 0.0)),
                        "nir_cls_ms": float(getattr(camera, "classification_ms", 0.0)),
                        "nir_line_counter": int(getattr(camera, "line_counter", 0)),
                        "nir_read_ms": float(read_ms),
                        "nir_display_decimation": int(nir_display_decimation),
                        "nir_class_ids": nir_class_id_counts(rolling_classes) if rolling_classes is not None else [],
                        "nir_acq_restarts": int(nir_diag.get("acquisition_restart_count", 0) or 0),
                        "nir_read_timeouts": int(nir_diag.get("read_timeout_failures", 0) or 0),
                        "nir_request_not_ok": int(nir_diag.get("request_not_ok_failures", 0) or 0),
                        "nir_last_acq_error": str(nir_diag.get("last_acquisition_error", "") or ""),
                    }

                    DISPLAY_QUEUE.put_nowait((display_nozzle_mask, img, nir_stats))

            except Full:
                pass
            except Exception as e:
                print(f"[Process-1] Error in NIR Display Queue Update: {e}")
                STOP_FLAG.set()

            try:
                if RECORD_RAW.value:
                    _append_nir_raw_line(
                        nir_raw_state,
                        camera,
                        RECORDING_PATHS.raw_dir,
                        chunk_lines=NIR_RAW_CHUNK_LINES_VALUE,
                    )
                    nir_recording_was_active = True
                elif nir_recording_was_active:
                    buffer_to_save = nir_raw_state["buffer"]
                    nir_raw_state["buffer"] = []
                
                    state_to_save = {
                        "buffer": buffer_to_save,
                        "mode": nir_raw_state.get("mode"),
                        "chunk_index": nir_raw_state.get("chunk_index", 0),
                        "wall_start": nir_raw_state.get("wall_start"),
                    }
                
                    nir_raw_state["wall_start"] = None
                    nir_raw_state["chunk_index"] += 1
                
                    threading.Thread(
                        target=_flush_nir_raw_chunk,
                        args=(state_to_save, RECORDING_PATHS.raw_dir, "stopped"),
                        daemon=True,
                    ).start()
                
                    nir_recording_was_active = False
            except Exception as e:
                print(f"[Process-1] Error in Saving of NIR Raw Chunk: {e}")

            continue

        try:
            # If switching between calibration and inference -> drain stale queues
            current_calibration_state = bool(RUN_VERT_CALIBRATION.value)
            
            if current_calibration_state != last_calibration_state:
            
                # Drain display queue
                while True:
                    try:
                        DISPLAY_QUEUE.get_nowait()
                    except Empty:
                        break
            
                # Drain mask queue
                while True:
                    try:
                        MASK_QUEUE.get_nowait()
                    except Empty:
                        break
            
                # Reset producer-local state
                last_mask.fill(0)
                previous_track_centers.clear()
            
                last_calibration_state = current_calibration_state
            
            if RUN_VERT_CALIBRATION.value:
                results = model.track(
                    img,
                    conf=CONF.value,
                    iou=IOU.value,
                    verbose=MODEL_VERBOSE,
                    agnostic_nms=True,
                    persist=True,
                )
                previous_track_centers = update_vertical_movement_estimate(
                    results,
                    previous_track_centers,
                    CALIBRATED_VERT_MOVEMENT,
                )
            else:
                results = model.predict(
                    img,
                    conf=CONF.value,
                    iou=IOU.value,
                    verbose=MODEL_VERBOSE,
                    agnostic_nms=True,
                )
        except Exception as e:            
            print(f"[Process-1] Error in Model Inference: {e}")
            print(f"{img.shape}")

        try:
            timestamp = time.monotonic()
            boxes = results[0].boxes
            target_classes = [cls for cls in all_classes if TARGET_CLASSES[cls] == 1]

            nozzle_mask, last_mask = drawNozzleMask(
                boxes,
                target_classes,
                VORSCHUSS,
                NACHSCHUSS,
                BEISCHUSS,
                THRESHOLD,
                N_NOZZLES,
                last_mask,
                VERT_MOVEMENT.value,
            )

            MASK_QUEUE.put_nowait((nozzle_mask, timestamp))
        except Full:
            pass
        except Exception as e:
            print(f"[Process-1] Error in Mask Creation: {e}")
            STOP_FLAG.set()
        
        try:
            if DRAW_BBOXES.value:
                display_image = results[0].plot()
            else:
                display_image = img
                
            # Display queue is live-preview only: keep newest frame, discard stale one.
            try:
                DISPLAY_QUEUE.get_nowait()
            except Empty:
                pass
            
            DISPLAY_QUEUE.put_nowait((nozzle_mask, display_image))
        except Full:
            pass
        except Exception as e:
            print(f"[Process-1] Error in Display Queue Update: {e}")
            if camera_connected:
                if CAMERA_TYPE.lower() == "basler":
                    camera.Close()
                else:
                    camera.release()
            STOP_FLAG.set()
            
        # Record RAW Camera Images
        try:
            raw_fps = max(0.1, RAW_RECORDING_FPS.value)
            current_time = time.time()
        
            if RECORD_RAW.value and current_time > last_save_time + (1.0 / raw_fps):
                os.makedirs(RECORDING_PATHS.raw_dir, exist_ok=True)
        
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
                cv2.imwrite(
                    os.path.join(
                        RECORDING_PATHS.raw_dir,
                        f"{save_name}_{timestamp}.jpg"
                    ),
                    img
                )
        
                print(f"Saved Raw Image: {timestamp}")
                last_save_time = current_time
        
        except Exception as e:
            print(f"[Process-1] Error in Saving of Raw Image: {e}")

    # End of While Loop -> Cleaning up after STOP FLAG has been set
    if is_nir_camera:
        try:
            _flush_nir_raw_chunk(nir_raw_state, RECORDING_PATHS.raw_dir, reason="shutdown")
        except Exception as e:
            print(f"[Process-1] Error flushing final NIR raw chunk: {e}")

    if camera_connected:
        if CAMERA_TYPE.lower() == "basler":
            camera.Close()
        else:
            camera.release()
    STOP_FLAG.set()


def createMask(MASK_QUEUE, NOZZLE_ACTIVATION_QUEUE, STOP_FLAG, DELAY, DETECTION_POS,RUN_VERT_CALIBRATION):
    last_timestamp = None

    while not STOP_FLAG.is_set():
        try:
            if last_timestamp is None:
                last_mask, last_timestamp = MASK_QUEUE.get(timeout=0.05)

            now = time.monotonic()
            remaining = (last_timestamp + DELAY.value) - now
            if remaining > 0:
                STOP_FLAG.wait(min(remaining, 0.01))
                continue

            # Area-camera/YOLO mode still uses a y crop around DETECTION_POS.
            # NIR line-scan mode enqueues a 1 x N_NOZZLES line mask whose
            # timestamp is the line acquisition/classification time; for that
            # case there is no meaningful y-position to crop.
            if np.asarray(last_mask).shape[0] <= 1:
                activation_mask = np.asarray(last_mask, dtype=np.uint8).reshape(1, -1).copy()
            else:
                h = last_mask.shape[0]
                y0 = max(0, min(h - 1, DETECTION_POS - 6))
                y1 = max(y0 + 1, min(h, DETECTION_POS + 6))
                activation_mask = last_mask[y0:y1, :].copy()

            if not RUN_VERT_CALIBRATION.value:
                NOZZLE_ACTIVATION_QUEUE.put_nowait(activation_mask)
            
            last_timestamp = None

        except Empty:
            STOP_FLAG.wait(0.01)
        except Full:
            STOP_FLAG.wait(0.01)
        except Exception as e:
            print(f"[Process-1] Error in Creating Mask: {e}")
            STOP_FLAG.set()


def consume(MASK_QUEUE, STOP_FLAG, DELAY, NOZZLE_CONTROL_FUNCTION, DETECTION_POS,RUN_VERT_CALIBRATION):
    NOZZLE_ACTIVATION_QUEUE = queue.Queue(maxsize=500)
    create_mask_thread = threading.Thread(
        target=createMask,
        args=(MASK_QUEUE, NOZZLE_ACTIVATION_QUEUE, STOP_FLAG, DELAY, DETECTION_POS,RUN_VERT_CALIBRATION),
    )
    control_nozzle_thread = threading.Thread(
        target=NOZZLE_CONTROL_FUNCTION,
        args=(NOZZLE_ACTIVATION_QUEUE, STOP_FLAG),
    )

    create_mask_thread.start()
    control_nozzle_thread.start()

    create_mask_thread.join()
    control_nozzle_thread.join()


def load_sklearn_bundle_metadata(path):
    if not path or not os.path.exists(path):
        return {}

    try:
        import joblib
        loaded = joblib.load(path)

        if isinstance(loaded, dict):
            return {
                "class_names": loaded.get("class_names", []),
                "class_labels": loaded.get("class_labels", []),
                "kind": loaded.get("kind", "SKLEARN_PIPELINE"),
            }

    except Exception as exc:
        print(f"[NIR] Could not load sklearn bundle metadata from {path}: {exc}")

    return {}

# ---------------------------------------------------------------------------
# Main startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from STARTUP_SETTINGS_GUI import startup_config_gui

    startup_cfg = startup_config_gui()

    CAMERA_TYPE = startup_cfg["CAMERA_TYPE"]
    PFS_PATH = startup_cfg["PFS_PATH"]
    CONNECTION_TYPE = startup_cfg["CONNECTION_TYPE"].lower()
    MODEL_PATH = startup_cfg["MODEL_PATH"]
    MODEL_VERBOSE = startup_cfg["MODEL_VERBOSE"]
    VIDEO_PATH = startup_cfg["VIDEO_PATH"]
    SCALEABLE_UI = startup_cfg.get("SCALEABLE_UI", False)
    FPS_value = int(startup_cfg.get("FPS", 30))
    USB_SETTINGS_PATH = startup_cfg.get("USB_CAMERA_SETTINGS_PATH", "")
    MVIMPACT_NIR_SETTINGS_PATH = startup_cfg.get("MVIMPACT_NIR_SETTINGS_PATH", "")
    NIR_CLASSIFIER_PATH = startup_cfg.get("NIR_CLASSIFIER_PATH", "")
    NIR_CLASSIFIER_KIND = startup_cfg.get("NIR_CLASSIFIER_KIND", "SAM_PLACEHOLDER")
    

    try:
        NIR_RAW_CHUNK_LINES_VALUE = max(1, int(startup_cfg.get("NIR_RAW_CHUNK_LINES", NIR_RAW_CHUNK_LINES)))
    except Exception:
        NIR_RAW_CHUNK_LINES_VALUE = NIR_RAW_CHUNK_LINES
    
    if CAMERA_TYPE not in ("USB", "Basler", "SIMULATED", "MVIMPACT_NIR"):
        raise ValueError("CAMERA_TYPE must be USB, SIMULATED, Basler or MVIMPACT_NIR.")

    if CONNECTION_TYPE == "udp":
        N_NOZZLES = 128
        NOZZLE_CONTROL_FUNCTION = nozzle_control_UDP
    elif CONNECTION_TYPE == "serial":
        N_NOZZLES = 8
        NOZZLE_CONTROL_FUNCTION = nozzle_control_ARDUINO
    elif CONNECTION_TYPE == "modbus":
        N_NOZZLES = 80
        NOZZLE_CONTROL_FUNCTION = nozzle_control_MODBUS
    elif CONNECTION_TYPE == "simulated":
        N_NOZZLES = 10
        NOZZLE_CONTROL_FUNCTION = nozzle_control_SIMULATED
    else:
        raise ValueError("CONNECTION_TYPE must be UDP, SERIAL, or MODBUS.")

    DETECTION_POS = 600

    multiprocessing.set_start_method("spawn", force=True)

    MODEL_KIND = "NIR"
    if is_nir_camera_type(CAMERA_TYPE):
        nir_settings = load_nir_camera_settings(MVIMPACT_NIR_SETTINGS_PATH)
        bundle_meta = load_sklearn_bundle_metadata(NIR_CLASSIFIER_PATH)
    
        configured_names = bundle_meta.get("class_names") or nir_settings.get("class_names", [])
    
        if isinstance(configured_names, list) and configured_names:
            MODEL_NAMES = {
                i: str(name)
                for i, name in enumerate(configured_names)
            }
        else:
            n_nir_classes = int(
                nir_settings.get(
                    "synthetic_classes",
                    nir_settings.get("classes", 4)
                )
            )
            n_nir_classes = max(1, n_nir_classes)
    
            MODEL_NAMES = {0: "Background"}
            MODEL_NAMES.update({
                i: f"NIR Class {i}"
                for i in range(1, n_nir_classes)
            })
    
        MODEL_KIND = bundle_meta.get("kind", NIR_CLASSIFIER_KIND or "NIR")
        ALL_CLASSES = list(range(len(MODEL_NAMES)))
    
        print("[Main] NIR camera selected: skipping YOLO model load for UI.")
        print(f"[Main] NIR classifier kind: {NIR_CLASSIFIER_KIND}")
        if NIR_CLASSIFIER_PATH:
            print(f"[Main] NIR classifier path: {NIR_CLASSIFIER_PATH}")
        print(f"[Main] NIR classes: {MODEL_NAMES}")
        print(f"[Main] NIR raw recording chunk size: {NIR_RAW_CHUNK_LINES_VALUE} lines")
    else:
        try:
            model_temp, MODEL_KIND, MODEL_NAMES = load_detection_model(MODEL_PATH)
            ALL_CLASSES = list(range(len(MODEL_NAMES)))
        except Exception as e:
            print(f"Error in Loading Model for UI. Error: {e}")
            model_temp = YOLO("yolov8n.pt", task="detect")
            MODEL_KIND = "YOLO"
            MODEL_NAMES = names_to_dict(model_temp.names)
            ALL_CLASSES = list(range(len(MODEL_NAMES)))
        del model_temp  # Only the names/kind are needed in the main process.

    runtime_cfg = load_runtime_config()
    manager = multiprocessing.Manager()
    targets = runtime_cfg.get(
        "TARGET_CLASSES",
        np.zeros(len(MODEL_NAMES)).astype(int).tolist()
    )
    
    # Make sure target list matches current model class count
    if len(targets) != len(MODEL_NAMES):
        targets = np.zeros(len(MODEL_NAMES)).astype(int).tolist()
    
    # The shared array is allocated at the hot-swap maximum so a newly loaded
    # model with a different class count can reuse it. Only the first
    # N_CLASSES.value entries are meaningful at any time.
    padded_targets = (list(targets) + [0] * MAX_MODEL_CLASSES)[:MAX_MODEL_CLASSES]
    TARGET_CLASSES = multiprocessing.Array("i", padded_targets)
    N_CLASSES = multiprocessing.Value("i", len(MODEL_NAMES))

    # Model hot-swap channel (RGB camera modes only):
    #   UI  -> MODEL_SWAP_QUEUE: requested model file path
    #   producer -> MODEL_INFO:  names/kind/path/status + generation counter
    MODEL_SWAP_QUEUE = multiprocessing.Queue(maxsize=4)
    MODEL_INFO = manager.dict()
    MODEL_INFO["names"] = {int(k): str(v) for k, v in MODEL_NAMES.items()}
    MODEL_INFO["kind"] = MODEL_KIND
    MODEL_INFO["path"] = NIR_CLASSIFIER_PATH if is_nir_camera_type(CAMERA_TYPE) else MODEL_PATH
    MODEL_INFO["status"] = "starting..."
    MODEL_INFO["generation"] = 0

    nir_settings_for_colors = load_nir_camera_settings(MVIMPACT_NIR_SETTINGS_PATH) if is_nir_camera_type(CAMERA_TYPE) else {}
    default_color_cfg = nir_settings_for_colors.get("class_colors", [])
    runtime_color_cfg = runtime_cfg.get("NIR_CLASS_COLORS", default_color_cfg)
    nir_color_list = normalise_nir_class_colors(runtime_color_cfg, len(MODEL_NAMES))
    NIR_CLASS_COLORS = manager.list([tuple(c) for c in nir_color_list])
    
    CONF = multiprocessing.Value("d", runtime_cfg.get("CONF", 0.10))
    IOU = multiprocessing.Value("d", runtime_cfg.get("IOU", 0.00))
    DELAY = multiprocessing.Value("d", runtime_cfg.get("DELAY", 0.000))
    
    VORSCHUSS = multiprocessing.Value("i", runtime_cfg.get("VORSCHUSS", 0))
    NACHSCHUSS = multiprocessing.Value("i", runtime_cfg.get("NACHSCHUSS", 0))
    BEISCHUSS = multiprocessing.Value("i", runtime_cfg.get("BEISCHUSS", 0))

    # For YOLO mode this value is kept for backward compatibility.
    # For NIR mode it is the raw ALU background threshold used by SAM:
    #     mean(spectrum) < THRESHOLD -> background class 0
    if is_nir_camera_type(CAMERA_TYPE):
        nir_threshold_default = float(
            nir_settings_for_colors.get(
                "background_threshold",
                nir_settings_for_colors.get("sam_background_threshold", 300.0)
            )
        )
        runtime_threshold = float(runtime_cfg.get("THRESHOLD", nir_threshold_default))
        # Older UI versions stored 0..1 values. Treat those as stale and reset
        # to the NIR YAML default so the control starts in useful ALU units.
        if 0.0 <= runtime_threshold <= 1.0:
            runtime_threshold = nir_threshold_default
    else:
        runtime_threshold = float(runtime_cfg.get("THRESHOLD", 0.1))
    THRESHOLD = multiprocessing.Value("d", runtime_threshold)
    
    FPS = multiprocessing.Value("i", FPS_value)
    
    VERT_MOVEMENT = multiprocessing.Value(
        "i",
        runtime_cfg.get("VERT_MOVEMENT", 40)
    )
    
    CALIBRATED_VERT_MOVEMENT = multiprocessing.Value(
        "d",
        runtime_cfg.get("CALIBRATED_VERT_MOVEMENT", 40.0)
    )
    
    RUN_VERT_CALIBRATION = multiprocessing.Value(
        "b",
        runtime_cfg.get("RUN_VERT_CALIBRATION", False)
    )
    
    STOP_FLAG = multiprocessing.Event()
    
    RECORD_RAW = multiprocessing.Value(
        "b",
        runtime_cfg.get("RECORD_RAW", False)
    )
    
    DRAW_BBOXES = multiprocessing.Value(
        "b",
        runtime_cfg.get("DRAW_BBOXES", True)
    )
    
    ROTATE = multiprocessing.Value("i", runtime_cfg.get("ROTATE", 0))
    FLIP_H = multiprocessing.Value("b", runtime_cfg.get("FLIP_H", False))
    FLIP_V = multiprocessing.Value("b", runtime_cfg.get("FLIP_V", False))

    # Recording Variables
    RAW_RECORDING_FPS = multiprocessing.Value(
    "d",
    runtime_cfg.get("RAW_RECORDING_FPS", 2.0)
    )
    
    GUI_RECORDING_FPS = multiprocessing.Value(
        "d",
        runtime_cfg.get("GUI_RECORDING_FPS", 2.0)
    )
    
    RECORDING_PATHS = manager.Namespace()
    RECORDING_PATHS.raw_dir = runtime_cfg.get("RAW_RECORDING_DIR", "Recordings_Camera")
    RECORDING_PATHS.gui_dir = runtime_cfg.get("GUI_RECORDING_DIR", "Recordings_GUI")
    
    
    # NIR Classifier Settings
    NIR_CLASSIFIER_PATH = runtime_cfg.get(
    "NIR_CLASSIFIER_PATH",
    startup_cfg.get("NIR_CLASSIFIER_PATH", "")
    )
    
    NIR_CLASSIFIER_KIND = runtime_cfg.get(
        "NIR_CLASSIFIER_KIND",
        startup_cfg.get("NIR_CLASSIFIER_KIND", "SAM_PLACEHOLDER")
    )


    # QUEUES
    MASK_QUEUE = multiprocessing.Queue(maxsize=500)
    DISPLAY_QUEUE = multiprocessing.Queue(maxsize=1)

    producer = multiprocessing.Process(
        target=produce,
        args=(
            DISPLAY_QUEUE,
            MASK_QUEUE,
            TARGET_CLASSES,
            STOP_FLAG,
            CONF,
            IOU,
            VORSCHUSS,
            NACHSCHUSS,
            BEISCHUSS,
            THRESHOLD,
            RECORD_RAW,
            DRAW_BBOXES,
            CAMERA_TYPE,
            PFS_PATH,
            MODEL_PATH,
            MODEL_VERBOSE,
            N_NOZZLES,
            VERT_MOVEMENT,
            CALIBRATED_VERT_MOVEMENT,
            RUN_VERT_CALIBRATION,
            ROTATE,
            FLIP_H,
            FLIP_V,
            VIDEO_PATH,
            RAW_RECORDING_FPS,
            RECORDING_PATHS,
            FPS,
            USB_SETTINGS_PATH,
            MVIMPACT_NIR_SETTINGS_PATH,
            NIR_CLASSIFIER_PATH,
            NIR_CLASSIFIER_KIND,
            NIR_CLASS_COLORS,
            NIR_RAW_CHUNK_LINES_VALUE,
            MODEL_INFO,
            MODEL_SWAP_QUEUE,
            N_CLASSES,
        ),
    )

    displayer = multiprocessing.Process(
        target=display,
        args=(
            DISPLAY_QUEUE,
            STOP_FLAG,
            DELAY,
            TARGET_CLASSES,
            CONF,
            IOU,
            VORSCHUSS,
            NACHSCHUSS,
            BEISCHUSS,
            THRESHOLD,
            RECORD_RAW,
            DRAW_BBOXES,
            ALL_CLASSES,
            MODEL_NAMES,
            DETECTION_POS,
            VERT_MOVEMENT,
            CALIBRATED_VERT_MOVEMENT,
            RUN_VERT_CALIBRATION,
            ROTATE,
            FLIP_H,
            FLIP_V,
            RAW_RECORDING_FPS,
            GUI_RECORDING_FPS,
            RECORDING_PATHS,
            FPS,
            SCALEABLE_UI,
            is_nir_camera_type(CAMERA_TYPE),
            NIR_CLASSIFIER_KIND,
            NIR_CLASS_COLORS,
            MODEL_INFO,
            MODEL_SWAP_QUEUE,
            N_CLASSES,
        ),
    )

    consumer = multiprocessing.Process(
        target=consume,
        args=(MASK_QUEUE, STOP_FLAG, DELAY, NOZZLE_CONTROL_FUNCTION, DETECTION_POS,RUN_VERT_CALIBRATION),
    )

    processes = [producer, displayer, consumer]
    p_names = ["producer", "displayer", "consumer"]

    for process in processes:
        process.start()

    try:
        while not STOP_FLAG.is_set():
            time.sleep(0.05)
    except KeyboardInterrupt:
        STOP_FLAG.set()

    for i, process in enumerate(processes):
        process.join(timeout=10)
        if process.is_alive():
            print(f"Warning: Process: \"{p_names[i]}\" did not terminate gracefully")
            process.terminate()
        else:
            print(f"Process: \"{p_names[i]}\" terminated gracefully")

    runtime_cfg_out = {
    "TARGET_CLASSES": list(TARGET_CLASSES)[:max(1, int(N_CLASSES.value))],

    "CONF": CONF.value,
    "IOU": IOU.value,
    "DELAY": DELAY.value,

    "VORSCHUSS": VORSCHUSS.value,
    "NACHSCHUSS": NACHSCHUSS.value,
    "BEISCHUSS": BEISCHUSS.value,
    "THRESHOLD": THRESHOLD.value,

    "VERT_MOVEMENT": VERT_MOVEMENT.value,
    "CALIBRATED_VERT_MOVEMENT": CALIBRATED_VERT_MOVEMENT.value,
    "RUN_VERT_CALIBRATION": False,

    "RECORD_RAW": False,
    "DRAW_BBOXES": bool(DRAW_BBOXES.value),

    "ROTATE": ROTATE.value,
    "FLIP_H": bool(FLIP_H.value),
    "FLIP_V": bool(FLIP_V.value),
    
    "RAW_RECORDING_FPS": RAW_RECORDING_FPS.value,
    "GUI_RECORDING_FPS": GUI_RECORDING_FPS.value,
    "FPS": FPS.value,
    "RAW_RECORDING_DIR": RECORDING_PATHS.raw_dir,
    "GUI_RECORDING_DIR": RECORDING_PATHS.gui_dir,
    "NIR_CLASS_COLORS": [list(c) for c in list(NIR_CLASS_COLORS)],
    }
    
    save_runtime_config(runtime_cfg_out)
    
    print("Runtime settings saved.")
    print("Main process shutdown complete")

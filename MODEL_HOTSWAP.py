# -*- coding: utf-8 -*-
"""
Model hot-swap helpers for SenSoRTC (RGB camera modes only).

Provides:
    detect_ultralytics_model_kind(path) -> "YOLO" | "RTDETR"
    load_detection_model(path, kind=None) -> (model, kind, names_dict)

Detection strategy (most reliable first):
    .pt      -> inspect the torch checkpoint: model class name, model.yaml,
                and module names (RT-DETR models contain RTDETRDecoder).
    .onnx    -> inspect embedded Ultralytics metadata; fall back to the
                output-tensor shape (RT-DETR: (1, 300, 4+nc) query layout,
                YOLO: (1, 4+nc, anchors)).
    other / on failure -> filename heuristic ("rtdetr" in name).

Both model kinds expose the same predict()/track() API in Ultralytics, so the
producer pipeline does not need to special-case them after loading.
"""

import os


def names_to_dict(names):
    """Normalise Ultralytics model.names (dict or list) to {int: str}."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def _filename_kind(path):
    base = os.path.basename(str(path)).lower()
    if "rtdetr" in base or "rt-detr" in base or "rt_detr" in base:
        return "RTDETR"
    return "YOLO"


def _detect_kind_pt(path):
    """Inspect a .pt checkpoint without building the full Ultralytics wrapper."""
    try:
        import torch
    except Exception:
        return None

    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch without weights_only kwarg.
            ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[ModelHotswap] Could not inspect checkpoint {path!r}: {exc}")
        return None

    mdl = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
    if mdl is None:
        return None

    cls_name = type(mdl).__name__.lower()
    if "rtdetr" in cls_name:
        return "RTDETR"

    yaml_cfg = getattr(mdl, "yaml", None)
    if yaml_cfg is not None and "rtdetr" in str(yaml_cfg).lower():
        return "RTDETR"

    # RT-DETR detection models always contain an RTDETRDecoder head module.
    try:
        for module in mdl.modules():
            if "rtdetr" in type(module).__name__.lower():
                return "RTDETR"
    except Exception:
        pass

    # It is a readable torch detection checkpoint without any RT-DETR marker.
    return "YOLO"


def _detect_kind_onnx(path):
    """Inspect Ultralytics export metadata and the output tensor layout."""
    meta_text = ""

    try:
        import onnx
        model = onnx.load(path, load_external_data=False)
        meta_text = " ".join(
            f"{p.key}={p.value}" for p in model.metadata_props
        ).lower()

        if "rtdetr" in meta_text or "rt-detr" in meta_text:
            return "RTDETR"

        # Output-layout heuristic:
        #   RT-DETR : (batch, 300 queries, 4 + nc)
        #   YOLO    : (batch, 4 + nc, n_anchors) with n_anchors >> classes
        try:
            out = model.graph.output[0]
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            if len(dims) == 3:
                if dims[1] == 300 and 0 < dims[2] < dims[1]:
                    return "RTDETR"
                if dims[2] > dims[1] > 0:
                    return "YOLO"
        except Exception:
            pass
    except Exception:
        # onnx package unavailable or load failed: try onnxruntime metadata.
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            mm = sess.get_modelmeta()
            meta_text = (str(mm.custom_metadata_map) + " " + str(mm.description)).lower()
        except Exception:
            meta_text = ""

    if "rtdetr" in meta_text or "rt-detr" in meta_text:
        return "RTDETR"
    if "yolo" in meta_text:
        return "YOLO"
    return None


def detect_ultralytics_model_kind(path):
    """Return 'RTDETR' or 'YOLO' for the given model file."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    kind = None
    if ext == ".pt":
        kind = _detect_kind_pt(path)
    elif ext == ".onnx":
        kind = _detect_kind_onnx(path)

    if kind is None:
        kind = _filename_kind(path)
        print(f"[ModelHotswap] Falling back to filename heuristic for {os.path.basename(path)} -> {kind}")
    return kind


def load_detection_model(path, kind=None):
    """Load an Ultralytics detection model with auto YOLO/RT-DETR dispatch.

    Returns (model, kind, names) where names is a {class_id: name} dict.
    Raises on failure so the caller can keep its previous model.
    """
    from ultralytics import YOLO

    if kind is None:
        kind = detect_ultralytics_model_kind(path)

    if kind == "RTDETR":
        try:
            from ultralytics import RTDETR
            model = RTDETR(path)
        except Exception as exc:
            print(f"[ModelHotswap] RTDETR loader failed ({exc}); retrying with YOLO loader.")
            model = YOLO(path)
            kind = "YOLO"
    else:
        model = YOLO(path, task="detect")
        kind = "YOLO"

    names = names_to_dict(model.names)
    print(f"[ModelHotswap] Loaded {kind} model: {os.path.basename(str(path))} ({len(names)} classes)")
    return model, kind, names

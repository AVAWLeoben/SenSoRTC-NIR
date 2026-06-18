# -*- coding: utf-8 -*-
"""Train SenSoRTC NIR sklearn classifiers from block-style YAML configs.

This backend is intended for a GUI but is also usable from the command line:
    python train_nir_blocks.py nir_blocks_config.yaml

It writes a SenSoRTC-compatible joblib bundle. The mvImpact NIR runtime loads
bundle["pipeline"] and calls .predict(raw_spectra), so all preprocessing blocks
must be inside the sklearn Pipeline saved here.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from scipy_transformer_adapter import SavGolTransformer
from SKLEARN_PIPELINES import (
    ConfidenceRejectingClassifier,
    MRMRBandSelector,
    NIRPipelineBase,
    PCALoadingBandSelector,
    SNVTransformer,
    SpectralAngleMapperClassifier,
)


class _LoaderOnly(NIRPipelineBase):
    """Use NIRPipelineBase's file loaders without using its fixed model classes."""

    def build_pipeline(self):
        return Pipeline([("identity", FunctionTransformer(validate=False))])


def _load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Training config must be a YAML mapping/object.")
    return cfg


def _as_tuple(value):
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(value, tuple):
        return value
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("(", "").replace(")", "").split(",") if p.strip()]
        return tuple(int(p) for p in parts)
    return value


def _none_if_text(value):
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return value


def _int_or_none(value):
    value = _none_if_text(value)
    return None if value is None else int(value)



def _bool_from_any(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _pca_n_components(value):
    value = _none_if_text(value)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        return value

def _classes_from_cfg(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    classes = cfg.get("classes", [])
    if not isinstance(classes, list) or not classes:
        raise ValueError("Config must contain a non-empty classes list.")
    out = []
    for i, item in enumerate(classes, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Class entry #{i} must be a mapping.")
        name = str(item.get("name", f"Class_{i}")).strip() or f"Class_{i}"
        files = item.get("files", [])
        if isinstance(files, (str, os.PathLike)):
            files = [str(files)]
        files = [str(p) for p in files]
        if not files:
            raise ValueError(f"Class {name!r} has no training files.")
        out.append({"name": name, "files": files})
    return out


def _load_training_data(loader: _LoaderOnly, classes: List[Dict[str, Any]], min_mean: float):
    loader.expected_bands = None
    X_parts, y_parts = [], []
    samples_per_class = {}
    source_files_by_class = {}

    for label, cls in enumerate(classes, start=1):
        name = cls["name"]
        source_files_by_class[name] = cls["files"]
        n_class = 0
        for file_path in cls["files"]:
            X_file, _ = loader.load_spectral_file(file_path, label=label, min_mean=min_mean)
            # Important: Excel files are manually picked spectra. The Excel
            # loader must not filter or alter them. MAT/NPY/NPZ loaders apply
            # min_mean internally before returning.
            y_file = np.full(X_file.shape[0], label, dtype=np.uint8)
            print(f"Loaded {Path(file_path).name}: class={name} label={label} X={X_file.shape}")
            X_parts.append(X_file.astype(np.float32, copy=False))
            y_parts.append(y_file)
            n_class += int(X_file.shape[0])
        samples_per_class[name] = n_class

    X = np.vstack(X_parts).astype(np.float32, copy=False)
    y = np.concatenate(y_parts).astype(np.uint8, copy=False)
    return X, y, samples_per_class, source_files_by_class


def _safe_split(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int):
    unique, counts = np.unique(y, return_counts=True)
    stratify = y if np.all(counts >= 2) and 0.0 < test_size < 1.0 else None
    return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)


def _build_preprocessing_steps(blocks: List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    steps = []
    counts = {}

    def unique_name(base: str) -> str:
        counts[base] = counts.get(base, 0) + 1
        return base if counts[base] == 1 else f"{base}_{counts[base]}"

    for block in blocks:
        btype = str(block.get("type", "")).lower()
        params = block.get("params", {}) or {}

        if btype in ("savgol", "sgolay", "savitzky_golay"):
            steps.append((unique_name("savgol"), SavGolTransformer(
                window_length=int(params.get("window_length", 15)),
                polyorder=int(params.get("polyorder", 2)),
                deriv=int(params.get("deriv", 1)),
            )))
        elif btype in ("zscore", "standardscaler", "standard_scaler"):
            steps.append((unique_name("zscore"), StandardScaler()))
        elif btype in ("snv", "standard_normal_variate"):
            steps.append((unique_name("snv"), SNVTransformer(
                eps=float(params.get("eps", 1e-8)),
            )))
        elif btype == "mrmr":
            steps.append((unique_name("mrmr"), MRMRBandSelector(
                n_features=int(params.get("n_features", params.get("n_bands_select", 30))),
                redundancy_weight=float(params.get("redundancy_weight", 1.0)),
                random_state=int(params.get("random_state", 42)),
                max_samples_for_fit=int(params.get("max_samples_for_fit", 20000)),
            )))
        elif btype in ("pca", "pca_projection"):
            steps.append((unique_name("pca"), PCA(
                n_components=_pca_n_components(params.get("n_components", 0.99)),
                whiten=_bool_from_any(params.get("whiten", False)),
                random_state=int(params.get("random_state", 42)),
            )))
        elif btype in ("pca_loadings", "pcaloadings", "pca_loading_selector"):
            steps.append((unique_name("pca_loadings"), PCALoadingBandSelector(
                n_features=int(params.get("n_features", 30)),
                n_components=int(params.get("n_components", 5)),
                random_state=int(params.get("random_state", 42)),
                max_samples_for_fit=int(params.get("max_samples_for_fit", 20000)),
                weight_by_variance=_bool_from_any(params.get("weight_by_variance", True)),
            )))
        else:
            raise ValueError(f"Unknown preprocessing block type: {block.get('type')!r}")
    return steps


def _build_model_step(model_cfg: Dict[str, Any]) -> Tuple[str, Any]:
    mtype = str(model_cfg.get("type", model_cfg.get("class", "SNN"))).lower()
    params = model_cfg.get("params", {}) or {}
    random_state = int(params.get("random_state", 42))

    if mtype in ("snn", "shallow_nn", "mlp"):
        hidden = _as_tuple(params.get("hidden_layer_sizes", [24]))
        return "mlp", MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=str(params.get("activation", "relu")),
            alpha=float(params.get("alpha", 1e-4)),
            learning_rate_init=float(params.get("learning_rate_init", 1e-3)),
            max_iter=int(params.get("max_iter", 500)),
            early_stopping=bool(params.get("early_stopping", True)),
            random_state=random_state,
        )

    if mtype in ("svm_linear", "linear_svm", "svm-linear"):
        return "linear_svm", LinearSVC(
            C=float(params.get("C", 1.0)),
            class_weight=params.get("class_weight", "balanced"),
            random_state=random_state,
            max_iter=int(params.get("max_iter", 10000)),
        )

    if mtype in ("svm_rbf", "rbf_svm", "svm-rbf"):
        return "svm_rbf", SVC(
            kernel="rbf",
            C=float(params.get("C", 10.0)),
            gamma=params.get("gamma", "scale"),
            class_weight=params.get("class_weight", "balanced"),
            probability=bool(params.get("probability", True)),
        )

    if mtype in ("sam", "spectral_angle_mapper", "spectralanglemapper"):
        return "sam", SpectralAngleMapperClassifier(
            temperature=float(params.get("temperature", 0.05)),
            eps=float(params.get("eps", 1e-12)),
        )

    if mtype in ("random_forest", "randomforest", "rf"):
        # Random forests provide predict_proba(), so they work directly with
        # ConfidenceRejectingClassifier and the live confidence threshold.
        max_depth = _int_or_none(params.get("max_depth", None))
        max_features = _none_if_text(params.get("max_features", "sqrt"))
        class_weight = _none_if_text(params.get("class_weight", "balanced_subsample"))
        return "random_forest", RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=max_depth,
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=random_state,
        )

    raise ValueError(f"Unknown model block type: {model_cfg.get('type')!r}")


def _selected_bands(pipe: Pipeline):
    for _, step in pipe.steps:
        if hasattr(step, "get_selected_bands"):
            selected = step.get_selected_bands()
            if selected is not None:
                return [int(v) for v in selected]
    return None


def _pca_loading_scores(pipe: Pipeline):
    for _, step in pipe.steps:
        if hasattr(step, "get_loading_scores"):
            scores = step.get_loading_scores()
            if scores is not None:
                return [float(v) for v in np.asarray(scores).reshape(-1)]
    return None


def _pipeline_step_names(pipe: Pipeline) -> List[str]:
    return [name for name, _ in pipe.steps]



def compute_preview(config_path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Compute preview data for the GUI spectral-visualization tab.

    Returns raw and processed mean/std spectra per class. Processed spectra are
    after all spectrum-preserving preprocessing steps (SavGol, ZScore, SNV) and
    before the first dimensionality-reducing/selection step. If MRMR or PCA
    loading selection is present, selected bands and importance scores are also
    returned.
    """
    cfg = _load_yaml(config_path)
    classes = _classes_from_cfg(cfg)
    training_cfg = cfg.get("training", {}) or {}
    min_mean = float(training_cfg.get("min_mean", 800))
    random_state = int(training_cfg.get("random_state", 42))

    loader = _LoaderOnly(random_state=random_state)
    X, y, samples_per_class, source_files_by_class = _load_training_data(loader, classes, min_mean)
    pipeline_cfg = cfg.get("pipeline", {}) or {}
    preprocessing_blocks = pipeline_cfg.get("preprocessing", []) or []
    steps = _build_preprocessing_steps(preprocessing_blocks)

    X_raw = X.astype(np.float32, copy=False)
    X_display = X_raw.copy()
    display_step_names = []
    selector_info = None
    pca_info = None

    for step_name, step in steps:
        lower = step_name.lower()
        # Fit PCA projection separately so the GUI can show explained variance.
        if lower.startswith("pca") and not lower.startswith("pca_loadings") and isinstance(step, PCA):
            Z = step.fit_transform(X_display, y)
            pca_info = {
                "type": "PCA",
                "step_name": step_name,
                "n_components_out": int(Z.shape[1]),
                "explained_variance_ratio": [float(v) for v in np.asarray(getattr(step, "explained_variance_ratio_", []), dtype=np.float32).reshape(-1)],
                "mean_components_by_class": {
                    classes[i - 1]["name"]: [float(v) for v in np.mean(Z[y == i], axis=0).reshape(-1)]
                    for i in range(1, len(classes) + 1)
                    if np.any(y == i)
                },
            }
            break

        if hasattr(step, "get_selected_bands"):
            step.fit(X_display, y)
            selected = step.get_selected_bands()
            selected = np.asarray(selected if selected is not None else [], dtype=np.int64).reshape(-1)
            full_scores = None
            selected_scores = None
            score_label = "importance"
            if hasattr(step, "relevance_"):
                full_scores = np.asarray(getattr(step, "relevance_", []), dtype=np.float32).reshape(-1)
                selected_scores = full_scores[selected] if full_scores.size else np.asarray([], dtype=np.float32)
                score_label = "relevance"
            elif hasattr(step, "get_loading_scores"):
                tmp = step.get_loading_scores()
                if tmp is not None:
                    full_scores = np.asarray(tmp, dtype=np.float32).reshape(-1)
                    selected_scores = full_scores[selected] if full_scores.size else np.asarray([], dtype=np.float32)
                    score_label = "loading score"
            selector_info = {
                "type": "PCA_Loadings" if lower.startswith("pca_loadings") else "MRMR",
                "step_name": step_name,
                "selected_bands": [int(v) for v in selected],
                "selected_scores": [] if selected_scores is None else [float(v) for v in np.asarray(selected_scores).reshape(-1)],
                "full_scores": None if full_scores is None else [float(v) for v in np.asarray(full_scores).reshape(-1)],
                "score_label": score_label,
            }
            break

        # Spectrum-preserving transforms: fit/transform and continue.
        step.fit(X_display, y)
        X_display = np.asarray(step.transform(X_display), dtype=np.float32)
        display_step_names.append(step_name)

    def _stats_by_class(X_matrix):
        means = {}
        stds = {}
        for label, cls in enumerate(classes, start=1):
            mask = y == label
            if not np.any(mask):
                continue
            subset = X_matrix[mask]
            means[cls["name"]] = [float(v) for v in np.mean(subset, axis=0).reshape(-1)]
            # ddof=1 only when possible; otherwise std=0 for single spectrum.
            ddof = 1 if subset.shape[0] > 1 else 0
            stds[cls["name"]] = [float(v) for v in np.std(subset, axis=0, ddof=ddof).reshape(-1)]
        return means, stds

    raw_means, raw_stds = _stats_by_class(X_raw)
    processed_means, processed_stds = _stats_by_class(X_display)

    return {
        "class_names": [c["name"] for c in classes],
        "samples_per_class": samples_per_class,
        "source_files_by_class": source_files_by_class,
        "raw_n_bands": int(X_raw.shape[1]),
        "display_n_features": int(X_display.shape[1]),
        "display_step_names": display_step_names,
        "display_stage_label": "Raw spectra" if not display_step_names else " → ".join(display_step_names),
        "raw_class_means": raw_means,
        "raw_class_stds": raw_stds,
        "processed_class_means": processed_means,
        "processed_class_stds": processed_stds,
        # Backward-compatible aliases used by older GUI preview code.
        "class_means": processed_means,
        "class_stds": processed_stds,
        "selector": selector_info,
        "pca": pca_info,
    }

def train_from_blocks(config_path: str | os.PathLike[str]) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    classes = _classes_from_cfg(cfg)
    training_cfg = cfg.get("training", {}) or {}
    min_mean = float(training_cfg.get("min_mean", 800))
    test_size = float(training_cfg.get("test_size", 0.2))
    random_state = int(training_cfg.get("random_state", 42))
    confidence_threshold = float(training_cfg.get("confidence_threshold", cfg.get("confidence_threshold", 0.70)))

    loader = _LoaderOnly(random_state=random_state)
    X, y, samples_per_class, source_files_by_class = _load_training_data(loader, classes, min_mean)
    X_train, X_test, y_train, y_test = _safe_split(X, y, test_size, random_state)

    pipeline_cfg = cfg.get("pipeline", {}) or {}
    preprocessing_blocks = pipeline_cfg.get("preprocessing", []) or []
    model_cfg = pipeline_cfg.get("model", {}) or {}
    if not model_cfg:
        raise ValueError("Config must contain pipeline.model.")

    steps = _build_preprocessing_steps(preprocessing_blocks)
    steps.append(_build_model_step(model_cfg))
    pipe = Pipeline(steps)

    print("Pipeline steps:", _pipeline_step_names(pipe))
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    material_names = [c["name"] for c in classes]
    labels = list(range(1, len(material_names) + 1))
    cm = confusion_matrix(y_test, pred, labels=labels)
    report = classification_report(y_test, pred, labels=labels, target_names=material_names, zero_division=0)

    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    reject_label = int(training_cfg.get("reject_label", len(material_names) + 1))
    wrapped = ConfidenceRejectingClassifier(pipe, threshold=confidence_threshold, reject_label=reject_label)

    n_bands = int(loader.expected_bands)
    dummy = np.zeros((4, n_bands), dtype=np.float32)
    dummy_pred = np.asarray(wrapped.predict(dummy)).reshape(-1)
    if dummy_pred.shape[0] != dummy.shape[0]:
        raise RuntimeError(f"Runtime smoke test failed: got predict shape {dummy_pred.shape}")

    selected = _selected_bands(pipe)
    output = Path(cfg.get("output", "nir_block_model.joblib"))
    output.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": wrapped,
        "class_names": ["Background"] + material_names + ["Not classified"],
        "class_labels": list(range(len(material_names) + 2)),
        "kind": "BLOCK_PIPELINE",
        "format": "sklearn_pipeline_bundle_v5_block_gui",
        "n_bands": n_bands,
        "selected_bands": selected,
        "n_selected_bands": None if selected is None else len(selected),
        "pca_loading_scores": _pca_loading_scores(pipe),
        "recommended_confidence_threshold": confidence_threshold,
        "confidence_threshold": confidence_threshold,
        "confidence_threshold_runtime_adjustable": True,
        "confidence_threshold_attribute": "threshold",
        "reject_label": reject_label,
        "background_label": 0,
        "not_classified_label": reject_label,
        "preprocessing": {
            "pipeline_steps": _pipeline_step_names(pipe),
            "blocks": preprocessing_blocks,
        },
        "training": {
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "min_mean": min_mean,
            "excel_policy": "Excel spectra are manually picked and not background-filtered.",
            "test_size": test_size,
            "random_state": random_state,
            "samples_per_class": samples_per_class,
            "source_files_by_class": source_files_by_class,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
    }

    joblib.dump(bundle, output)

    # Reload exactly how the NIR runtime loads it and call predict on raw spectra.
    reloaded = joblib.load(output)
    runtime_pipe = reloaded["pipeline"] if isinstance(reloaded, dict) else reloaded
    reload_pred = np.asarray(runtime_pipe.predict(dummy)).reshape(-1)
    if reload_pred.shape[0] != dummy.shape[0]:
        raise RuntimeError("Reloaded joblib failed runtime compatibility smoke test.")

    summary = output.with_suffix(".training_summary.yaml")
    with open(summary, "w", encoding="utf-8") as f:
        yaml.safe_dump({k: v for k, v in bundle.items() if k != "pipeline"}, f, sort_keys=False)

    print(f"\nSaved: {output}")
    print(f"Saved summary: {summary}")
    return bundle



def validate_joblib(joblib_path: str | os.PathLike[str], config_path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    """Load a joblib exactly like the runtime and run predict() on raw spectra."""
    joblib_path = Path(joblib_path)
    if not joblib_path.exists():
        raise FileNotFoundError(f"Joblib not found: {joblib_path}")

    obj = joblib.load(joblib_path)
    bundle = obj if isinstance(obj, dict) else {"pipeline": obj}
    pipe = bundle.get("pipeline", obj)
    if not hasattr(pipe, "predict"):
        raise TypeError("Loaded joblib pipeline has no predict() method.")

    n_bands = int(bundle.get("n_bands", 0) or 0)
    if n_bands <= 0:
        raise ValueError("Joblib bundle is missing a valid n_bands value.")

    print(f"Loaded joblib: {joblib_path}")
    print(f"class_names: {bundle.get('class_names')}")
    print(f"background_label: {bundle.get('background_label')}")
    print(f"not_classified_label: {bundle.get('not_classified_label')}")
    print(f"confidence_threshold: {bundle.get('confidence_threshold', getattr(pipe, 'threshold', None))}")
    print(f"n_bands: {n_bands}")
    print(f"selected_bands: {bundle.get('selected_bands')}")

    samples_per_class = None
    y = None
    if config_path:
        cfg = _load_yaml(config_path)
        classes = _classes_from_cfg(cfg)
        min_mean = float((cfg.get("training", {}) or {}).get("min_mean", 800))
        loader = _LoaderOnly(random_state=int((cfg.get("training", {}) or {}).get("random_state", 42)))
        X, y, samples_per_class, _ = _load_training_data(loader, classes, min_mean)
        if X.shape[1] != n_bands:
            raise ValueError(f"Validation spectra have {X.shape[1]} bands but joblib expects {n_bands}.")
    else:
        X = np.zeros((16, n_bands), dtype=np.float32)

    t0 = time.perf_counter()
    pred = np.asarray(pipe.predict(X)).reshape(-1)
    dt = time.perf_counter() - t0
    if pred.shape[0] != X.shape[0]:
        raise RuntimeError(f"predict() returned {pred.shape[0]} labels for {X.shape[0]} spectra.")

    unique_pred, pred_counts = np.unique(pred, return_counts=True)
    print(f"Prediction OK: X={X.shape}, labels={pred.shape}")
    print(f"Prediction time: {dt * 1000:.3f} ms total, {(dt / max(1, X.shape[0])) * 1000:.6f} ms/spectrum")
    print("Predicted label counts:", {int(k): int(v) for k, v in zip(unique_pred, pred_counts)})

    if y is not None:
        labels = sorted(int(v) for v in np.unique(y))
        print("Validation confusion matrix:")
        print(confusion_matrix(y, pred, labels=labels))
        print(classification_report(y, pred, labels=labels, zero_division=0))

    return {
        "joblib": str(joblib_path),
        "n_bands": n_bands,
        "n_spectra": int(X.shape[0]),
        "prediction_ms_total": float(dt * 1000),
        "prediction_ms_per_spectrum": float((dt / max(1, X.shape[0])) * 1000),
        "predicted_label_counts": {int(k): int(v) for k, v in zip(unique_pred, pred_counts)},
        "samples_per_class": samples_per_class,
    }


def compare_models(config_path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """Train common candidate models on the same data/split and rank them."""
    cfg = _load_yaml(config_path)
    classes = _classes_from_cfg(cfg)
    training_cfg = cfg.get("training", {}) or {}
    min_mean = float(training_cfg.get("min_mean", 800))
    test_size = float(training_cfg.get("test_size", 0.2))
    random_state = int(training_cfg.get("random_state", 42))

    loader = _LoaderOnly(random_state=random_state)
    X, y, samples_per_class, _ = _load_training_data(loader, classes, min_mean)
    X_train, X_test, y_train, y_test = _safe_split(X, y, test_size, random_state)

    pipeline_cfg = cfg.get("pipeline", {}) or {}
    preprocessing_blocks = pipeline_cfg.get("preprocessing", []) or []
    candidates = [
        {"name": "Linear SVM", "type": "SVM_Linear", "params": {"C": 1.0, "class_weight": "balanced", "max_iter": 10000, "random_state": random_state}},
        {"name": "SNN", "type": "SNN", "params": {"hidden_layer_sizes": [48, 24], "alpha": 0.0001, "learning_rate_init": 0.001, "max_iter": 500, "early_stopping": True, "random_state": random_state}},
        {"name": "SAM", "type": "SAM", "params": {"temperature": 0.05, "eps": 1e-12}},
        {"name": "Random Forest", "type": "Random_Forest", "params": {"n_estimators": 300, "max_depth": "None", "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1, "random_state": random_state}},
        {"name": "SVM RBF", "type": "SVM_RBF", "params": {"C": 10.0, "gamma": "scale", "class_weight": "balanced", "probability": True, "random_state": random_state}},
    ]

    results = []
    labels = sorted(int(v) for v in np.unique(y))
    for cand in candidates:
        print(f"\n=== Comparing model: {cand['name']} ===")
        try:
            steps = _build_preprocessing_steps(preprocessing_blocks)
            steps.append(_build_model_step({"type": cand["type"], "params": cand["params"]}))
            pipe = Pipeline(steps)
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            fit_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            pred = pipe.predict(X_test)
            pred_s = time.perf_counter() - t1
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, labels=labels, average="macro", zero_division=0)
            cm = confusion_matrix(y_test, pred, labels=labels)
            result = {
                "model": cand["name"],
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "fit_seconds": float(fit_s),
                "predict_ms_total": float(pred_s * 1000),
                "predict_ms_per_spectrum": float((pred_s / max(1, X_test.shape[0])) * 1000),
                "confusion_matrix": cm.tolist(),
                "pipeline_steps": _pipeline_step_names(pipe),
                "selected_bands": _selected_bands(pipe),
                "error": "",
            }
            print(f"accuracy={acc:.4f} f1_macro={f1:.4f} fit={fit_s:.3f}s pred={pred_s*1000:.3f}ms")
            print(cm)
        except Exception as exc:
            result = {
                "model": cand["name"],
                "accuracy": -1.0,
                "f1_macro": -1.0,
                "fit_seconds": None,
                "predict_ms_total": None,
                "predict_ms_per_spectrum": None,
                "confusion_matrix": None,
                "pipeline_steps": [],
                "selected_bands": None,
                "error": str(exc),
            }
            print(f"FAILED: {exc}")
        results.append(result)

    def _rank_key(row):
        if row.get("error"):
            return (-1.0, -1.0, float("-inf"))
        pred_ms = row.get("predict_ms_per_spectrum")
        pred_ms = float(pred_ms) if pred_ms is not None else 1e9
        return (float(row.get("f1_macro", -1.0)), float(row.get("accuracy", -1.0)), -pred_ms)

    results.sort(key=_rank_key, reverse=True)
    output = Path(cfg.get("output", "nir_block_model.joblib"))
    csv_path = output.with_suffix(".model_comparison.csv")
    yaml_path = output.with_suffix(".model_comparison.yaml")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "model", "accuracy", "f1_macro", "fit_seconds", "predict_ms_total", "predict_ms_per_spectrum", "pipeline_steps", "selected_bands", "error"])
        writer.writeheader()
        for rank, row in enumerate(results, start=1):
            out = {k: row.get(k) for k in writer.fieldnames if k != "rank"}
            out["rank"] = rank
            writer.writerow(out)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"samples_per_class": samples_per_class, "results": results}, f, sort_keys=False)

    print("\n=== Model comparison ranking ===")
    for rank, row in enumerate(results, start=1):
        if row.get("error"):
            print(f"{rank}. {row['model']}: FAILED - {row['error']}")
        else:
            print(f"{rank}. {row['model']}: f1={row['f1_macro']:.4f}, acc={row['accuracy']:.4f}, pred={row['predict_ms_per_spectrum']:.6f} ms/spectrum")
    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison YAML: {yaml_path}")
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Train/validate/compare SenSoRTC NIR classifiers from block config.")
    parser.add_argument("config", help="Path to block YAML config, or joblib path with --validate")
    parser.add_argument("--validate", action="store_true", help="Validate a saved joblib. The positional config is the joblib path.")
    parser.add_argument("--validation-config", dest="validation_config", default=None, help="Optional training config path for validation spectra.")
    parser.add_argument("--compare", action="store_true", help="Compare common model types using the given config.")
    args = parser.parse_args()
    if args.validate:
        validate_joblib(args.config, args.validation_config)
    elif args.compare:
        compare_models(args.config)
    else:
        train_from_blocks(args.config)


if __name__ == "__main__":
    main()

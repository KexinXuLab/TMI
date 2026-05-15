from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


INDIAN_PINES_CLASSES = {
    1: "Alfalfa",
    2: "Corn-notill",
    3: "Corn-mintill",
    4: "Corn",
    5: "Grass-pasture",
    6: "Grass-trees",
    7: "Grass-pasture-mowed",
    8: "Hay-windrowed",
    9: "Oats",
    10: "Soybean-notill",
    11: "Soybean-mintill",
    12: "Soybean-clean",
    13: "Wheat",
    14: "Woods",
    15: "Buildings-Grass-Trees-Drives",
    16: "Stone-Steel-Towers",
}


@dataclass(frozen=True)
class DatasetGroup:
    group: int
    cube_shape: tuple[int, int, int]
    class_a: int
    class_b: int
    y: np.ndarray
    coords: np.ndarray
    patches: np.ndarray
    x_flat: np.ndarray


@dataclass(frozen=True)
class MethodSpec:
    name: str
    feature_kind: str
    model_factory: Callable[[int], object]


def resolve_local_path(path_text: str, script_file: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (Path(script_file).resolve().parent / path).resolve()


def paper_code_dirs() -> list[Path]:
    original_code_dir = Path(__file__).resolve().parent.parent
    return [
        original_code_dir / "MNIST",
        original_code_dir,
        original_code_dir / "CIFAR10",
        original_code_dir / "ODIR",
    ]


def add_original_code_to_path() -> None:
    for code_dir in reversed(paper_code_dirs()):
        if not code_dir.exists():
            continue
        code_dir_text = str(code_dir)
        if code_dir_text not in sys.path:
            sys.path.insert(0, code_dir_text)
    ensure_sklearn_metrics_compat()


def ensure_sklearn_metrics_compat() -> None:
    try:
        importlib.import_module("sklearn.metrics")
        return
    except ModuleNotFoundError:
        pass

    sklearn_module = sys.modules.get("sklearn") or types.ModuleType("sklearn")
    metrics_module = types.ModuleType("sklearn.metrics")
    metrics_module.normalized_mutual_info_score = normalized_mutual_info_score_compat
    metrics_module.pairwise_distances = pairwise_distances_compat
    sklearn_module.metrics = metrics_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.metrics"] = metrics_module


def entropy_from_labels(labels: np.ndarray) -> float:
    labels = np.asarray(labels).ravel()
    if labels.size == 0:
        return 0.0
    _values, counts = np.unique(labels, return_counts=True)
    probs = counts.astype(np.float64) / float(counts.sum())
    return float(-np.sum(probs * np.log(probs)))


def mutual_info_from_labels(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    labels_a = np.asarray(labels_a).ravel()
    labels_b = np.asarray(labels_b).ravel()
    if labels_a.shape[0] != labels_b.shape[0]:
        raise ValueError("Label arrays must have the same length")
    if labels_a.size == 0:
        return 0.0

    _a_values, a_inverse = np.unique(labels_a, return_inverse=True)
    _b_values, b_inverse = np.unique(labels_b, return_inverse=True)
    table = np.zeros((int(a_inverse.max()) + 1, int(b_inverse.max()) + 1), dtype=np.float64)
    np.add.at(table, (a_inverse, b_inverse), 1.0)
    table /= float(labels_a.size)

    pa = table.sum(axis=1, keepdims=True)
    pb = table.sum(axis=0, keepdims=True)
    expected = pa @ pb
    mask = table > 0.0
    return float(np.sum(table[mask] * np.log(table[mask] / expected[mask])))


def normalized_mutual_info_score_compat(labels_true, labels_pred, average_method: str = "arithmetic") -> float:
    h_true = entropy_from_labels(labels_true)
    h_pred = entropy_from_labels(labels_pred)
    if h_true == 0.0 and h_pred == 0.0:
        return 1.0

    mi = mutual_info_from_labels(labels_true, labels_pred)
    if average_method == "min":
        normalizer = min(h_true, h_pred)
    elif average_method == "geometric":
        normalizer = float(np.sqrt(h_true * h_pred))
    elif average_method == "max":
        normalizer = max(h_true, h_pred)
    else:
        normalizer = 0.5 * (h_true + h_pred)
    return float(mi / normalizer) if normalizer > 0.0 else 0.0


def pairwise_distances_compat(x, y=None, metric: str = "euclidean", **_kwargs):
    from scipy.spatial.distance import cdist

    x = np.asarray(x)
    y = x if y is None else np.asarray(y)
    metric_aliases = {
        "manhattan": "cityblock",
        "l1": "cityblock",
        "l2": "euclidean",
    }
    return cdist(x, y, metric=metric_aliases.get(metric, metric))


def import_paper_module(module_name: str):
    add_original_code_to_path()
    try:
        return importlib.import_module(module_name)
    except TypeError as exc:
        if "unsupported operand type" not in str(exc):
            raise
        sys.modules.pop(module_name, None)
        return load_paper_module_with_future_annotations(module_name)


def load_paper_module_with_future_annotations(module_name: str):
    module_path = find_paper_module_path(module_name)
    source = "from __future__ import annotations\n" + module_path.read_text(encoding="utf-8", errors="replace")
    module = types.ModuleType(module_name)
    module.__file__ = str(module_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile(source, str(module_path), "exec"), module.__dict__)
    return module


def find_paper_module_path(module_name: str) -> Path:
    for code_dir in paper_code_dirs():
        module_path = code_dir / f"{module_name}.py"
        if module_path.exists():
            return module_path
    raise ModuleNotFoundError(module_name)


def find_data_file(data_dir: Path, candidates: Iterable[str]) -> Path:
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    available = ", ".join(p.name for p in sorted(data_dir.glob("*.mat")))
    raise FileNotFoundError(
        f"Could not find any of {list(candidates)} in {data_dir}. "
        f"Available .mat files: {available or '(none)'}"
    )


def first_existing_key(mat: dict, keys: Iterable[str]) -> np.ndarray:
    for key in keys:
        if key in mat:
            return np.asarray(mat[key])
    public_keys = [key for key in mat if not key.startswith("__")]
    raise KeyError(f"None of keys {list(keys)} found. Public keys: {public_keys}")


def load_indian_pines(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise RuntimeError("scipy is required to read Indian Pines .mat files") from exc

    data_file = find_data_file(
        data_dir,
        [
            "Indian_pines_corrected.mat",
            "indian_pines_corrected.mat",
            "Indian_pines.mat",
            "indian_pines.mat",
        ],
    )
    gt_file = find_data_file(
        data_dir,
        [
            "Indian_pines_gt.mat",
            "indian_pines_gt.mat",
            "Indian_pines_groundtruth.mat",
            "indian_pines_groundtruth.mat",
        ],
    )

    data_mat = loadmat(data_file)
    gt_mat = loadmat(gt_file)
    cube = first_existing_key(data_mat, ["indian_pines_corrected", "indian_pines"])
    gt = first_existing_key(gt_mat, ["indian_pines_gt", "indian_pines_groundtruth", "gt"])

    if cube.ndim != 3:
        raise ValueError(f"Expected cube shape (H, W, bands), got {cube.shape}")
    if gt.ndim != 2:
        gt = np.squeeze(gt)
    if gt.shape != cube.shape[:2]:
        raise ValueError(f"Ground truth shape {gt.shape} does not match cube shape {cube.shape[:2]}")
    return cube, gt.astype(np.int64, copy=False)


def make_class_groups(gt: np.ndarray, class_a: int, class_b: int, group_size: int, seed: int) -> list[np.ndarray]:
    coords_a = np.argwhere(gt == class_a)
    coords_b = np.argwhere(gt == class_b)
    if len(coords_a) < group_size * 10:
        raise ValueError(
            f"Class {class_a} has {len(coords_a)} samples, fewer than {group_size * 10} needed for 10 groups."
        )
    if len(coords_b) < group_size:
        raise ValueError(f"Class {class_b} has {len(coords_b)} samples, fewer than {group_size}.")

    rng = np.random.default_rng(seed)
    shuffled_a = coords_a[rng.permutation(len(coords_a))]
    shuffled_b = coords_b[rng.permutation(len(coords_b))[:group_size]]

    groups = []
    for group in range(10):
        chunk_a = shuffled_a[group * group_size : (group + 1) * group_size]
        interleaved = np.empty((group_size * 2, 3), dtype=np.int64)
        interleaved[0::2, :2] = chunk_a
        interleaved[0::2, 2] = class_a
        interleaved[1::2, :2] = shuffled_b
        interleaved[1::2, 2] = class_b
        groups.append(interleaved)
    return groups


def extract_spatial_patches(cube: np.ndarray, coords_with_labels: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("patch radius must be non-negative")
    pad = int(radius)
    padded = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    side = 2 * pad + 1
    patches = np.empty((len(coords_with_labels), side, side, cube.shape[2]), dtype=cube.dtype)
    for i, (row, col, _label) in enumerate(coords_with_labels):
        rr = int(row) + pad
        cc = int(col) + pad
        patches[i] = padded[rr - pad : rr + pad + 1, cc - pad : cc + pad + 1, :]
    return patches


def load_group(
    data_dir: Path,
    group: int,
    class_a: int,
    class_b: int,
    group_size: int,
    seed: int,
    patch_radius: int,
) -> DatasetGroup:
    cube, gt = load_indian_pines(data_dir)
    groups = make_class_groups(gt, class_a, class_b, group_size, seed)
    if group < 0 or group >= len(groups):
        raise ValueError(f"group must be in [0, 9], got {group}")

    coords_with_labels = groups[group]
    patches = extract_spatial_patches(cube, coords_with_labels, patch_radius)
    y = coords_with_labels[:, 2].astype(np.int64)
    x_flat = patches.reshape(patches.shape[0], -1)
    return DatasetGroup(
        group=group,
        cube_shape=tuple(int(v) for v in cube.shape),
        class_a=class_a,
        class_b=class_b,
        y=y,
        coords=coords_with_labels[:, :2].copy(),
        patches=patches,
        x_flat=x_flat,
    )


def zscore_round(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(np.std(values))
    if std <= 1e-12 or not np.isfinite(std):
        return np.zeros_like(values, dtype=np.int16)
    normalized = (values - float(np.mean(values))) / std
    return np.rint(normalized).astype(np.int16)


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return 0.0
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def spectral_subtensors(patch: np.ndarray, spectral_radius: int) -> np.ndarray:
    if patch.ndim != 3:
        raise ValueError(f"Expected one patch with 3 dimensions, got {patch.shape}")
    bands = patch.shape[2]
    width = 2 * spectral_radius + 1
    if bands < width:
        raise ValueError(f"Patch has {bands} bands, fewer than spectral window {width}")
    blocks = []
    for center in range(spectral_radius, bands - spectral_radius):
        block = patch[:, :, center - spectral_radius : center + spectral_radius + 1]
        blocks.append(block.reshape(-1))
    return np.asarray(blocks)


def extract_p_tmi_feature(patch: np.ndarray, spectral_radius: int) -> np.ndarray:
    blocks = spectral_subtensors(patch, spectral_radius).astype(np.float64, copy=False)
    center_col = blocks.shape[1] // 2
    center = blocks[:, center_col]
    coeffs = np.array([safe_corrcoef(blocks[:, i], center) for i in range(blocks.shape[1])], dtype=np.float64)
    coeffs[center_col] = 1.0
    return zscore_round(np.mean(blocks * coeffs, axis=1))


def extract_m_tmi_feature(patch: np.ndarray, spectral_radius: int) -> np.ndarray:
    blocks = spectral_subtensors(patch, spectral_radius).astype(np.float64, copy=False)
    return zscore_round(np.max(blocks, axis=1))


def build_tmi_features(patches: np.ndarray, kind: str, spectral_radius: int) -> np.ndarray:
    extractor = extract_p_tmi_feature if kind == "P_TMI" else extract_m_tmi_feature
    return np.asarray([extractor(patch, spectral_radius) for patch in patches], dtype=np.int16)


def data_for_method(group_data: DatasetGroup, feature_kind: str, spectral_radius: int) -> np.ndarray:
    if feature_kind == "flat":
        return group_data.x_flat
    if feature_kind in {"P_TMI", "M_TMI"}:
        return build_tmi_features(group_data.patches, feature_kind, spectral_radius)
    raise ValueError(f"Unsupported feature kind {feature_kind}")


def majority_vote_adjust(y_true: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true)
    cluster_labels = np.asarray(cluster_labels)
    adjusted = np.empty_like(y_true)
    for label in np.unique(cluster_labels):
        idx = np.where(cluster_labels == label)[0]
        true_values, counts = np.unique(y_true[idx], return_counts=True)
        adjusted[idx] = true_values[int(np.argmax(counts))]
    return adjusted


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int) -> dict[str, float | int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    negative_labels = [int(v) for v in np.unique(y_true) if int(v) != int(positive_label)]
    if len(negative_labels) != 1:
        raise ValueError(f"Expected one negative label, got {negative_labels}")
    negative_label = negative_labels[0]

    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    fp = int(np.sum((y_true == negative_label) & (y_pred == positive_label)))
    tn = int(np.sum((y_true == negative_label) & (y_pred == negative_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred == negative_label)))
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "specificity": specificity * 100.0,
        "accuracy": accuracy * 100.0,
    }


def fit_predict(model: object, x: np.ndarray, iter_max: int) -> np.ndarray:
    try:
        model.fit(x, iter_max=iter_max)
    except TypeError:
        model.fit(x)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(x))
    if hasattr(model, "cluster_labels_"):
        return np.asarray(model.cluster_labels_)
    raise AttributeError(f"Model {type(model).__name__} has neither predict() nor cluster_labels_")


def run_one_method(
    group_data: DatasetGroup,
    spec: MethodSpec,
    repeats: int,
    seed: int,
    spectral_radius: int,
    iter_max: int,
    max_fit_attempts: int,
) -> tuple[list[dict], dict]:
    x = data_for_method(group_data, spec.feature_kind, spectral_radius)
    run_rows = []
    for repeat in range(repeats):
        run_seed = seed + group_data.group * 10000 + repeat
        labels = None
        last_exc = None
        attempts_used = 0
        for attempt in range(max_fit_attempts):
            attempt_seed = run_seed + attempt * 1000000
            attempts_used = attempt + 1
            np.random.seed(attempt_seed)
            model = spec.model_factory(attempt_seed)
            try:
                labels = fit_predict(model, x, iter_max=iter_max)
                break
            except Exception as exc:
                last_exc = exc
                print(
                    f"[group={group_data.group} method={spec.name} repeat={repeat + 1}] "
                    f"attempt {attempt + 1}/{max_fit_attempts} failed: {exc}",
                    flush=True,
                )
        if labels is None:
            raise RuntimeError(
                f"All {max_fit_attempts} fit attempts failed for group={group_data.group}, "
                f"method={spec.name}, repeat={repeat + 1}: {last_exc}"
            )

        adjusted = majority_vote_adjust(group_data.y, labels)
        metrics = compute_metrics(group_data.y, adjusted, positive_label=group_data.class_a)
        row = {
            "dataset": "Indian Pines",
            "group": group_data.group,
            "method": spec.name,
            "repeat": repeat + 1,
            "seed": run_seed,
            "fit_attempts": attempts_used,
            "class_a": group_data.class_a,
            "class_a_name": INDIAN_PINES_CLASSES.get(group_data.class_a, str(group_data.class_a)),
            "class_b": group_data.class_b,
            "class_b_name": INDIAN_PINES_CLASSES.get(group_data.class_b, str(group_data.class_b)),
            "n_samples": int(len(group_data.y)),
            "n_features": int(x.shape[1]),
            **metrics,
        }
        run_rows.append(row)
        print(
            f"[group={group_data.group} method={spec.name} repeat={repeat + 1}/{repeats}] "
            f"Acc={metrics['accuracy']:.2f} F1={metrics['f1']:.2f}",
            flush=True,
        )

    best = max(run_rows, key=lambda row: (float(row["accuracy"]), float(row["f1"])))
    acc_values = np.array([float(row["accuracy"]) for row in run_rows], dtype=np.float64)
    summary = {
        "dataset": "Indian Pines",
        "group": group_data.group,
        "method": spec.name,
        "class_a": group_data.class_a,
        "class_a_name": INDIAN_PINES_CLASSES.get(group_data.class_a, str(group_data.class_a)),
        "class_b": group_data.class_b,
        "class_b_name": INDIAN_PINES_CLASSES.get(group_data.class_b, str(group_data.class_b)),
        "n_samples": int(len(group_data.y)),
        "n_features": int(run_rows[0]["n_features"]),
        "repeats": int(repeats),
        "max_accuracy": float(best["accuracy"]),
        "mean_accuracy": float(np.mean(acc_values)),
        "sd_accuracy": float(np.std(acc_values, ddof=1)) if len(acc_values) > 1 else 0.0,
        "best_repeat": int(best["repeat"]),
        "best_tp": int(best["tp"]),
        "best_fp": int(best["fp"]),
        "best_tn": int(best["tn"]),
        "best_fn": int(best["fn"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_f1": float(best["f1"]),
        "best_specificity": float(best["specificity"]),
    }
    return run_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def parse_groups(group_text: str) -> list[int]:
    if str(group_text).upper() == "ALL":
        return list(range(10))
    group = int(group_text)
    if group < 0 or group > 9:
        raise ValueError(f"group must be 0-9 or ALL, got {group_text}")
    return [group]


def select_methods(method_text: str, specs: list[MethodSpec]) -> list[MethodSpec]:
    if str(method_text).upper() == "ALL":
        return specs
    by_name = {spec.name.upper(): spec for spec in specs}
    key = str(method_text).upper()
    if key not in by_name:
        raise ValueError(f"method must be one of {', '.join(by_name)}, or ALL")
    return [by_name[key]]


def build_parser(method_specs: list[MethodSpec], default_data_folder: str) -> argparse.ArgumentParser:
    default_method = "ALL" if len(method_specs) > 1 else method_specs[0].name
    parser = argparse.ArgumentParser(description="Run Indian Pines hyperspectral experiment.")
    parser.add_argument("--data-folder", default=default_data_folder, help="Folder containing Indian Pines .mat files.")
    parser.add_argument("--results-folder", default="./results", help="Folder for CSV and JSON outputs.")
    parser.add_argument("--group", default="0", help="Group id 0-9, or ALL.")
    parser.add_argument("--method", default=default_method, help="Method name, or ALL for this script.")
    parser.add_argument("--repeats", default=100, type=int, help="Independent runs per method/group.")
    parser.add_argument("--seed", default=42, type=int, help="Seed for group construction and run seeds.")
    parser.add_argument("--class-a", default=11, type=int, help="Positive class label.")
    parser.add_argument("--class-b", default=13, type=int, help="Negative class label.")
    parser.add_argument("--group-size", default=205, type=int, help="Samples per class in each binary group.")
    parser.add_argument("--patch-radius", default=1, type=int, help="Spatial patch radius; 1 gives 3x3 patches.")
    parser.add_argument("--spectral-radius", default=1, type=int, help="Spectral radius; 1 gives 3-band windows.")
    parser.add_argument("--iter-max", default=200, type=int, help="Maximum iterations passed to compatible algorithms.")
    parser.add_argument("--max-fit-attempts", default=3, type=int, help="Retries per repeat if initialization fails.")
    parser.add_argument("--no-save", action="store_true", help="Print results only; do not write CSV/JSON files.")
    parser.add_argument("--list-methods", action="store_true", help="Print available method names and exit.")
    return parser


def run_cli(method_specs: list[MethodSpec], default_data_folder: str, script_file: str) -> int:
    parser = build_parser(method_specs, default_data_folder)
    args = parser.parse_args()

    if args.list_methods:
        print("\n".join(spec.name for spec in method_specs))
        return 0
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.max_fit_attempts <= 0:
        parser.error("--max-fit-attempts must be positive")

    try:
        groups = parse_groups(args.group)
        selected_specs = select_methods(args.method, method_specs)
    except ValueError as exc:
        parser.error(str(exc))

    data_dir = resolve_local_path(args.data_folder, script_file)
    results_dir = resolve_local_path(args.results_folder, script_file)
    all_summaries = []
    for group in groups:
        group_data = load_group(
            data_dir=data_dir,
            group=group,
            class_a=args.class_a,
            class_b=args.class_b,
            group_size=args.group_size,
            seed=args.seed,
            patch_radius=args.patch_radius,
        )
        print(
            f"Loaded group {group}: cube={group_data.cube_shape}, "
            f"patches={group_data.patches.shape}, x_flat={group_data.x_flat.shape}",
            flush=True,
        )

        for spec in selected_specs:
            run_rows, summary = run_one_method(
                group_data=group_data,
                spec=spec,
                repeats=args.repeats,
                seed=args.seed,
                spectral_radius=args.spectral_radius,
                iter_max=args.iter_max,
                max_fit_attempts=args.max_fit_attempts,
            )
            all_summaries.append(summary)
            if not args.no_save:
                stem = f"indianpines_g{group:02d}_{spec.name}"
                write_csv(results_dir / f"{stem}_runs.csv", run_rows)
                write_json(results_dir / f"{stem}_summary.json", summary)
            print(
                f"[summary group={group} method={spec.name}] "
                f"Max={summary['max_accuracy']:.2f}, "
                f"Mean+/-SD={summary['mean_accuracy']:.2f}+/-{summary['sd_accuracy']:.2f}",
                flush=True,
            )

    if not args.no_save and len(all_summaries) > 1:
        write_csv(results_dir / "indianpines_all_summaries.csv", all_summaries)
    return 0

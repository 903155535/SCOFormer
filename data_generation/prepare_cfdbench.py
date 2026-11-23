#!/usr/bin/env python3

"""Prepare CFDBench (Luo et al., 2023) in the layout of utils/data.py.

Cavity, cylinder, and tube cases are split 8:1:1 by case with random seed 0 (the validation cases are unused).
Each case is cut into 20-frame segments, resized to 64 x 64 (bilinear, align_corners=True), and stored with the
channels u, v, and the flow-region mask. Only u and v are scored (utils/data.py).

    python data_generation/prepare_cfdbench.py --source-root <folder with cavity.zip, tube.zip, cylinder/> \
        --target-root ./data/cfdbench
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import zipfile
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
import torch.nn.functional as F


INFER_STEPS = 20
GRID_SIZE = 64
CASE_FILES = {"case.json", "u.npy", "v.npy"}
SUBSETS = ("prop", "bc", "geo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing the original CFDBench archives.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("./data/cfdbench"),
        help="Directory to store the extracted case folders and HDF5 outputs.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float64", "float32"),
        default="float64",
        help="Output dtype for the generated HDF5 files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def case_sort_key(path: Path) -> int:
    return int(path.name[4:])


def extract_selected_members(
    zip_path: Path,
    target_root: Path,
    member_mapper: Callable[[tuple[str, ...]], Path | None],
) -> int:
    copied = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            destination_rel = member_mapper(parts)
            if destination_rel is None:
                continue
            destination = target_root / destination_rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            copied += 1
    return copied


def extract_archives(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)

    def mixed_problem_mapper(parts: tuple[str, ...]) -> Path | None:
        if len(parts) != 3:
            return None
        if parts[0] not in SUBSETS or not parts[1].startswith("case"):
            return None
        if parts[2] not in CASE_FILES:
            return None
        return Path(*parts)

    def cylinder_mapper(subset: str) -> Callable[[tuple[str, ...]], Path | None]:
        def _mapper(parts: tuple[str, ...]) -> Path | None:
            if len(parts) != 2:
                return None
            if not parts[0].startswith("case") or parts[1] not in CASE_FILES:
                return None
            return Path(subset, *parts)

        return _mapper

    tasks = [
        ("cavity", source_root / "cavity.zip", target_root / "cavity", mixed_problem_mapper),
        ("tube", source_root / "tube.zip", target_root / "tube", mixed_problem_mapper),
        (
            "cylinder/bc",
            source_root / "cylinder" / "bc.zip",
            target_root / "cylinder",
            cylinder_mapper("bc"),
        ),
        (
            "cylinder/prop",
            source_root / "cylinder" / "prop.zip",
            target_root / "cylinder",
            cylinder_mapper("prop"),
        ),
        (
            "cylinder/geo",
            source_root / "cylinder" / "geo.zip",
            target_root / "cylinder",
            cylinder_mapper("geo"),
        ),
    ]

    for label, zip_path, destination_root, mapper in tasks:
        copied = extract_selected_members(zip_path, destination_root, mapper)
        print(f"[extract] {label}: copied {copied} files")


def collect_case_dirs(problem_root: Path) -> list[Path]:
    case_dirs: list[Path] = []
    for subset in SUBSETS:
        subset_dir = problem_root / subset
        this_case_dirs = sorted(subset_dir.glob("case*"), key=case_sort_key)
        case_dirs.extend(this_case_dirs)
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found under {problem_root}")
    return case_dirs


def split_case_dirs(case_dirs: list[Path], use_round: bool) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = list(case_dirs)
    random.seed(0)
    random.shuffle(shuffled)
    num_cases = len(shuffled)
    if use_round:
        num_train = round(num_cases * 0.8)
        num_dev = round(num_cases * 0.1)
    else:
        num_train = int(num_cases * 0.8)
        num_dev = int(num_cases * 0.1)
    train = shuffled[:num_train]
    dev = shuffled[num_train : num_train + num_dev]
    test = shuffled[num_train + num_dev :]
    return train, dev, test


def load_cavity_case(case_dir: Path) -> np.ndarray:
    u = np.load(case_dir / "u.npy")
    v = np.load(case_dir / "v.npy")
    mask = np.ones_like(u)
    return np.stack([u, v, mask], axis=1)


def load_tube_case(case_dir: Path) -> np.ndarray:
    case_params = load_json(case_dir / "case.json")
    u = np.load(case_dir / "u.npy")
    v = np.load(case_dir / "v.npy")
    mask = np.ones_like(u)

    u = np.pad(
        u,
        ((0, 0), (0, 0), (1, 0)),
        mode="constant",
        constant_values=case_params["vel_in"],
    )
    v = np.pad(v, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
    mask = np.pad(mask, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)

    u = np.pad(u, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    v = np.pad(v, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    mask = np.pad(mask, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    return np.stack([u, v, mask], axis=1)


def load_cylinder_case(case_dir: Path) -> np.ndarray:
    case_params = load_json(case_dir / "case.json")
    u = np.load(case_dir / "u.npy")
    v = np.load(case_dir / "v.npy")

    mask = np.ones_like(u)
    x_min = case_params["x_min"]
    x_max = case_params["x_max"]
    y_min = case_params["y_min"]
    y_max = case_params["y_max"]
    radius = case_params["radius"]

    height = y_max - y_min
    width = x_max - x_min
    dx = width / u.shape[2]
    dy = height / u.shape[1]
    for i in range(u.shape[1]):
        for j in range(u.shape[2]):
            x = x_min + j * dx
            y = y_min + i * dy
            if (x - 0.5) ** 2 + (y - 0.5) ** 2 <= radius**2:
                mask[:, i, j] = 0

    u = np.pad(
        u,
        ((0, 0), (0, 0), (1, 0)),
        mode="constant",
        constant_values=case_params["vel_in"],
    )
    v = np.pad(v, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
    mask = np.pad(mask, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)

    u = np.pad(u, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    v = np.pad(v, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    mask = np.pad(mask, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
    return np.stack([u, v, mask], axis=1)


def count_segments(case_dirs: list[Path]) -> int:
    total = 0
    for case_dir in case_dirs:
        time_steps = int(np.load(case_dir / "u.npy", mmap_mode="r").shape[0])
        total += math.ceil(time_steps / INFER_STEPS)
    return total


def split_trajectory(features: np.ndarray, output_dtype: np.dtype) -> np.ndarray:
    time_steps = features.shape[0]
    num_segments = math.ceil(time_steps / INFER_STEPS)
    padded_length = num_segments * INFER_STEPS

    padded = np.zeros((padded_length, *features.shape[1:]), dtype=features.dtype)
    padded[:time_steps] = features
    if time_steps % INFER_STEPS != 0:
        padded[time_steps:] = features[-1]

    resized = F.interpolate(
        torch.from_numpy(padded),
        size=(GRID_SIZE, GRID_SIZE),
        mode="bilinear",
        align_corners=True,
    ).numpy()
    resized = resized.reshape(num_segments, INFER_STEPS, *resized.shape[1:])
    resized = resized.transpose(0, 3, 4, 1, 2)
    return resized.astype(output_dtype, copy=False)


def write_split(
    output_path: Path,
    case_groups: list[tuple[str, list[Path], Callable[[Path], np.ndarray]]],
    output_dtype: np.dtype,
) -> tuple[int, tuple[int, int, int, int, int]]:
    total_segments = sum(count_segments(case_dirs) for _, case_dirs, _ in case_groups)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as handle:
        dataset = handle.create_dataset(
            "data",
            shape=(total_segments, GRID_SIZE, GRID_SIZE, INFER_STEPS, 3),
            dtype=output_dtype,
            compression=None,
        )

        write_index = 0
        for problem_name, case_dirs, loader in case_groups:
            print(f"[write] {output_path.name}: {problem_name} cases={len(case_dirs)}")
            for case_idx, case_dir in enumerate(case_dirs, start=1):
                features = loader(case_dir)
                segments = split_trajectory(features, output_dtype)
                next_index = write_index + segments.shape[0]
                dataset[write_index:next_index] = segments
                write_index = next_index
                if case_idx % 25 == 0 or case_idx == len(case_dirs):
                    print(
                        f"  - {problem_name}: {case_idx}/{len(case_dirs)} cases, "
                        f"segments={write_index}/{total_segments}"
                    )

        if write_index != total_segments:
            raise RuntimeError(
                f"Wrote {write_index} segments but expected {total_segments}"
            )
        final_shape = dataset.shape

    return total_segments, final_shape


def verify_outputs(target_root: Path) -> None:
    for file_name in ("ns2d_cdb_train.hdf5", "ns2d_cdb_test.hdf5"):
        path = target_root / file_name
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            print(
                f"[verify] {file_name}: shape={data.shape}, dtype={data.dtype}, "
                f"min={float(data[0].min()):.6f}, max={float(data[0].max()):.6f}"
            )


def main() -> None:
    args = parse_args()
    output_dtype = np.float64 if args.dtype == "float64" else np.float32

    extract_archives(args.source_root, args.target_root)

    cavity_train, _, cavity_test = split_case_dirs(
        collect_case_dirs(args.target_root / "cavity"),
        use_round=True,
    )
    tube_train, _, tube_test = split_case_dirs(
        collect_case_dirs(args.target_root / "tube"),
        use_round=False,
    )
    cylinder_train, _, cylinder_test = split_case_dirs(
        collect_case_dirs(args.target_root / "cylinder"),
        use_round=False,
    )

    train_groups = [
        ("cavity", cavity_train, load_cavity_case),
        ("cylinder", cylinder_train, load_cylinder_case),
        ("tube", tube_train, load_tube_case),
    ]
    test_groups = [
        ("cavity", cavity_test, load_cavity_case),
        ("cylinder", cylinder_test, load_cylinder_case),
        ("tube", tube_test, load_tube_case),
    ]

    train_total, train_shape = write_split(
        args.target_root / "ns2d_cdb_train.hdf5",
        train_groups,
        output_dtype,
    )
    print(f"[done] train segments={train_total}, shape={train_shape}")

    test_total, test_shape = write_split(
        args.target_root / "ns2d_cdb_test.hdf5",
        test_groups,
        output_dtype,
    )
    print(f"[done] test segments={test_total}, shape={test_shape}")

    verify_outputs(args.target_root)


if __name__ == "__main__":
    main()

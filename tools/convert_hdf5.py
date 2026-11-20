#!/usr/bin/env python3
"""Convert the ANN-Benchmarks GloVe-200-angular HDF5 file to raw binaries.

Expected input:
    data/raw/glove-200-angular.hdf5

Outputs:
    data/processed/glove-200-angular/base.f32
    data/processed/glove-200-angular/queries.f32
    data/processed/glove-200-angular/neighbors.i32
    data/processed/glove-200-angular/distances.f32
    data/processed/glove-200-angular/metadata.json

All binary arrays are written in C-order (row-major), little-endian format,
with no header. Their shapes and dtypes are recorded in metadata.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


# Resolve paths relative to this script, not the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "glove-200-angular.hdf5"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "glove-200-angular"

# HDF5 key: (output filename, expected shape, output dtype)
DATASETS = {
    "train": (
        "base.f32",
        (1_183_514, 200),
        np.dtype("<f4"),
    ),
    "test": (
        "queries.f32",
        (10_000, 200),
        np.dtype("<f4"),
    ),
    "neighbors": (
        "neighbors.i32",
        (10_000, 100),
        np.dtype("<i4"),
    ),
    "distances": (
        "distances.f32",
        (10_000, 100),
        np.dtype("<f4"),
    ),
}


def json_compatible(value: Any) -> Any:
    """Convert common NumPy/HDF5 attribute values into JSON-compatible data."""
    if isinstance(value, np.ndarray):
        return [json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_compatible(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, tuple):
        return [json_compatible(item) for item in value]
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def validate_dataset(key: str, dataset: h5py.Dataset, expected_shape: tuple[int, ...],
                     expected_dtype: np.dtype) -> None:
    """Validate an HDF5 dataset before exporting it."""
    if dataset.shape != expected_shape:
        raise ValueError(
            f"Dataset '{key}' has shape {dataset.shape}; expected {expected_shape}"
        )

    source_dtype = np.dtype(dataset.dtype)
    if source_dtype.kind != expected_dtype.kind or source_dtype.itemsize != expected_dtype.itemsize:
        raise TypeError(
            f"Dataset '{key}' has dtype {source_dtype}; expected a compatible "
            f"{expected_dtype} value type"
        )


def export_dataset(hdf5_dataset: h5py.Dataset, output_path: Path,
                   output_dtype: np.dtype, key: str) -> dict[str, Any]:
    """Read, validate, and export one dataset as a headerless binary array."""
    array = np.asarray(hdf5_dataset[()], dtype=output_dtype, order="C")

    if output_dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"Dataset '{key}' contains NaN or infinite values")

    if key == "neighbors":
        if array.size:
            minimum = int(array.min())
            maximum = int(array.max())
            database_size = DATASETS["train"][1][0]
            if minimum < 0 or maximum >= database_size:
                raise ValueError(
                    "Ground-truth neighbor indices are outside the valid range "
                    f"[0, {database_size - 1}]: min={minimum}, max={maximum}"
                )

    # Write to a temporary file first, so an interrupted write does not leave
    # a partial file with the final filename.
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        array.tofile(temporary_path)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    expected_bytes = array.size * output_dtype.itemsize
    actual_bytes = output_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise IOError(
            f"Size check failed for '{output_path.name}': "
            f"expected {expected_bytes} bytes, got {actual_bytes}"
        )

    return {
        "filename": output_path.name,
        "shape": list(array.shape),
        "dtype": output_dtype.name,
        "byte_order": "little-endian",
        "order": "C",
        "elements": int(array.size),
        "size_bytes": actual_bytes,
    }


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}\n"
            "Download glove-200-angular.hdf5 into data/raw/ first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, dict[str, Any]] = {}

    with h5py.File(INPUT_PATH, "r") as hdf5_file:
        missing_keys = set(DATASETS) - set(hdf5_file.keys())
        if missing_keys:
            raise KeyError(f"Missing HDF5 datasets: {sorted(missing_keys)}")

        # Preserve the original benchmark attributes as JSON-compatible values.
        source_attributes = {
            str(name): json_compatible(value)
            for name, value in hdf5_file.attrs.items()
        }

        for key, (filename, expected_shape, output_dtype) in DATASETS.items():
            dataset = hdf5_file[key]
            validate_dataset(key, dataset, expected_shape, output_dtype)

            output_path = OUTPUT_DIR / filename
            print(f"Exporting {key}: {dataset.shape} -> {output_path}")
            exported_files[key] = export_dataset(
                dataset, output_path, output_dtype, key
            )

    metadata = {
        "dataset": "GloVe-200-angular",
        "source_file": INPUT_PATH.name,
        "distance_metric": source_attributes.get("distance", "angular"),
        "source_attributes": source_attributes,
        "database_size": DATASETS["train"][1][0],
        "query_count": DATASETS["test"][1][0],
        "dimensions": DATASETS["train"][1][1],
        "ground_truth_k": DATASETS["neighbors"][1][1],
        "files": exported_files,
    }

    metadata_path = OUTPUT_DIR / "metadata.json"
    temporary_metadata_path = metadata_path.with_suffix(".json.tmp")
    temporary_metadata_path.write_text(
        json.dumps(metadata, indent=4) + "\n",
        encoding="utf-8",
    )
    temporary_metadata_path.replace(metadata_path)

    print("\nConversion completed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata: {metadata_path}")
    for key, info in exported_files.items():
        size_mib = info["size_bytes"] / (1024 * 1024)
        print(f"  {info['filename']}: {size_mib:.2f} MiB")


if __name__ == "__main__":
    main()

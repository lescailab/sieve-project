"""
Helpers for optional sample-level covariates.

This module currently supports ancestry principal components (PCs) loaded
from a TSV file. Sex encoding is kept separate so the existing sex-only
path remains unchanged unless extra covariates are explicitly requested.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def encode_sex_for_covariate(sex: Optional[str]) -> float:
    """
    Encode sex as a float covariate.

    Returns
    -------
    float
        1.0 for male, 0.0 for female, -1.0 for unknown / missing.
    """
    if sex == 'M':
        return 1.0
    if sex == 'F':
        return 0.0
    return -1.0


def compute_file_sha256(path: str | Path) -> str:
    """Return the SHA256 hash of a file."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_pc_columns(columns: Iterable[str]) -> Dict[int, str]:
    """Map PC index -> column name for columns named like PC1, PC2, ..."""
    mapping: Dict[int, str] = {}
    for column in columns:
        col = str(column).strip()
        upper = col.upper()
        if not upper.startswith('PC'):
            continue
        suffix = upper[2:]
        if suffix.isdigit():
            mapping[int(suffix)] = column
    return mapping


def load_pc_map(pc_map_file: str | Path, num_pcs: int) -> Dict[str, np.ndarray]:
    """
    Load a sample -> PC vector mapping from a TSV file.

    Parameters
    ----------
    pc_map_file : str or Path
        TSV with columns ``sample_id, PC1, PC2, ...``.
    num_pcs : int
        Number of PCs to extract.

    Returns
    -------
    Dict[str, np.ndarray]
        Sample ID mapped to a float32 vector of length ``num_pcs``.
    """
    if num_pcs <= 0:
        raise ValueError("--num-pcs must be > 0 when --pc-map is supplied")

    pc_df = pd.read_csv(pc_map_file, sep='\t')
    if 'sample_id' not in pc_df.columns:
        raise ValueError(
            f"PC map must contain a 'sample_id' column; got {list(pc_df.columns)}"
        )

    if pc_df['sample_id'].duplicated().any():
        duplicated = pc_df.loc[pc_df['sample_id'].duplicated(), 'sample_id'].iloc[0]
        raise ValueError(f"PC map contains duplicate sample_id entries (e.g. {duplicated!r})")

    pc_columns = _normalise_pc_columns(pc_df.columns)
    required = list(range(1, num_pcs + 1))
    missing = [f'PC{i}' for i in required if i not in pc_columns]
    if missing:
        raise ValueError(
            f"PC map is missing required columns for --num-pcs {num_pcs}: {missing}. "
            f"Available columns: {list(pc_df.columns)}"
        )

    ordered_columns = [pc_columns[i] for i in required]
    pc_map: Dict[str, np.ndarray] = {}
    for _, row in pc_df.iterrows():
        sample_id = str(row['sample_id'])
        values = row[ordered_columns].to_numpy(dtype=np.float32, copy=True)
        pc_map[sample_id] = values

    return pc_map


def attach_pc_covariates_to_samples(
    samples,
    pc_map: Dict[str, np.ndarray],
    include_sex: bool,
) -> None:
    """
    Attach per-sample PC vectors and combined covariate vectors in-place.

    The combined covariate vector follows the bound convention:
    sex occupies column 0 when ``include_sex`` is true, followed by PCs.
    """
    for sample in samples:
        if sample.sample_id not in pc_map:
            raise ValueError(
                f"Sample {sample.sample_id!r} is missing from the PC map."
            )

        pcs = np.asarray(pc_map[sample.sample_id], dtype=np.float32)
        if pcs.ndim != 1:
            raise ValueError(
                f"PC values for sample {sample.sample_id!r} must be 1D; got shape {pcs.shape}"
            )

        setattr(sample, 'pcs', pcs.copy())

        parts = []
        if include_sex:
            parts.append(
                np.asarray([encode_sex_for_covariate(getattr(sample, 'sex', None))], dtype=np.float32)
            )
        parts.append(pcs)
        setattr(sample, 'covariates', np.concatenate(parts).astype(np.float32, copy=False))

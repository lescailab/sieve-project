#!/usr/bin/env python3
"""
Render a publishable model architecture diagram for trained SIEVE checkpoints.

This script loads a trained model checkpoint, infers architecture metadata
from the saved state dict, and writes a figure that contains:
  - A graphical representation of the model layers (left)
  - A detailed layer table with input/output shapes, type, parameters (right)

Usage:
    python scripts/render_model_architecture.py \
        --checkpoint outputs/L3_run/best_model.pt \
        --output outputs/L3_run/model_architecture.png

Author: Lescai Lab
"""

import argparse
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a publishable model architecture diagram from a trained checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained model checkpoint (best_model.pt or fold checkpoint)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output file path. If omitted, writes <checkpoint_stem>_architecture.png "
            "in the checkpoint directory."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="One or more output formats (e.g., png pdf svg)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the figure",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution (DPI) for raster outputs",
    )
    parser.add_argument(
        "--allow-unsafe",
        action="store_true",
        help=(
            "Allow unsafe pickle-based checkpoint loading when tensor-only loading fails. "
            "Only use this for trusted checkpoints."
        ),
    )
    return parser.parse_args()


def load_state_dict(checkpoint_path: Path, allow_unsafe: bool = False) -> Dict[str, torch.Tensor]:
    """
    Load a model state dict from a checkpoint file.

    Prefers tensor-only loading (weights_only=True) when supported by the
    installed PyTorch version and falls back to legacy pickle loading when not.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            if allow_unsafe:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            else:
                raise
        else:
            try:
                with safe_globals([np.core.multiarray.scalar]):
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except Exception:
                if allow_unsafe:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                else:
                    raise exc
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Checkpoint does not contain a recognizable model state dict. "
        "Expected 'model_state_dict' or 'state_dict' in the checkpoint."
    )


def find_key_with_suffix(state_dict: Dict[str, torch.Tensor], suffix: str) -> str:
    for key in state_dict:
        if key.endswith(suffix):
            return key
    raise KeyError(f"Could not find key ending with '{suffix}' in state dict.")


def count_params(state_dict: Dict[str, torch.Tensor], prefix: str) -> int:
    return int(
        sum(tensor.numel() for key, tensor in state_dict.items() if key.startswith(prefix))
    )


def infer_architecture(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    is_chunked = any(key.startswith("base_model.") for key in state_dict)
    prefix = "base_model." if is_chunked else ""

    encoder_first_key = find_key_with_suffix(state_dict, f"{prefix}variant_encoder.encoder.0.weight")
    encoder_second_key = find_key_with_suffix(state_dict, f"{prefix}variant_encoder.encoder.4.weight")
    classifier_first_key = find_key_with_suffix(state_dict, f"{prefix}classifier.classifier.0.weight")

    input_dim = state_dict[encoder_first_key].shape[1]
    hidden_dim = state_dict[encoder_first_key].shape[0]
    latent_dim = state_dict[encoder_second_key].shape[0]
    classifier_input_dim = state_dict[classifier_first_key].shape[1]
    classifier_hidden_dim = state_dict[classifier_first_key].shape[0]

    num_covariates = classifier_input_dim % latent_dim
    num_genes = (classifier_input_dim - num_covariates) // latent_dim

    attention_bias_key = find_key_with_suffix(
        state_dict,
        f"{prefix}attention.attention_layers.0.position_bias.weight",
    )
    num_position_buckets = state_dict[attention_bias_key].shape[0]
    num_heads = state_dict[attention_bias_key].shape[1]

    attention_layer_pattern = re.compile(
        re.escape(prefix) + r"attention\.attention_layers\.(\d+)\.query\.weight$"
    )
    attention_layers = {
        int(match.group(1))
        for key in state_dict
        if (match := attention_layer_pattern.match(key))
    }
    num_attention_layers = max(attention_layers) + 1 if attention_layers else 0

    has_chunk_attention = is_chunked and any(
        key.startswith("attention.") and not key.startswith("attention.attention_layers")
        for key in state_dict
    )

    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "classifier_hidden_dim": classifier_hidden_dim,
        "num_covariates": num_covariates,
        "num_genes": num_genes,
        "num_heads": num_heads,
        "num_attention_layers": num_attention_layers,
        "num_position_buckets": num_position_buckets,
        "is_chunked": int(is_chunked),
        "has_chunk_attention": int(has_chunk_attention),
    }


def format_shape(shape: Tuple[str, ...]) -> str:
    return "(" + ", ".join(shape) + ")"


def build_layer_table(
    arch: Dict[str, int], state_dict: Dict[str, torch.Tensor]
) -> Tuple[List[str], List[List[str]], Dict[str, int]]:
    is_chunked = bool(arch["is_chunked"])
    prefix = "base_model." if is_chunked else ""

    total_params = sum(tensor.numel() for tensor in state_dict.values())
    encoder_params = count_params(state_dict, f"{prefix}variant_encoder")
    attention_params = count_params(state_dict, f"{prefix}attention")
    classifier_params = count_params(state_dict, f"{prefix}classifier")
    aggregator_params = count_params(state_dict, f"{prefix}gene_aggregator")
    chunk_attention_params = (
        count_params(state_dict, "attention") if arch["has_chunk_attention"] else 0
    )

    b = "B"
    v = "V"
    g = "G"
    d = "D"
    c = "C" if arch["num_covariates"] > 0 else "0"

    input_shape = format_shape((b, v, str(arch["input_dim"])))
    encoder_output = format_shape((b, v, d))
    attention_output = encoder_output
    aggregator_output = format_shape((b, g, d))

    classifier_input = (
        f"{format_shape((b, g, d))} + {format_shape((b, c))}"
        if arch["num_covariates"] > 0
        else format_shape((b, g, d))
    )
    classifier_output = format_shape((b, "1"))

    columns = [
        "Layer",
        "Type",
        "Input shape",
        "Output shape",
        "Parameters",
        "Characteristics",
    ]

    rows = [
        [
            "Variant Encoder",
            "MLP",
            input_shape,
            encoder_output,
            f"{encoder_params:,}",
            f"Linear {arch['input_dim']}→{arch['hidden_dim']}→{arch['latent_dim']}",
        ],
        [
            "Attention Stack",
            "Position-aware Multi-head",
            encoder_output,
            attention_output,
            f"{attention_params:,}",
            f"{arch['num_attention_layers']} layers, {arch['num_heads']} heads",
        ],
        [
            "Gene Aggregator",
            "Scatter aggregation",
            f"{encoder_output} + gene_ids {format_shape((b, v))}",
            aggregator_output,
            f"{aggregator_params:,}",
            "Permutation-invariant gene pooling",
        ],
        [
            "Phenotype Classifier",
            "Flatten + MLP",
            classifier_input,
            classifier_output,
            f"{classifier_params:,}",
            f"Hidden dim {arch['classifier_hidden_dim']}",
        ],
    ]

    if arch["has_chunk_attention"]:
        rows.insert(
            3,
            [
                "Chunk Aggregation",
                "Attention",
                aggregator_output,
                aggregator_output,
                f"{chunk_attention_params:,}",
                "Learned weighted chunk pooling",
            ],
        )

    totals = {
        "total_params": total_params,
        "encoder_params": encoder_params,
        "attention_params": attention_params,
        "aggregator_params": aggregator_params,
        "classifier_params": classifier_params,
        "chunk_attention_params": chunk_attention_params,
    }

    return columns, rows, totals


def draw_architecture(
    arch: Dict[str, int],
    columns: List[str],
    rows: List[List[str]],
    totals: Dict[str, int],
    output_paths: List[Path],
    title: str,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    ax_left.axis("off")
    ax_right.axis("off")

    box_labels = [
        "Variant Encoder",
        f"Attention Stack\n({arch['num_attention_layers']} layers, {arch['num_heads']} heads)",
        "Gene Aggregator",
    ]
    if arch["has_chunk_attention"]:
        box_labels.append("Chunk Aggregation")
    box_labels.append("Phenotype Classifier")

    num_boxes = len(box_labels)
    y_positions = list(reversed([0.1 + i * (0.8 / (num_boxes - 1)) for i in range(num_boxes)]))

    for i, (label, y) in enumerate(zip(box_labels, y_positions)):
        ax_left.add_patch(
            plt.Rectangle((0.1, y - 0.05), 0.8, 0.1, fill=False, linewidth=1.5)
        )
        ax_left.text(0.5, y, label, ha="center", va="center", fontsize=10)
        if i < num_boxes - 1:
            ax_left.annotate(
                "",
                xy=(0.5, y - 0.05),
                xytext=(0.5, y_positions[i + 1] + 0.05),
                arrowprops=dict(arrowstyle="->", lw=1.2),
            )

    table = ax_right.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    hyperparam_pairs = [
        f"input_dim={arch['input_dim']}",
        f"hidden_dim={arch['hidden_dim']}",
        f"latent_dim={arch['latent_dim']}",
        f"num_genes={arch['num_genes']}",
        f"num_heads={arch['num_heads']}",
        f"num_attention_layers={arch['num_attention_layers']}",
        f"num_position_buckets={arch['num_position_buckets']}",
        f"num_covariates={arch['num_covariates']}",
    ]
    hyperparams = ", ".join(hyperparam_pairs)
    hyperparam_count = len(hyperparam_pairs)

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.text(
        0.5,
        0.02,
        (
            f"Total parameters: {totals['total_params']:,} | "
            f"Hyperparameters ({hyperparam_count}): {hyperparams}"
        ),
        ha="center",
        fontsize=9,
    )

    for output_path in output_paths:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def resolve_output_paths(
    checkpoint_path: Path, output_arg: Optional[str], formats: List[str]
) -> List[Path]:
    if output_arg:
        output_path = Path(output_arg)
        if output_path.suffix:
            ext = output_path.suffix.lstrip(".")
            normalized_formats = [fmt.lower() for fmt in formats]
            if len(normalized_formats) != 1 or normalized_formats[0] != ext.lower():
                raise ValueError(
                    "Conflict between --output and --formats: when --output includes a file "
                    "extension, --formats must contain exactly that single format. "
                    f"Got --output extension '{ext}' and --formats={formats!r}."
                )
            return [output_path]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return [output_path.with_suffix(f".{ext}") for ext in formats]

    default_name = checkpoint_path.stem + "_architecture"
    base_path = checkpoint_path.parent / default_name
    return [base_path.with_suffix(f".{ext}") for ext in formats]


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = load_state_dict(checkpoint_path, allow_unsafe=args.allow_unsafe)
    arch = infer_architecture(state_dict)

    columns, rows, totals = build_layer_table(arch, state_dict)

    title = args.title or f"SIEVE Model Architecture ({checkpoint_path.name})"
    output_paths = resolve_output_paths(checkpoint_path, args.output, args.formats)

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    draw_architecture(arch, columns, rows, totals, output_paths, title, args.dpi)

    print("Wrote architecture figure(s):")
    for path in output_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()

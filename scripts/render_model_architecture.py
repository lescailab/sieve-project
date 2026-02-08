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

Author: Francesco Lescai
"""

import argparse
import inspect
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

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
    supports_weights_only = False
    try:
        supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        supports_weights_only = False

    if supports_weights_only:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            safe_globals = getattr(torch.serialization, "safe_globals", None)
            if safe_globals is None:
                if allow_unsafe:
                    checkpoint = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=False
                    )
                else:
                    raise
            else:
                try:
                    with safe_globals([np.core.multiarray.scalar, np.dtype]):
                        checkpoint = torch.load(
                            checkpoint_path, map_location="cpu", weights_only=True
                        )
                except Exception as retry_exc:
                    if allow_unsafe:
                        checkpoint = torch.load(
                            checkpoint_path, map_location="cpu", weights_only=False
                        )
                    else:
                        raise exc.with_traceback(exc.__traceback__) from retry_exc
    else:
        if not allow_unsafe:
            raise RuntimeError(
                "Installed PyTorch does not support weights_only loading. "
                "Re-run with --allow-unsafe to permit legacy pickle loading."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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


def find_key_with_suffixes(state_dict: Dict[str, torch.Tensor], suffixes: Iterable[str]) -> str:
    for suffix in suffixes:
        for key in state_dict:
            if key.endswith(suffix):
                return key
    raise KeyError(
        "Could not find key ending with any of the expected suffixes: "
        f"{', '.join(suffixes)}."
    )


def count_params(state_dict: Dict[str, torch.Tensor], prefix: str) -> int:
    return int(
        sum(tensor.numel() for key, tensor in state_dict.items() if key.startswith(prefix))
    )


def infer_architecture(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    encoder_prefix_key = find_key_with_suffix(
        state_dict, "variant_encoder.encoder.0.weight"
    )
    prefix = encoder_prefix_key.rsplit("variant_encoder.encoder.0.weight", 1)[0]
    is_chunked = "base_model." in prefix

    encoder_first_key = encoder_prefix_key
    encoder_second_key = find_key_with_suffix(
        state_dict, f"{prefix}variant_encoder.encoder.4.weight"
    )
    classifier_first_key = find_key_with_suffixes(
        state_dict,
        [
            f"{prefix}classifier.classifier.0.weight",
            f"{prefix}classifier.classifier.1.weight",
            f"{prefix}classifier.0.weight",
        ],
    )

    input_dim = state_dict[encoder_first_key].shape[1]
    hidden_dim = state_dict[encoder_first_key].shape[0]
    latent_dim = state_dict[encoder_second_key].shape[0]
    classifier_input_dim = state_dict[classifier_first_key].shape[1]
    classifier_hidden_dim = state_dict[classifier_first_key].shape[0]

    num_covariates = classifier_input_dim % latent_dim
    num_genes = (classifier_input_dim - num_covariates) // latent_dim

    attention_bias_key = find_key_with_suffix(
        state_dict, f"{prefix}attention.attention_layers.0.position_bias.weight"
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
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    # -- colour palette --
    colors = {
        "input": "#E8F5E9",
        "encoder": "#BBDEFB",
        "attention": "#FFE0B2",
        "aggregator": "#E1BEE7",
        "classifier": "#FFCDD2",
        "output": "#F5F5F5",
        "border": "#424242",
        "arrow": "#616161",
        "shape_text": "#1565C0",
    }

    input_dim = arch["input_dim"]
    hidden_dim = arch["hidden_dim"]
    latent_dim = arch["latent_dim"]
    num_heads = arch["num_heads"]
    num_attn = arch["num_attention_layers"]
    num_genes = arch["num_genes"]
    num_pos_buckets = arch["num_position_buckets"]
    cls_hidden = arch["classifier_hidden_dim"]
    num_cov = arch["num_covariates"]
    has_chunk = bool(arch["has_chunk_attention"])

    # -- build block definitions (top → bottom = data flow) --
    blocks: List[Dict] = []

    # Input
    blocks.append({
        "label": "Input",
        "details": (
            f"variant_features  (B, V, {input_dim})\n"
            f"positions  (B, V)\n"
            f"gene_ids  (B, V)"
            + (f"\ncovariates  (B, {num_cov})" if num_cov > 0 else "")
        ),
        "color": colors["input"],
        "params": None,
        "output_shape": None,
    })

    # Variant Encoder
    blocks.append({
        "label": "Variant Encoder",
        "details": (
            f"Linear({input_dim} \u2192 {hidden_dim}) \u2192 ReLU \u2192 LayerNorm \u2192 Dropout\n"
            f"\u2192 Linear({hidden_dim} \u2192 {latent_dim})"
        ),
        "color": colors["encoder"],
        "params": totals["encoder_params"],
        "output_shape": f"(B, V, {latent_dim})",
    })

    # Attention Stack
    blocks.append({
        "label": f"Position-Aware Sparse Attention  \u00d7{num_attn}",
        "details": (
            f"Multi-head self-attention ({num_heads} heads, d_k={latent_dim // num_heads})\n"
            f"+ relative position bias ({num_pos_buckets} buckets)\n"
            f"Residual connection + LayerNorm per layer"
        ),
        "color": colors["attention"],
        "params": totals["attention_params"],
        "output_shape": f"(B, V, {latent_dim})",
    })

    # Gene Aggregator
    blocks.append({
        "label": "Gene Aggregator",
        "details": (
            f"Scatter-based permutation-invariant pooling\n"
            f"Groups V variants into {num_genes:,} genes via gene_ids"
        ),
        "color": colors["aggregator"],
        "params": totals["aggregator_params"],
        "output_shape": f"(B, {num_genes:,}, {latent_dim})",
    })

    # Optional chunk attention
    if has_chunk:
        blocks.append({
            "label": "Chunk Aggregation",
            "details": "Learned weighted chunk pooling",
            "color": colors["attention"],
            "params": totals["chunk_attention_params"],
            "output_shape": f"(B, {num_genes:,}, {latent_dim})",
        })

    # Phenotype Classifier
    flat_dim = num_genes * latent_dim + num_cov
    blocks.append({
        "label": "Phenotype Classifier",
        "details": (
            f"Flatten \u2192 Linear({flat_dim:,} \u2192 {cls_hidden}) \u2192 ReLU \u2192 Dropout\n"
            f"\u2192 Linear({cls_hidden} \u2192 1)"
            + (f"  (+ {num_cov} covariate{'s' if num_cov != 1 else ''} concatenated)" if num_cov > 0 else "")
        ),
        "color": colors["classifier"],
        "params": totals["classifier_params"],
        "output_shape": "(B, 1)",
    })

    # Output
    blocks.append({
        "label": "Output",
        "details": "Logit \u2192 sigmoid for P(case)",
        "color": colors["output"],
        "params": None,
        "output_shape": None,
    })

    # -- layout constants --
    num_blocks = len(blocks)
    fig_width = 10
    box_width = 7.0
    box_x = (fig_width - box_width) / 2
    box_height = 0.9
    gap = 0.55  # space between boxes (for arrow + shape label)
    top_margin = 1.4
    bottom_margin = 1.2
    fig_height = top_margin + num_blocks * box_height + (num_blocks - 1) * gap + bottom_margin

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    # -- draw blocks top-to-bottom --
    y_positions_list: List[float] = []
    for i in range(num_blocks):
        y = fig_height - top_margin - i * (box_height + gap)
        y_positions_list.append(y)

    for i, (block, y_top) in enumerate(zip(blocks, y_positions_list)):
        # Draw rounded box
        box = FancyBboxPatch(
            (box_x, y_top - box_height),
            box_width,
            box_height,
            boxstyle="round,pad=0.12",
            facecolor=block["color"],
            edgecolor=colors["border"],
            linewidth=1.5,
        )
        ax.add_patch(box)

        y_center = y_top - box_height / 2

        # Block label (bold)
        label_y = y_center + 0.22 if block["details"] else y_center
        ax.text(
            fig_width / 2,
            label_y,
            block["label"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            family="sans-serif",
        )

        # Block details (smaller, below label)
        if block["details"]:
            ax.text(
                fig_width / 2,
                y_center - 0.12,
                block["details"],
                ha="center",
                va="center",
                fontsize=8,
                color="#37474F",
                family="monospace",
                linespacing=1.4,
            )

        # Parameter count (right-aligned inside box)
        if block["params"] is not None:
            ax.text(
                box_x + box_width - 0.25,
                y_top - 0.15,
                f"{block['params']:,} params",
                ha="right",
                va="top",
                fontsize=7,
                color="#757575",
                style="italic",
            )

        # Arrow + output shape annotation between blocks
        if i < num_blocks - 1:
            arrow_start_y = y_top - box_height
            arrow_end_y = y_positions_list[i + 1]
            arrow_mid_y = (arrow_start_y + arrow_end_y) / 2

            ax.annotate(
                "",
                xy=(fig_width / 2, arrow_end_y + 0.02),
                xytext=(fig_width / 2, arrow_start_y - 0.02),
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=1.5,
                    color=colors["arrow"],
                    mutation_scale=14,
                ),
            )

            # Shape annotation next to arrow
            if block["output_shape"]:
                ax.text(
                    fig_width / 2 + box_width / 2 + 0.15,
                    arrow_mid_y,
                    block["output_shape"],
                    ha="left",
                    va="center",
                    fontsize=8,
                    color=colors["shape_text"],
                    family="monospace",
                    fontweight="bold",
                )

    # -- title --
    ax.text(
        fig_width / 2,
        fig_height - 0.5,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # -- footer with total params --
    footer = f"Total parameters: {totals['total_params']:,}"
    ax.text(
        fig_width / 2,
        0.35,
        footer,
        ha="center",
        va="center",
        fontsize=10,
        color="#424242",
    )

    plt.tight_layout(pad=0.3)

    for output_path in output_paths:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

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

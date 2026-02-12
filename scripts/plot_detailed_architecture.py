#!/usr/bin/env python3
"""
Generate a detailed computational graph for a trained SIEVE checkpoint using torchviz.

Example:
    python scripts/plot_detailed_architecture.py -c path/to/best_model.pt -o best_model_graph.png --V 64
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except Exception:  # pragma: no cover - fallback path if yaml is unavailable
    yaml = None

try:
    import torch
except Exception as exc:  # pragma: no cover - hard dependency
    print("ERROR: PyTorch is required to run this script.")
    print(f"Import error: {exc}")
    raise


PREFERRED_OUTPUT_KEYS = ("logits", "logit", "pred", "y_hat", "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a detailed computational graph for a trained SIEVE checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained .pt checkpoint.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="sieve_model_graph.png",
        help="Output path or output prefix.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output graph format.",
    )
    parser.add_argument("--B", type=int, default=1, help="Dummy batch size.")
    parser.add_argument("--V", type=int, default=64, help="Dummy number of variants.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the forward pass on.",
    )
    parser.add_argument(
        "--no-params",
        action="store_true",
        help="Do not include model parameters in torchviz.make_dot.",
    )
    return parser.parse_args()


def _find_key_with_suffix(state_dict: Dict[str, torch.Tensor], suffix: str) -> str:
    for key in state_dict:
        if key.endswith(suffix):
            return key
    raise KeyError(f"Could not find key ending with '{suffix}' in state_dict.")


def _find_key_with_suffixes(
    state_dict: Dict[str, torch.Tensor], suffixes: Iterable[str]
) -> str:
    for suffix in suffixes:
        for key in state_dict:
            if key.endswith(suffix):
                return key
    raise KeyError(
        "Could not find a key ending with any expected suffix: "
        + ", ".join(suffixes)
    )


def _to_positive_int(value: Any) -> Optional[int]:
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return None
    if int_value <= 0:
        return None
    return int_value


def _load_checkpoint(checkpoint_path: Path) -> Any:
    supports_weights_only = False
    try:
        supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        supports_weights_only = False

    if supports_weights_only:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return torch.load(checkpoint_path, map_location="cpu")


def _extract_state_dict(checkpoint_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        if (
            "model_state_dict" in checkpoint_obj
            and isinstance(checkpoint_obj["model_state_dict"], dict)
        ):
            return checkpoint_obj["model_state_dict"]
        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            return checkpoint_obj["state_dict"]
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError(
        "Checkpoint does not contain a recognizable model state dict "
        "(expected 'model_state_dict' or 'state_dict')."
    )


def _discover_config_paths(checkpoint_path: Path) -> List[Path]:
    candidates = [
        checkpoint_path.parent / "config.yaml",
        checkpoint_path.parent.parent / "config.yaml",
        checkpoint_path.parent.parent.parent / "config.yaml",
        checkpoint_path.with_name("config.yaml"),
    ]
    deduped: List[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def _load_config_near_checkpoint(
    checkpoint_path: Path,
) -> Tuple[Dict[str, Any], Optional[Path], List[Path]]:
    attempted_paths = _discover_config_paths(checkpoint_path)
    if yaml is None:
        return {}, None, attempted_paths

    for candidate in attempted_paths:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        except Exception:
            continue
        if isinstance(config, dict):
            return config, candidate, attempted_paths
    return {}, None, attempted_paths


def _infer_architecture_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, int]:
    encoder_first_key = _find_key_with_suffix(state_dict, "variant_encoder.encoder.0.weight")
    prefix = encoder_first_key.rsplit("variant_encoder.encoder.0.weight", 1)[0]

    encoder_second_key = _find_key_with_suffix(state_dict, f"{prefix}variant_encoder.encoder.4.weight")
    classifier_first_key = _find_key_with_suffixes(
        state_dict,
        (
            f"{prefix}classifier.classifier.0.weight",
            f"{prefix}classifier.classifier.1.weight",
            f"{prefix}classifier.0.weight",
        ),
    )

    input_dim = int(state_dict[encoder_first_key].shape[1])
    hidden_dim = int(state_dict[encoder_first_key].shape[0])
    latent_dim = int(state_dict[encoder_second_key].shape[0])
    classifier_hidden_dim = int(state_dict[classifier_first_key].shape[0])
    classifier_input_dim = int(state_dict[classifier_first_key].shape[1])

    if latent_dim > 0:
        num_covariates = classifier_input_dim % latent_dim
        num_genes = (classifier_input_dim - num_covariates) // latent_dim
    else:
        num_covariates = 0
        num_genes = 0

    attention_bias_key = _find_key_with_suffix(
        state_dict, f"{prefix}attention.attention_layers.0.position_bias.weight"
    )
    num_position_buckets = int(state_dict[attention_bias_key].shape[0])
    num_heads = int(state_dict[attention_bias_key].shape[1])

    attention_layer_pattern = re.compile(
        re.escape(prefix) + r"attention\.attention_layers\.(\d+)\.query\.weight$"
    )
    layers = {
        int(match.group(1))
        for key in state_dict
        if (match := attention_layer_pattern.match(key))
    }
    num_attention_layers = (max(layers) + 1) if layers else 0

    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "classifier_hidden_dim": classifier_hidden_dim,
        "num_covariates": int(num_covariates),
        "num_genes": int(num_genes),
        "num_heads": num_heads,
        "num_attention_layers": num_attention_layers,
    }


def _determine_num_genes(
    config: Dict[str, Any],
    checkpoint_obj: Any,
    inferred_arch: Dict[str, int],
) -> Tuple[int, str]:
    for key in ("num_genes", "n_genes"):
        value = _to_positive_int(config.get(key))
        if value is not None:
            return value, f"config[{key}]"

    if isinstance(checkpoint_obj, dict):
        for key in ("num_genes", "n_genes"):
            value = _to_positive_int(checkpoint_obj.get(key))
            if value is not None:
                return value, f"checkpoint[{key}]"

        metadata = checkpoint_obj.get("metadata")
        if isinstance(metadata, dict):
            for key in ("num_genes", "n_genes"):
                value = _to_positive_int(metadata.get(key))
                if value is not None:
                    return value, f"checkpoint.metadata[{key}]"

    inferred_num_genes = _to_positive_int(inferred_arch.get("num_genes"))
    if inferred_num_genes is not None:
        return inferred_num_genes, "state_dict inference"

    return 16089, "default"


def _select_tensor_output(model_output: Any) -> torch.Tensor:
    current = model_output
    for _ in range(8):
        if torch.is_tensor(current):
            return current

        if isinstance(current, dict):
            if not current:
                raise ValueError("Model returned an empty dict; cannot choose output tensor.")

            chosen_key = None
            for key in PREFERRED_OUTPUT_KEYS:
                if key in current:
                    chosen_key = key
                    break
            if chosen_key is None:
                chosen_key = next(iter(current.keys()))
            current = current[chosen_key]
            continue

        if isinstance(current, (list, tuple)):
            if len(current) == 0:
                raise ValueError("Model returned an empty list/tuple; cannot choose output tensor.")
            current = current[0]
            continue

        raise TypeError(
            f"Model output type {type(current).__name__} is not supported for graph rendering."
        )

    raise RuntimeError("Could not resolve a tensor output from the model forward result.")


def _run_forward_with_fallbacks(
    model: torch.nn.Module,
    variant_features: torch.Tensor,
    positions: torch.Tensor,
    gene_ids: torch.Tensor,
    covariates: torch.Tensor,
) -> Tuple[Any, str]:
    attempts = [
        (
            "model(variant_features, positions, gene_ids, covariates)",
            lambda: model(variant_features, positions, gene_ids, covariates),
        ),
        (
            "model(variant_features, positions, gene_ids, None, covariates=covariates)",
            lambda: model(
                variant_features,
                positions,
                gene_ids,
                None,
                covariates=covariates,
            ),
        ),
        (
            "model(variant_features, positions, gene_ids, covariates=covariates)",
            lambda: model(
                variant_features,
                positions,
                gene_ids,
                covariates=covariates,
            ),
        ),
    ]

    errors: List[str] = []
    for label, call in attempts:
        try:
            return call(), label
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "All forward-call variants failed:\n  - " + "\n  - ".join(errors)
    )


def _resolve_render_paths(out_arg: str, fmt: str) -> Tuple[Path, Path]:
    out_path = Path(out_arg)
    prefix = out_path.with_suffix("") if out_path.suffix else out_path
    final_path = prefix.with_suffix(f".{fmt}")
    return prefix, final_path


def _is_graphviz_missing_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return ("executable not found" in text) or ("graphviz" in text and "not found" in text) or ("dot" in text and "not found" in text)


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return 1

    try:
        from src.encoding import AnnotationLevel, get_feature_dimension
        from src.models import ChunkedSIEVEModel
        from src.models.sieve import create_sieve_model
    except Exception as exc:
        print("ERROR: Failed to import project model-loading utilities.")
        print("Tried imports:")
        print("  - src.encoding.AnnotationLevel")
        print("  - src.encoding.get_feature_dimension")
        print("  - src.models.sieve.create_sieve_model")
        print("  - src.models.ChunkedSIEVEModel")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Project root used: {PROJECT_ROOT}")
        print(f"Import error: {exc}")
        return 1

    try:
        from torchviz import make_dot
        import graphviz  # noqa: F401 - explicit dependency check
    except Exception as exc:
        print("ERROR: Missing torchviz/graphviz dependency.")
        print("Install with: pip install torchviz graphviz")
        print("Also install the Graphviz system package and ensure the 'dot' binary is on PATH.")
        print(f"Import error: {exc}")
        return 1

    config, used_config_path, attempted_config_paths = _load_config_near_checkpoint(checkpoint_path)

    try:
        checkpoint_obj = _load_checkpoint(checkpoint_path)
        state_dict = _extract_state_dict(checkpoint_obj)
    except Exception as exc:
        print("ERROR: Failed to load checkpoint/state_dict.")
        print(f"Checkpoint: {checkpoint_path}")
        print("Config paths checked:")
        for path in attempted_config_paths:
            print(f"  - {path}")
        print(f"Load error: {type(exc).__name__}: {exc}")
        return 1

    try:
        inferred_arch = _infer_architecture_from_state_dict(state_dict)
    except Exception as exc:
        inferred_arch = {}
        print(f"WARNING: Could not infer full architecture from state_dict ({type(exc).__name__}: {exc}).")

    if "input_dim" not in config:
        level = config.get("level")
        if level is not None:
            try:
                config["input_dim"] = int(get_feature_dimension(AnnotationLevel[str(level)]))
            except Exception:
                pass
    if "input_dim" not in config and "input_dim" in inferred_arch:
        config["input_dim"] = inferred_arch["input_dim"]
    for key in (
        "hidden_dim",
        "latent_dim",
        "classifier_hidden_dim",
        "num_covariates",
        "num_heads",
        "num_attention_layers",
    ):
        if key not in config and key in inferred_arch:
            config[key] = inferred_arch[key]

    n_genes, n_genes_source = _determine_num_genes(config, checkpoint_obj, inferred_arch)

    try:
        base_model = create_sieve_model(config, num_genes=n_genes)
        is_chunked = any(k.startswith("base_model.") for k in state_dict)
        if is_chunked:
            model = ChunkedSIEVEModel(
                base_model=base_model,
                aggregation_method=config.get("aggregation_method", "mean"),
            )
        else:
            model = base_model
        model.load_state_dict(state_dict)
    except Exception as exc:
        print("ERROR: Failed to instantiate and load model weights.")
        print(f"Checkpoint: {checkpoint_path}")
        print("Config source:")
        if used_config_path is None:
            print("  - No config.yaml found near checkpoint; used inferred/default config values.")
        else:
            print(f"  - Loaded: {used_config_path}")
        print("Config paths checked:")
        for path in attempted_config_paths:
            print(f"  - {path}")
        print(f"n_genes source: {n_genes_source} ({n_genes})")
        print(f"Load error: {type(exc).__name__}: {exc}")
        return 1

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    model = model.to(device)
    model.eval()

    gene_high = max(int(n_genes), 1)
    variant_features = torch.randn(args.B, args.V, 71, device=device, requires_grad=True)
    positions = torch.randint(0, 1_000_000, (args.B, args.V), dtype=torch.long, device=device)
    gene_ids = torch.randint(0, gene_high, (args.B, args.V), dtype=torch.long, device=device)
    covariates = torch.randn(args.B, 1, device=device)

    try:
        forward_out, forward_call_used = _run_forward_with_fallbacks(
            model=model,
            variant_features=variant_features,
            positions=positions,
            gene_ids=gene_ids,
            covariates=covariates,
        )
        output_tensor = _select_tensor_output(forward_out)
    except Exception as exc:
        print("ERROR: Model forward pass failed.")
        print(f"Checkpoint: {checkpoint_path}")
        if used_config_path is not None:
            print(f"Config used: {used_config_path}")
        print(f"n_genes source: {n_genes_source} ({n_genes})")
        print(f"Forward error: {type(exc).__name__}: {exc}")
        return 1

    params = None if args.no_params else dict(model.named_parameters())
    dot = make_dot(output_tensor, params=params)
    dot.format = args.format

    render_prefix, expected_final_path = _resolve_render_paths(args.out, args.format)
    render_prefix.parent.mkdir(parents=True, exist_ok=True)

    try:
        rendered_path = Path(dot.render(filename=str(render_prefix), cleanup=True))
    except Exception as exc:
        if _is_graphviz_missing_error(exc):
            print("ERROR: Graphviz rendering failed because dependencies are missing.")
            print("Install with: pip install torchviz graphviz")
            print("Also install the Graphviz system package and ensure the 'dot' binary is on PATH.")
            print(f"Render error: {type(exc).__name__}: {exc}")
            return 1
        print(f"ERROR: Failed to render graph: {type(exc).__name__}: {exc}")
        return 1

    final_output = rendered_path.resolve() if rendered_path.exists() else expected_final_path.resolve()
    print(final_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

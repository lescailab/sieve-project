"""
Shell-level tests for null-baseline protocol matching.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_null_baseline_analysis.sh"


def _make_python_wrapper(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake python executable that logs train/explain commands."""
    log_path = tmp_path / "wrapper.log"
    wrapper_path = tmp_path / "fake_python.py"
    wrapper_path.write_text(
        f"""#!{sys.executable}
import os
from pathlib import Path
import subprocess
import sys
import yaml

REAL = {sys.executable!r}
LOG = Path({str(log_path)!r})

def _value(flag: str) -> str:
    idx = sys.argv.index(flag)
    return sys.argv[idx + 1]

def _touch(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("stub\\n", encoding="utf-8")

if len(sys.argv) > 1 and sys.argv[1] == "-c":
    raise SystemExit(subprocess.call([REAL] + sys.argv[1:]))

LOG.parent.mkdir(parents=True, exist_ok=True)
with LOG.open("a", encoding="utf-8") as handle:
    handle.write(" ".join(sys.argv[1:]) + "\\n")

script = Path(sys.argv[1]).name if len(sys.argv) > 1 else ""
if script == "create_null_baseline.py":
    _touch(_value("--output"))
elif script == "train.py":
    output_dir = Path(_value("--output-dir"))
    experiment_name = _value("--experiment-name")
    run_dir = output_dir / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if "--cv-folds" in sys.argv or "--cv" in sys.argv:
        (run_dir / "fold_0").mkdir(parents=True, exist_ok=True)
        _touch(str(run_dir / "fold_0" / "best_model.pt"))
        with open(run_dir / "cv_results.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump({{"fold_results": [{{"auc": 0.9}}]}}, handle)
    else:
        _touch(str(run_dir / "best_model.pt"))
elif script == "explain.py":
    output_dir = Path(_value("--output-dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _touch(str(output_dir / "sieve_variant_rankings.csv"))
elif script == "compare_attributions.py":
    output_dir = Path(_value("--output-dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
""",
        encoding="utf-8",
    )
    wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IEXEC)
    return wrapper_path, log_path


def _prepare_layout(tmp_path: Path, config: dict) -> dict[str, str]:
    input_data = tmp_path / "preprocessed.pt"
    input_data.write_text("stub\n", encoding="utf-8")

    real_experiment = tmp_path / "real_training"
    real_experiment.mkdir(parents=True, exist_ok=True)
    with open(real_experiment / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    real_results = tmp_path / "real_results"
    real_results.mkdir(parents=True, exist_ok=True)
    (real_results / "sieve_variant_rankings.csv").write_text("chromosome,position\n1,100\n", encoding="utf-8")

    return {
        "INPUT_DATA": str(input_data),
        "REAL_EXPERIMENT": str(real_experiment),
        "REAL_RESULTS": str(real_results),
        "OUTPUT_BASE": str(tmp_path / "outputs"),
        "DEVICE": "cpu",
    }


def test_single_split_protocol_is_preserved(tmp_path):
    wrapper, log_path = _make_python_wrapper(tmp_path)
    env = os.environ.copy()
    env.update(
        _prepare_layout(
            tmp_path,
            config={
                "level": "L3",
                "latent_dim": 32,
                "hidden_dim": 64,
                "num_attention_layers": 1,
                "num_heads": 2,
                "chunk_size": 3000,
                "chunk_overlap": 0,
                "aggregation_method": "mean",
                "lr": 1e-5,
                "lambda_attr": 0.1,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "early_stopping": 5,
                "epochs": 10,
                "seed": 42,
                "genome_build": "GRCh37",
                "val_split": 0.2,
            },
        )
    )
    env["PYTHON"] = str(wrapper)

    subprocess.run(["bash", str(SCRIPT_PATH)], check=True, env=env, cwd=REPO_ROOT)

    log_text = log_path.read_text(encoding="utf-8")
    assert "--val-split 0.2" in log_text
    assert "--cv-folds" not in log_text
    assert "--experiment-dir " + str(Path(env["OUTPUT_BASE"]) / "experiments" / "NULL_BASELINE") in log_text


def test_cv_protocol_uses_best_null_fold(tmp_path):
    wrapper, log_path = _make_python_wrapper(tmp_path)
    env = os.environ.copy()
    env.update(
        _prepare_layout(
            tmp_path,
            config={
                "level": "L3",
                "latent_dim": 32,
                "hidden_dim": 64,
                "num_attention_layers": 1,
                "num_heads": 2,
                "chunk_size": 3000,
                "chunk_overlap": 0,
                "aggregation_method": "mean",
                "lr": 1e-5,
                "lambda_attr": 0.1,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "early_stopping": 5,
                "epochs": 10,
                "seed": 42,
                "genome_build": "GRCh37",
                "cv": 5,
            },
        )
    )
    env["PYTHON"] = str(wrapper)

    subprocess.run(["bash", str(SCRIPT_PATH)], check=True, env=env, cwd=REPO_ROOT)

    log_text = log_path.read_text(encoding="utf-8")
    assert "--cv-folds 5" in log_text
    assert "--experiment-dir " + str(Path(env["OUTPUT_BASE"]) / "experiments" / "NULL_BASELINE" / "fold_0") in log_text

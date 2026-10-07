"""Tests for shared CodeCarbon tracking and CLI integration seams."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import explain, train
from src import co2footprint


class FakeTracker:
    """Minimal tracker test double."""

    def __init__(self, events, *, start_error=None, stop_error=None, **config):
        self.events = events
        self.start_error = start_error
        self.stop_error = stop_error
        self.config = config
        events.append(("init", config))

    def start(self):
        self.events.append(("start", None))
        if self.start_error is not None:
            raise self.start_error

    def stop(self):
        self.events.append(("stop", None))
        if self.stop_error is not None:
            raise self.stop_error


def test_tracking_configures_local_machine_measurement_and_returns_value(tmp_path):
    events = []

    def factory(**config):
        return FakeTracker(events, **config)

    def operation(left, right):
        events.append(("operation", None))
        return left + right

    result = co2footprint.run_with_co2_tracking(
        "training",
        tmp_path,
        operation,
        2,
        3,
        tracker_factory=factory,
    )

    assert result == 5
    assert [event[0] for event in events] == ["init", "start", "operation", "stop"]
    config = events[0][1]
    assert config["project_name"] == "sieve-training"
    assert config["tracking_mode"] == "machine"
    assert config["output_dir"] == str(tmp_path / "co2footprint")
    assert config["output_file"] == "emissions.csv"
    assert config["on_csv_write"] == "append"
    assert config["save_to_file"] is True
    assert config["save_to_api"] is False
    assert config["save_to_logger"] is False
    assert config["save_to_prometheus"] is False
    assert config["save_to_logfire"] is False


def test_operation_exception_is_preserved_and_tracker_stops(tmp_path):
    events = []

    def factory(**config):
        return FakeTracker(events, stop_error=RuntimeError("stop failed"), **config)

    def operation():
        events.append(("operation", None))
        raise ValueError("analysis failed")

    with pytest.raises(ValueError, match="analysis failed"):
        co2footprint.run_with_co2_tracking(
            "explain",
            tmp_path,
            operation,
            tracker_factory=factory,
        )

    assert [event[0] for event in events] == ["init", "start", "operation", "stop"]


@pytest.mark.parametrize("failure_point", ["import", "init", "start", "stop"])
def test_tracking_failures_warn_and_do_not_block_operation(
    tmp_path,
    monkeypatch,
    capsys,
    failure_point,
):
    events = []

    if failure_point == "import":
        def fail_import():
            raise ImportError("not installed")

        monkeypatch.setattr(co2footprint, "_load_tracker_factory", fail_import)
        factory = None
    elif failure_point == "init":
        def factory(**_config):
            raise RuntimeError("init failed")
    else:
        def factory(**config):
            return FakeTracker(
                events,
                start_error=RuntimeError("start failed") if failure_point == "start" else None,
                stop_error=RuntimeError("stop failed") if failure_point == "stop" else None,
                **config,
            )

    called = []

    def operation():
        called.append(True)
        return "completed"

    result = co2footprint.run_with_co2_tracking(
        "training",
        tmp_path,
        operation,
        tracker_factory=factory,
    )

    assert result == "completed"
    assert called == [True]
    assert "WARNING: CO2 footprint tracking" in capsys.readouterr().err


def test_train_main_resolves_experiment_output_and_training_stage(tmp_path, monkeypatch):
    args = SimpleNamespace(
        seed=42,
        preprocessed_data="input.pt",
        vcf=None,
        phenotypes=None,
        pc_map=None,
        num_pcs=0,
        experiment_name=None,
        level="L3",
        output_dir=str(tmp_path),
    )
    captured = {}

    monkeypatch.setattr(train, "parse_args", lambda: args)
    monkeypatch.setattr(train, "set_seed", lambda seed: captured.setdefault("seed", seed))

    def fake_tracking(stage, output_dir, operation, *operation_args):
        captured.update(
            stage=stage,
            output_dir=output_dir,
            operation=operation,
            operation_args=operation_args,
        )

    monkeypatch.setattr(train, "run_with_co2_tracking", fake_tracking)
    train.main()

    expected_output = tmp_path / "L3_run"
    assert captured["seed"] == 42
    assert captured["stage"] == "training"
    assert captured["output_dir"] == expected_output
    assert captured["operation"] is train._run_training
    assert captured["operation_args"] == (args, expected_output)
    assert expected_output.is_dir()


def test_explain_main_uses_requested_output_and_explain_stage(tmp_path, monkeypatch):
    args = SimpleNamespace(output_dir=str(tmp_path))
    captured = {}

    monkeypatch.setattr(explain, "parse_args", lambda: args)

    def fake_tracking(stage, output_dir, operation, *operation_args):
        captured.update(
            stage=stage,
            output_dir=output_dir,
            operation=operation,
            operation_args=operation_args,
        )

    monkeypatch.setattr(explain, "run_with_co2_tracking", fake_tracking)
    explain.main()

    assert captured["stage"] == "explain"
    assert captured["output_dir"] == tmp_path
    assert captured["operation"] is explain._run_explain
    assert captured["operation_args"] == (args, tmp_path)
    assert tmp_path.is_dir()

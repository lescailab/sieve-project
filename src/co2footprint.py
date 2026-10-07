"""Shared CodeCarbon instrumentation for compute-heavy SIEVE commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def _warn(message: str, error: BaseException | None = None) -> None:
    """Emit a tracking warning without using or modifying application logging."""
    detail = f": {error}" if error is not None else ""
    print(f"WARNING: CO2 footprint tracking {message}{detail}", file=sys.stderr)


def _load_tracker_factory() -> Callable[..., Any]:
    """Import CodeCarbon lazily so unavailable telemetry cannot block a command."""
    from codecarbon import EmissionsTracker

    return EmissionsTracker


def run_with_co2_tracking(
    stage: str,
    output_dir: str | Path,
    operation: Callable[..., T],
    *args: Any,
    tracker_factory: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> T:
    """
    Run an operation while recording one CodeCarbon measurement.

    Tracking is deliberately fail-open: setup, start, and stop failures are
    reported to stderr but never replace the operation's return value or
    exception. Raw CodeCarbon rows are appended below the command's existing
    output directory.
    """
    footprint_dir = Path(output_dir) / "co2footprint"
    try:
        footprint_dir.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        _warn("could not create its output directory; continuing without measurement", error)
        return operation(*args, **kwargs)

    if tracker_factory is None:
        try:
            tracker_factory = _load_tracker_factory()
        except Exception as error:
            _warn("is unavailable; continuing without measurement", error)
            return operation(*args, **kwargs)

    try:
        tracker = tracker_factory(
            project_name=f"sieve-{stage}",
            tracking_mode="machine",
            output_dir=str(footprint_dir),
            output_file="emissions.csv",
            on_csv_write="append",
            save_to_file=True,
            save_to_api=False,
            save_to_logger=False,
            save_to_prometheus=False,
            save_to_logfire=False,
            log_level="warning",
        )
    except Exception as error:
        _warn("could not initialize; continuing without measurement", error)
        return operation(*args, **kwargs)

    try:
        tracker.start()
    except Exception as error:
        _warn("could not start; continuing without measurement", error)
        try:
            tracker.stop()
        except Exception:
            pass
        return operation(*args, **kwargs)

    try:
        return operation(*args, **kwargs)
    finally:
        try:
            tracker.stop()
        except Exception as error:
            _warn("could not stop or write its measurement", error)

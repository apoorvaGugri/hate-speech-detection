"""Optional MLflow experiment tracking.

Training must never fail because of tracking, so every MLflow call is wrapped:
if mlflow is missing, disabled in the config, or errors out, we log a warning
and carry on with a no-op tracker.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from src.utils.logger import get_logger

log = get_logger(__name__)


class _Tracker:
    def __init__(self, mlflow_module=None):
        self._mlflow = mlflow_module

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow is None:
            return
        try:
            clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            self._mlflow.log_metrics(clean, step=step)
        except Exception as exc:  # noqa: BLE001 - tracking is best-effort
            log.warning("MLflow log_metrics failed: %s", exc)


@contextmanager
def tracked_run(cfg: dict[str, Any], run_name: str, params: dict[str, Any]) -> Iterator[_Tracker]:
    tracking_cfg = cfg.get("tracking", {})
    mlflow = None
    if tracking_cfg.get("use_mlflow", False):
        try:
            import mlflow as _mlflow

            _mlflow.set_tracking_uri(tracking_cfg.get("tracking_uri", "sqlite:///mlflow.db"))
            _mlflow.set_experiment(tracking_cfg.get("experiment_name", "default"))
            _mlflow.start_run(run_name=run_name)
            _mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow = _mlflow
        except ImportError:
            log.info("mlflow not installed; continuing without experiment tracking.")
        except Exception as exc:  # noqa: BLE001
            log.warning("MLflow unavailable (%s); continuing without tracking.", exc)
    try:
        yield _Tracker(mlflow)
    finally:
        if mlflow is not None:
            try:
                mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass

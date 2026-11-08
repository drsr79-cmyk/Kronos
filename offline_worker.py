"""Secretless offline forecast worker.

Reads a local price snapshot, keeps numeric OHLC columns, and writes a
research export. It does not read market-data API keys, does not call an
ingestion client, and does not download a model. Forecasts are not orders
and carry no promise of winning.

Python code here cannot provide OS network isolation. Run it in a network
namespace or an offline container if that boundary is required.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd

from kronos_safety import (
    FORECAST_LIMITATION,
    ISOLATION_BOUNDARY,
    NO_WIN_PROMISE,
    SECRET_ENV_NAMES,
    SafetyError,
    install_log_redactor,
    limit,
    resolve_local_data_file,
    scrub_text,
    name_looks_secret,
    text_contains_secret,
)

LOGGER = logging.getLogger("kronos.offline_worker")
install_log_redactor(LOGGER)

PRICE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
_SECRET_NAME = ("key", "token", "secret", "password", "credential", "canary", "bearer", "ut")


def _column_is_secret(name: str) -> bool:
    return name_looks_secret(str(name)) or any(part in str(name).lower() for part in _SECRET_NAME)


def sanitize_price_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric price frame. Secret-like columns and cells are dropped."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise SafetyError("invalid_snapshot", "price snapshot must be a non-empty table")
    kept = []
    for column in frame.columns:
        if _column_is_secret(str(column)):
            continue
        if str(column) not in PRICE_COLUMNS and str(column).lower() not in PRICE_COLUMNS:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        kept.append(numeric.rename(str(column).lower()))
    if not kept or not {"open", "high", "low", "close"}.issubset({series.name for series in kept}):
        raise SafetyError("invalid_snapshot", "snapshot is missing numeric open, high, low, and close columns")
    clean = pd.concat(kept, axis=1)
    clean = clean.dropna(subset=["open", "high", "low", "close"])
    if clean.empty:
        raise SafetyError("invalid_snapshot", "snapshot has no numeric price rows")
    if len(clean) > limit("max_rows"):
        raise SafetyError("too_many_rows", "snapshot exceeds the engineering row limit")
    rendered = clean.to_csv(index=False)
    if text_contains_secret(rendered):
        raise SafetyError("secret_in_snapshot", "snapshot still contains secret-like material after sanitizing")
    return clean.reset_index(drop=True)


def load_price_snapshot(path: str, root=None) -> pd.DataFrame:
    resolved = resolve_local_data_file(path, root)
    if resolved.suffix.lower() != ".csv":
        raise SafetyError("unsupported_type", "the offline worker reads local CSV snapshots only")
    frame = pd.read_csv(resolved, nrows=limit("max_rows") + 1)
    return sanitize_price_snapshot(frame)


def model_input_matrix(frame: pd.DataFrame) -> list[list[float]]:
    """Numeric model input. Strings, secrets, and extra columns are not included."""
    clean = sanitize_price_snapshot(frame)
    columns = [name for name in PRICE_COLUMNS if name in clean.columns]
    matrix = []
    for row in clean[columns].itertuples(index=False, name=None):
        numeric_row = []
        for value in row:
            number = float(value)
            if not math.isfinite(number):
                raise SafetyError("invalid_snapshot", "model input must be finite numbers")
            numeric_row.append(number)
        matrix.append(numeric_row)
    blob = json.dumps(matrix)
    if text_contains_secret(blob):
        raise SafetyError("secret_in_model_input", "model input contains secret-like material")
    return matrix


def refuse_outbound(url: str, params=None, headers=None) -> None:
    """The worker has no HTTP client. Arguments are scrubbed before the error is logged."""
    detail = scrub_text(f"url={url} params={params} headers={headers}")
    LOGGER.info("outbound refused: %s", detail)
    raise SafetyError("outbound_disabled", "The offline worker does not send endpoint requests. " + ISOLATION_BOUNDARY)


def export_research_bundle(destination: Path, snapshot: pd.DataFrame, forecast_rows: list[dict] | None = None) -> Path:
    clean = sanitize_price_snapshot(snapshot)
    rows = forecast_rows if forecast_rows is not None else []
    safe_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        safe_row = {}
        for key, value in row.items():
            if _column_is_secret(str(key)):
                continue
            if isinstance(value, str):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                safe_row[str(key)] = number
        if safe_row:
            safe_rows.append(safe_row)
    payload = {
        "limitation": FORECAST_LIMITATION,
        "profit_promise": False,
        "winning_promise": False,
        "notice": NO_WIN_PROMISE,
        "snapshot_rows": int(len(clean)),
        "snapshot_columns": list(clean.columns),
        "forecast_rows": safe_rows,
        "ingestion": "not-deployed",
        "secrets_read": [],
    }
    for name in SECRET_ENV_NAMES:
        if name in payload["secrets_read"]:
            raise SafetyError("secret_export", "export tried to record a secret env name as a value")
    text = scrub_text(json.dumps(payload, ensure_ascii=True))
    if text_contains_secret(text):
        raise SafetyError("secret_export", "export still contains secret-like material")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise SafetyError("symlink_rejected", "refusing to write through a symlink")
    destination.write_text(text + "\n", encoding="utf-8")
    LOGGER.info("wrote research export rows=%s", len(clean))
    return destination


def run_offline_snapshot(snapshot_path: str, export_path: str, root=None) -> dict:
    """Sanitize a local snapshot and write a forecast-only bundle.

    This does not load model weights and does not produce a trading return.
    """
    snapshot = load_price_snapshot(snapshot_path, root)
    matrix = model_input_matrix(snapshot)
    LOGGER.info("sanitized local snapshot rows=%s cols=%s", len(matrix), len(matrix[0]))
    written = export_research_bundle(Path(export_path), snapshot, forecast_rows=[])
    return {
        "export": str(written),
        "rows": len(matrix),
        "limitation": FORECAST_LIMITATION,
        "model_loaded": False,
    }

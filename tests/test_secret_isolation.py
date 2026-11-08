"""Canary-secret tests. The canaries are fake and are not credentials."""

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from ingestion_gate import INGESTION_DEPLOYED, refuse_market_ingestion
from kronos_safety import (
    ISOLATION_BOUNDARY,
    NO_WIN_PROMISE,
    PythonSocketBlock,
    SafetyError,
    reject_active_promotion,
    scrub_text,
)
from offline_worker import (
    export_research_bundle,
    load_price_snapshot,
    model_input_matrix,
    refuse_outbound,
    run_offline_snapshot,
    sanitize_price_snapshot,
)

CANARY = "CANARY_MARKET_KEY_do-not-ship-7f3a"
CSV = "open,high,low,close,volume\n1,2,0.5,1.5,10\n1.1,2.1,0.6,1.6,11\n"


def test_canary_cannot_enter_model_input_export_or_log(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_DATA_API_KEY", CANARY)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "high": [2.0, 2.1],
            "low": [0.5, 0.6],
            "close": [1.5, 1.6],
            "api_key": [CANARY, CANARY],
            "note": [f"token={CANARY}", "ok"],
        }
    )
    clean = sanitize_price_snapshot(frame)
    matrix = model_input_matrix(frame)
    rendered_input = json.dumps(matrix)
    assert CANARY not in rendered_input
    assert CANARY not in clean.to_csv(index=False)
    assert "api_key" not in clean.columns

    destination = tmp_path / "export.json"
    export_research_bundle(
        destination,
        frame,
        forecast_rows=[{"close": 1.25, "api_key": CANARY, "comment": CANARY}],
    )
    exported = destination.read_text(encoding="utf-8")
    assert CANARY not in exported
    assert NO_WIN_PROMISE in exported
    assert '"profit_promise": false' in exported
    assert '"winning_promise": false' in exported

    records = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("kronos.offline_worker")
    handler = _ListHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("leaked %s", CANARY)
        with pytest.raises(SafetyError) as blocked:
            refuse_outbound("https://example.invalid/quote", params={"ut": CANARY}, headers={"Authorization": f"Bearer {CANARY}"})
    finally:
        logger.removeHandler(handler)
    assert blocked.value.code == "outbound_disabled"
    assert CANARY not in str(blocked.value)
    assert records
    assert all(CANARY not in message for message in records)


def test_snapshot_file_canary_is_removed_and_worker_stays_local(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", CANARY)
    root = tmp_path / "data"
    root.mkdir()
    snapshot = root / "bars.csv"
    snapshot.write_text(
        "open,high,low,close,api_key\n1,2,0.5,1.5," + CANARY + "\n1.1,2.1,0.6,1.6," + CANARY + "\n",
        encoding="utf-8",
    )
    result = run_offline_snapshot("bars.csv", str(tmp_path / "out.json"), root)
    body = Path(result["export"]).read_text(encoding="utf-8")
    assert CANARY not in body
    assert result["model_loaded"] is False
    loaded = load_price_snapshot("bars.csv", root)
    assert list(loaded.columns) == ["open", "high", "low", "close"]


def test_ingestion_is_not_deployed_and_examples_have_no_market_token():
    assert INGESTION_DEPLOYED is False
    with pytest.raises(SafetyError) as blocked:
        refuse_market_ingestion("eastmoney")
    assert blocked.value.code == "ingestion_not_deployed"
    assert CANARY not in str(blocked.value)
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*.py"):
        if ".git" in path.parts or "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "push2his.eastmoney.com" not in text
        assert "requests.get(" not in text
        assert "import akshare" not in text


def test_python_socket_block_is_not_os_isolation():
    assert "not an OS" in ISOLATION_BOUNDARY or "OS-enforced" in ISOLATION_BOUNDARY
    import socket

    original = socket.socket
    with PythonSocketBlock():
        blocked = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(SafetyError) as exc:
            blocked.connect(("127.0.0.1", 9))
        assert exc.value.code == "python_network_block"
        blocked.close()
    assert socket.socket is original


def test_learning_cannot_promote_active_policy():
    with pytest.raises(SafetyError) as policy:
        reject_active_promotion("kronos_safety.py")
    assert policy.value.code == "policy_frozen"
    with pytest.raises(SafetyError):
        reject_active_promotion("model/kronos.py")
    with pytest.raises(SafetyError):
        reject_active_promotion("webui/app.py")
    reject_active_promotion("candidates/trial_forecast.py")


def test_scrub_removes_env_canary_and_inline_assignment(monkeypatch):
    monkeypatch.setenv("COMET_API_KEY", CANARY)
    raw = f"connecting with api_key={CANARY} and leftover {CANARY}"
    cleaned = scrub_text(raw)
    assert CANARY not in cleaned
    assert "api_key=" not in cleaned or "[redacted]" in cleaned

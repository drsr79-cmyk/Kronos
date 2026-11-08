"""Web UI security tests. The model library is not required and no weights are loaded."""

import importlib.util
import socket
from pathlib import Path

import pytest

CSV = "open,high,low,close,volume\n1,2,0.5,1.5,10\n1.1,2.1,0.6,1.6,11\n1.2,2.2,0.7,1.7,12\n"


def _load_app(monkeypatch, data_dir: Path):
    monkeypatch.setenv("KRONOS_DATA_DIR", str(data_dir))
    monkeypatch.delenv("KRONOS_WEBUI_TOKEN", raising=False)
    monkeypatch.delenv("KRONOS_ALLOW_NONLOCAL_BIND", raising=False)
    monkeypatch.delenv("KRONOS_ALLOW_HUB_DOWNLOAD", raising=False)
    app_path = Path(__file__).resolve().parents[1] / "webui" / "app.py"
    spec = importlib.util.spec_from_file_location("kronos_webui_under_test", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def ui(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "bars.csv").write_text(CSV, encoding="utf-8")
    module = _load_app(monkeypatch, data)
    client = module.app.test_client()
    return module, client, data


def _authorize(client):
    page = client.get("/")
    assert page.status_code == 200
    return {"X-Kronos-Request": "1"}


def test_page_is_local_and_has_no_cdn_or_token(ui):
    _module, client, _data = ui
    page = client.get("/")
    body = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "cdn.plot.ly" not in body
    assert "jsdelivr" not in body
    assert "not paper orders" in body
    cookie = page.headers.get("Set-Cookie", "")
    assert "kronos_session=" in cookie
    assert "HttpOnly" in cookie
    token = cookie.split("kronos_session=", 1)[1].split(";", 1)[0]
    assert token not in body
    assert "Access-Control-Allow-Origin" not in page.headers
    assert "Access-Control-Allow-Private-Network" not in page.headers
    assert "frame-ancestors 'none'" in page.headers.get("Content-Security-Policy", "")


def test_api_requires_local_session_and_custom_header(ui):
    _module, client, _data = ui
    denied = client.get("/api/data-files")
    assert denied.status_code == 401
    headers = _authorize(client)
    allowed = client.get("/api/data-files", headers=headers)
    assert allowed.status_code == 200
    assert allowed.get_json()[0]["path"] == "bars.csv"
    assert "/" not in allowed.get_json()[0]["path"]


def test_remote_address_is_rejected(ui):
    _module, client, _data = ui
    response = client.get("/", environ_overrides={"REMOTE_ADDR": "203.0.113.8"})
    assert response.status_code == 403


def test_ssrf_symlink_and_traversal_do_not_read(ui, tmp_path, monkeypatch):
    module, client, data = ui
    outside = tmp_path / "secret.csv"
    outside.write_text("open,high,low,close\n9,9,9,9\n", encoding="utf-8")
    (data / "link.csv").symlink_to(outside)
    headers = _authorize(client)

    def refuse(*_args, **_kwargs):
        raise AssertionError("network connection attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    cases = {
        "http://127.0.0.1:9/internal.csv": "rejected_url",
        "file:///etc/passwd.csv": "rejected_url",
        "../secret.csv": "path_escape",
        str(outside): "path_escape",
        "link.csv": "symlink_rejected",
    }
    for raw, code in cases.items():
        response = client.post("/api/load-data", json={"file_path": raw}, headers=headers)
        assert response.status_code == 400
        body = response.get_json()
        assert body["error"] == code
        assert "Traceback" not in response.get_data(as_text=True)
        assert "9,9,9,9" not in response.get_data(as_text=True)

    ok = client.post("/api/load-data", json={"file_path": "bars.csv"}, headers=headers)
    assert ok.status_code == 200
    assert ok.get_json()["data_info"]["rows"] == 3
    assert module.app.config["MAX_CONTENT_LENGTH"] == 1024 * 1024


def test_predict_bounds_and_private_network_preflight(ui):
    _module, client, _data = ui
    headers = _authorize(client)
    huge = client.post(
        "/api/predict",
        json={"file_path": "bars.csv", "lookback": 2, "pred_len": 10**8, "sample_count": 1},
        headers=headers,
    )
    assert huge.status_code == 400
    assert huge.get_json()["error"] == "parameter_out_of_range"
    preflight = client.options(
        "/api/predict",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Private-Network": "true",
            "X-Kronos-Request": "1",
        },
    )
    assert "Access-Control-Allow-Private-Network" not in preflight.headers
    assert "Access-Control-Allow-Origin" not in preflight.headers


def test_secret_column_does_not_reach_the_response(ui):
    _module, client, data = ui
    canary = "CANARY_MARKET_KEY_do-not-ship-7f3a"
    (data / "bars.csv").write_text(
        "open,high,low,close,api_key\n1,2,0.5,1.5," + canary + "\n1.1,2.1,0.6,1.6," + canary + "\n1.2,2.2,0.7,1.7," + canary + "\n",
        encoding="utf-8",
    )
    headers = _authorize(client)
    response = client.post("/api/load-data", json={"file_path": "bars.csv"}, headers=headers)
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert canary not in body
    assert "api_key" not in response.get_json()["data_info"]["columns"]
    assert "no promise of profit" in response.get_json()["limitation"]


def test_model_endpoint_does_not_download(ui, monkeypatch):
    _module, client, _data = ui
    headers = _authorize(client)

    def refuse(*_args, **_kwargs):
        raise AssertionError("network connection attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    response = client.post("/api/load-model", json={"model_key": "kronos-small", "device": "cpu"}, headers=headers)
    assert response.status_code == 400
    assert response.get_json()["error"] in {"model_unavailable", "invalid_model", "hub_download_disabled"}
    weird = client.post("/api/load-model", json={"model_key": "kronos-small", "device": "cuda;rm"}, headers=headers)
    assert weird.status_code == 400
    assert weird.get_json()["error"] == "invalid_device"

"""Security policy tests. No model weights, datasets, or network services."""

import json
import os
import pickle
import socket
from pathlib import Path

import pytest

import kronos_safety
from kronos_safety import (
    PINNED_REVISIONS,
    SafetyError,
    comet_experiment_kwargs,
    enforce_inference_budget,
    list_data_files,
    load_pickle,
    prepare_pretrained_load,
    redact_mapping,
    resolve_local_data_file,
    safe_named_file,
    safe_symbol,
    webui_runtime_settings,
)


CSV = "open,high,low,close,volume\n1,2,0.5,1.5,10\n1.1,2.1,0.6,1.6,11\n"


def _config(directory: Path, **overrides):
    payload = {
        "s1_bits": 10,
        "s2_bits": 10,
        "n_layers": 4,
        "d_model": 128,
        "n_heads": 4,
        "ff_dim": 256,
        "ffn_dropout_p": 0.0,
        "attn_dropout_p": 0.0,
        "resid_dropout_p": 0.0,
        "token_dropout_p": 0.0,
        "learn_te": True,
    }
    payload.update(overrides)
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_url_and_traversal_and_symlink_are_rejected(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("KRONOS_DATA_DIR", str(root))
    outside = tmp_path / "secret.csv"
    outside.write_text(CSV, encoding="utf-8")
    (root / "ok.csv").write_text(CSV, encoding="utf-8")
    (root / "link.csv").symlink_to(outside)

    with pytest.raises(SafetyError) as url_error:
        resolve_local_data_file("http://127.0.0.1/internal.csv", root)
    assert url_error.value.code == "rejected_url"

    with pytest.raises(SafetyError) as file_url:
        resolve_local_data_file("file:///tmp/x.csv", root)
    assert file_url.value.code == "rejected_url"

    with pytest.raises(SafetyError) as escape:
        resolve_local_data_file("../secret.csv", root)
    assert escape.value.code in {"path_escape", "not_found"}

    with pytest.raises(SafetyError) as absolute:
        resolve_local_data_file(str(outside), root)
    assert absolute.value.code == "path_escape"

    with pytest.raises(SafetyError) as link:
        resolve_local_data_file("link.csv", root)
    assert link.value.code == "symlink_rejected"

    resolved = resolve_local_data_file("ok.csv", root)
    assert resolved == (root / "ok.csv").resolve()
    assert resolved.name == "ok.csv"
    names = [item["path"] for item in list_data_files(root)]
    assert names == ["ok.csv"]
    assert str(outside) not in names[0]


def test_metadata_url_and_query_string_do_not_pass(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(SafetyError):
        resolve_local_data_file("http://169.254.169.254/latest/meta-data", root)
    with pytest.raises(SafetyError):
        resolve_local_data_file("https://example.invalid/a.csv?x=1", root)


def test_inference_budget_rejects_resource_bombs():
    ok = enforce_inference_budget(lookback=400, pred_len=120, sample_count=1, temperature=1.0, top_p=0.9)
    assert ok["pred_len"] == 120
    with pytest.raises(SafetyError) as huge:
        enforce_inference_budget(lookback=10, pred_len=10**9, sample_count=1)
    assert huge.value.code == "parameter_out_of_range"
    with pytest.raises(SafetyError):
        enforce_inference_budget(lookback=10, pred_len=8, sample_count=10**6)
    with pytest.raises(SafetyError):
        enforce_inference_budget(lookback=True, pred_len=8, sample_count=1)
    with pytest.raises(SafetyError):
        enforce_inference_budget(lookback=10, pred_len=8, sample_count=1, temperature=float("inf"))


def test_env_limit_cannot_exceed_hard_cap(monkeypatch):
    monkeypatch.setenv("KRONOS_MAX_PRED_LEN", "999999")
    assert kronos_safety.limit("max_pred_len") == kronos_safety._HARD_LIMITS["max_pred_len"]


def test_pickle_is_disabled_by_default(tmp_path, monkeypatch):
    payload = tmp_path / "data.pkl"
    with payload.open("wb") as handle:
        pickle.dump({"value": 1}, handle)
    monkeypatch.delenv("KRONOS_ALLOW_PICKLE", raising=False)
    with pytest.raises(SafetyError) as blocked:
        load_pickle(payload)
    assert blocked.value.code == "pickle_disabled"
    monkeypatch.setenv("KRONOS_ALLOW_PICKLE", "1")
    assert load_pickle(payload)["value"] == 1


def test_local_model_requires_safetensors_and_rejects_pickle_weights(tmp_path, monkeypatch):
    monkeypatch.delenv("KRONOS_ALLOW_HUB_DOWNLOAD", raising=False)
    model_dir = tmp_path / "Kronos-small"
    model_dir.mkdir()
    _config(model_dir)
    (model_dir / "model.safetensors").write_bytes(b"synthetic-weights")
    path, kwargs = prepare_pretrained_load(model_dir, {})
    assert path == str(model_dir.resolve())
    assert kwargs["local_files_only"] is True

    (model_dir / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(SafetyError) as unsafe:
        prepare_pretrained_load(model_dir, {})
    assert unsafe.value.code == "unsafe_weight_format"


def test_hostile_config_and_symlink_model_are_rejected(tmp_path):
    model_dir = tmp_path / "bad"
    model_dir.mkdir()
    _config(model_dir, s1_bits=30)
    (model_dir / "model.safetensors").write_bytes(b"synthetic-weights")
    with pytest.raises(SafetyError) as hostile:
        prepare_pretrained_load(model_dir, {})
    assert hostile.value.code == "invalid_model"

    linked = tmp_path / "linked"
    real = tmp_path / "real"
    real.mkdir()
    _config(real)
    (real / "model.safetensors").write_bytes(b"synthetic-weights")
    linked.symlink_to(real)
    with pytest.raises(SafetyError) as link:
        prepare_pretrained_load(linked, {})
    assert link.value.code == "symlink_rejected"


def test_hub_download_is_off_and_revision_is_pinned(monkeypatch):
    monkeypatch.delenv("KRONOS_ALLOW_HUB_DOWNLOAD", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    repo = "NeoQuasar/Kronos-small"
    with pytest.raises(SafetyError) as blocked:
        prepare_pretrained_load(repo, {})
    assert blocked.value.code == "hub_download_disabled"

    monkeypatch.setenv("KRONOS_ALLOW_HUB_DOWNLOAD", "1")
    with pytest.raises(SafetyError) as offline:
        prepare_pretrained_load(repo, {})
    assert offline.value.code == "hub_offline"

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    path, kwargs = prepare_pretrained_load(repo, {})
    assert path == repo
    assert kwargs["revision"] == PINNED_REVISIONS[repo]
    with pytest.raises(SafetyError) as mismatch:
        prepare_pretrained_load(repo, {"revision": "deadbeef"})
    assert mismatch.value.code == "unpinned_revision"
    with pytest.raises(SafetyError):
        prepare_pretrained_load("NeoQuasar/Not-A-Reviewed-Model", {})


def test_symbol_output_cannot_escape(tmp_path):
    with pytest.raises(SafetyError):
        safe_symbol("../../outside")
    with pytest.raises(SafetyError):
        safe_named_file(tmp_path, "/tmp/abs", "", "_stock_data.csv")
    dest = safe_named_file(tmp_path, "000001", "", "_stock_data.csv")
    assert dest.parent == tmp_path.resolve()
    assert dest.name == "000001_stock_data.csv"


def test_comet_requires_opt_in_and_redacts_secrets(monkeypatch):
    monkeypatch.delenv("KRONOS_ALLOW_COMET", raising=False)
    monkeypatch.setenv("COMET_API_KEY", "super-secret-value")
    assert comet_experiment_kwargs({"project_name": "demo", "workspace": "lab"}) is None
    monkeypatch.setenv("KRONOS_ALLOW_COMET", "1")
    monkeypatch.setenv("COMET_PROJECT_NAME", "demo")
    monkeypatch.setenv("COMET_WORKSPACE", "lab")
    kwargs = comet_experiment_kwargs({})
    assert kwargs["api_key"] == "super-secret-value"
    redacted = redact_mapping({"comet_config": {"api_key": "super-secret-value"}, "seed": 1})
    assert redacted["comet_config"]["api_key"] == "***"
    assert "super-secret-value" not in json.dumps(redacted)


def test_webui_bind_is_loopback_without_debugger(monkeypatch):
    monkeypatch.delenv("KRONOS_WEBUI_HOST", raising=False)
    monkeypatch.delenv("KRONOS_WEBUI_DEBUG", raising=False)
    monkeypatch.delenv("KRONOS_ALLOW_NONLOCAL_BIND", raising=False)
    settings = webui_runtime_settings()
    assert settings["host"] == "127.0.0.1"
    assert settings["debug"] is False
    assert settings["use_reloader"] is False
    assert settings["load_dotenv"] is False
    monkeypatch.setenv("KRONOS_WEBUI_HOST", "0.0.0.0")
    with pytest.raises(SafetyError):
        webui_runtime_settings()
    monkeypatch.setenv("KRONOS_ALLOW_NONLOCAL_BIND", "1")
    monkeypatch.setenv("KRONOS_WEBUI_TOKEN", "short")
    with pytest.raises(SafetyError):
        webui_runtime_settings()
    monkeypatch.setenv("KRONOS_WEBUI_TOKEN", "a" * 16)
    exposed = webui_runtime_settings()
    assert exposed["host"] == "0.0.0.0"
    assert exposed["debug"] is False


def test_offline_csv_read_does_not_open_a_connection(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    (root / "bars.csv").write_text(CSV, encoding="utf-8")

    def refuse(*_args, **_kwargs):
        raise AssertionError("network connection attempted")

    monkeypatch.setattr(socket, "create_connection", refuse)
    resolved = resolve_local_data_file("bars.csv", root)
    text = resolved.read_text(encoding="utf-8")
    assert "open,high,low,close" in text


def test_source_does_not_keep_removed_footguns():
    root = Path(__file__).resolve().parents[1]
    app_source = (root / "webui" / "app.py").read_text(encoding="utf-8")
    html = (root / "webui" / "templates" / "index.html").read_text(encoding="utf-8")
    start = (root / "webui" / "start.sh").read_text(encoding="utf-8")
    web_requirements = (root / "webui" / "requirements.txt").read_text(encoding="utf-8")
    root_requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    dataset = (root / "finetune" / "dataset.py").read_text(encoding="utf-8")
    assert "0.0.0.0" not in app_source
    assert "debug=True" not in app_source
    assert "flask_cors" not in app_source
    assert "cdn.plot.ly" not in html
    assert "jsdelivr" not in html
    assert "pip install" not in start
    assert "flask-cors==" not in web_requirements
    assert "CORS(" not in app_source
    assert "torch>=2.6.0" in root_requirements
    assert "pickle.load" not in dataset
    assert "YOUR_COMET_API_KEY" not in (root / "finetune" / "config.py").read_text(encoding="utf-8")

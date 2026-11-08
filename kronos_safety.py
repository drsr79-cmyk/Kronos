"""Engineering safeguards for offline Kronos research.

Limits here are process and memory guards. They are not investment
policy, position limits, or a paper-trading simulator. Forecasts are
not orders and not portfolio returns.
"""

from __future__ import annotations

import hmac
import json
import os
import pickle
import re
from pathlib import Path

FORECAST_LIMITATION = (
    "Research forecast only. This output is not a paper order, "
    "a paper trade, or a portfolio return. There is no promise of profit or of winning."
)
NO_WIN_PROMISE = "No promise of profit or of winning."
# Python-level socket patches apply only inside this interpreter.
# They are not an OS network namespace, seccomp filter, firewall, or container egress policy.
ISOLATION_BOUNDARY = (
    "Python network blocking is in-process only. "
    "OS-enforced deployment isolation is a separate control and is not provided by this repository."
)
SECRET_ENV_NAMES = (
    "COMET_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "KRONOS_WEBUI_TOKEN",
    "MARKET_DATA_API_KEY",
    "EASTMONEY_UT",
    "AKSHARE_TOKEN",
    "API_KEY",
    "API_SECRET",
)
ACTIVE_POLICY_FILES = frozenset(
    {
        "kronos_safety.py",
        "offline_worker.py",
        "ingestion_gate.py",
        "model/kronos.py",
        "webui/app.py",
    }
)
_SECRET_FIELD_RE = re.compile(
    r"(?i)\b(api[_-]?key|api[_-]?secret|access[_-]?token|secret|password|bearer|eastmoney[_-]?ut)\b\s*[:=]\s*\S+"
)
_CANARY_RE = re.compile(r"CANARY_[A-Za-z0-9_-]{4,}")

# Reviewed Hub commits observed for NeoQuasar Kronos artifacts.
# Hub download stays off unless a process sets KRONOS_ALLOW_HUB_DOWNLOAD=1
# before Kronos is imported.
PINNED_REVISIONS = {
    "NeoQuasar/Kronos-mini": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
    "NeoQuasar/Kronos-small": "901c26c1332695a2a8f243eb2f37243a37bea320",
    "NeoQuasar/Kronos-base": "2b554741eca47781b64468546e77fef3e85130e6",
    "NeoQuasar/Kronos-Tokenizer-2k": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
    "NeoQuasar/Kronos-Tokenizer-base": "0e0117387f39004a9016484a186a908917e22426",
}

# Defaults can be lowered, or raised up to the hard cap, with KRONOS_MAX_*.
_DEFAULT_LIMITS = {
    "max_lookback": 2048,
    "max_pred_len": 512,
    "max_sample_count": 8,
    "max_batch": 1024,
    "max_rows": 50000,
    "max_file_bytes": 32 * 1024 * 1024,
}
_HARD_LIMITS = {
    "max_lookback": 8192,
    "max_pred_len": 2048,
    "max_sample_count": 32,
    "max_batch": 4096,
    "max_rows": 200000,
    "max_file_bytes": 256 * 1024 * 1024,
}
MAX_WEIGHT_BYTES = 2 * 1024 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
_ARCH_CAPS = {
    "s1_bits": 16,
    "s2_bits": 16,
    "n_layers": 48,
    "n_enc_layers": 24,
    "n_dec_layers": 24,
    "d_model": 4096,
    "ff_dim": 16384,
    "n_heads": 64,
    "group_size": 1024,
}
_SYMBOL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}")
_DEVICE_RE = re.compile(r"cpu|mps|cuda(?::[0-7])?")
_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")


class SafetyError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_data_root() -> Path:
    raw = os.environ.get("KRONOS_DATA_DIR")
    if raw:
        return Path(raw).expanduser()
    return repo_root() / "data"


def limit(name: str) -> int:
    if name not in _DEFAULT_LIMITS:
        raise KeyError(name)
    default = _DEFAULT_LIMITS[name]
    hard = _HARD_LIMITS[name]
    raw = os.environ.get("KRONOS_" + name.upper())
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 1:
        return default
    return min(value, hard)


def apply_hub_environment() -> None:
    """Disable Hub telemetry. Keep downloads offline unless explicitly allowed."""
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    if os.environ.get("KRONOS_ALLOW_HUB_DOWNLOAD") != "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _reject_url(value: str) -> None:
    lowered = value.strip().lower()
    if "://" in value or lowered.startswith("file:") or lowered.startswith("\\\\"):
        raise SafetyError("rejected_url", "remote data sources are disabled")
    if "\x00" in value:
        raise SafetyError("invalid_data_path", "invalid data path")


def resolve_local_data_file(user_path: str, root: Path | None = None) -> Path:
    if not isinstance(user_path, str) or not user_path.strip():
        raise SafetyError("invalid_data_path", "invalid data path")
    raw = user_path.strip()
    _reject_url(raw)
    root_path = root or default_data_root()
    root_path.mkdir(parents=True, exist_ok=True)
    root_real = root_path.resolve()
    if root_real.is_symlink() or not root_real.is_dir():
        raise SafetyError("invalid_data_root", "data directory is unavailable")

    candidate = Path(os.path.expanduser(raw))
    if candidate.is_absolute():
        try:
            rel = Path(os.path.normpath(str(candidate))).relative_to(root_real)
        except ValueError:
            raise SafetyError("path_escape", "data path is outside the data directory") from None
    else:
        rel = Path(os.path.normpath(raw))
        if rel.is_absolute() or ".." in rel.parts:
            raise SafetyError("path_escape", "data path is outside the data directory")

    current = root_real
    if not rel.parts:
        raise SafetyError("invalid_data_path", "invalid data path")
    for part in rel.parts:
        if part in {"", ".", ".."}:
            raise SafetyError("path_escape", "data path is outside the data directory")
        current = current / part
        if current.is_symlink():
            raise SafetyError("symlink_rejected", "symlinked data files are disabled")
        if not current.exists():
            raise SafetyError("not_found", "data file was not found")
    if not current.is_file():
        raise SafetyError("invalid_data_path", "data path is not a file")
    if current.suffix.lower() not in {".csv", ".feather"}:
        raise SafetyError("unsupported_type", "only .csv and .feather files are allowed")
    if name_looks_secret(current.name) or text_contains_secret(current.name):
        raise SafetyError("secret_in_name", "data file name contains secret-like material")
    if current.stat().st_size > limit("max_file_bytes"):
        raise SafetyError("file_too_large", "data file exceeds the engineering size limit")
    return current


def list_data_files(root: Path | None = None) -> list[dict]:
    root_path = root or default_data_root()
    root_path.mkdir(parents=True, exist_ok=True)
    root_real = root_path.resolve()
    found = []
    for entry in sorted(root_real.iterdir(), key=lambda p: p.name):
        if entry.is_symlink() or not entry.is_file():
            continue
        if name_looks_secret(entry.name) or text_contains_secret(entry.name):
            continue
        if entry.suffix.lower() not in {".csv", ".feather"}:
            continue
        size = entry.stat().st_size
        if size > limit("max_file_bytes"):
            continue
        found.append(
            {
                "name": entry.name,
                "path": entry.name,
                "size": f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB",
            }
        )
        if len(found) >= 200:
            break
    return found


def _as_int(name: str, value, lo: int, hi: int) -> int:
    if isinstance(value, bool) or isinstance(value, (list, dict)):
        raise SafetyError("invalid_parameter", "invalid forecast parameter")
    try:
        if isinstance(value, str) and value.strip() == "":
            raise ValueError
        number = int(value)
    except (TypeError, ValueError):
        raise SafetyError("invalid_parameter", "invalid forecast parameter") from None
    if number < lo or number > hi:
        raise SafetyError("parameter_out_of_range", f"{name} is outside the engineering limit")
    return number


def _as_float(name: str, value, lo: float, hi: float) -> float:
    if isinstance(value, bool) or isinstance(value, (list, dict)):
        raise SafetyError("invalid_parameter", "invalid forecast parameter")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SafetyError("invalid_parameter", "invalid forecast parameter") from None
    if number != number or number in (float("inf"), float("-inf")) or number < lo or number > hi:
        raise SafetyError("parameter_out_of_range", f"{name} is outside the engineering limit")
    return number


def enforce_inference_budget(
    *,
    lookback: int,
    pred_len: int,
    sample_count: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    batch_size: int = 1,
) -> dict:
    """Reject forecasts that can exhaust memory. Not an investment limit."""
    checked = {
        "lookback": _as_int("lookback", lookback, 1, limit("max_lookback")),
        "pred_len": _as_int("pred_len", pred_len, 1, limit("max_pred_len")),
        "sample_count": _as_int("sample_count", sample_count, 1, limit("max_sample_count")),
        "batch_size": _as_int("batch_size", batch_size, 1, limit("max_batch")),
        "temperature": _as_float("temperature", temperature, 0.01, 5.0),
        "top_p": _as_float("top_p", top_p, 0.0, 1.0),
    }
    return checked


def validate_device(device: str) -> str:
    if not isinstance(device, str) or _DEVICE_RE.fullmatch(device.strip()) is None:
        raise SafetyError("invalid_device", "device must be cpu, mps, or cuda with an optional index")
    return device.strip()


def safe_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None or ".." in symbol:
        raise SafetyError("invalid_symbol", "symbol must be a short ticker-like token")
    return symbol


def safe_named_file(directory, label: str, prefix: str, suffix: str) -> Path:
    symbol = safe_symbol(label)
    if "/" in prefix or "\\" in prefix or "/" in suffix or "\\" in suffix:
        raise SafetyError("invalid_filename", "invalid output name")
    filename = f"{prefix}{symbol}{suffix}"
    if filename.startswith(".") or Path(filename).name != filename:
        raise SafetyError("invalid_filename", "invalid output name")
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    root_real = root.resolve()
    dest = root_real / filename
    if dest.is_symlink():
        raise SafetyError("symlink_rejected", "symlinked output paths are disabled")
    try:
        dest.resolve(strict=False).relative_to(root_real)
    except ValueError:
        raise SafetyError("path_escape", "output path is outside the output directory") from None
    return dest


def load_pickle(path):
    if os.environ.get("KRONOS_ALLOW_PICKLE") != "1":
        raise SafetyError(
            "pickle_disabled",
            "pickle loading is disabled. Set KRONOS_ALLOW_PICKLE=1 only for local files you created.",
        )
    file_path = Path(path)
    _reject_url(str(file_path))
    if file_path.is_symlink() or not file_path.is_file():
        raise SafetyError("invalid_pickle", "pickle path must be a regular local file")
    if file_path.stat().st_size > limit("max_file_bytes"):
        raise SafetyError("file_too_large", "pickle file exceeds the engineering size limit")
    with file_path.open("rb") as handle:
        return pickle.load(handle)


def secret_values(extra: tuple[str, ...] | list[str] = ()) -> list[str]:
    found = []
    for name in SECRET_ENV_NAMES:
        raw = os.environ.get(name, "")
        if isinstance(raw, str) and len(raw.strip()) >= 6:
            found.append(raw.strip())
    for item in extra:
        if isinstance(item, str) and len(item) >= 6:
            found.append(item)
    return found


def scrub_text(value, extra: tuple[str, ...] | list[str] = ()) -> str:
    text = value if isinstance(value, str) else str(value)
    for secret in secret_values(extra):
        text = text.replace(secret, "[redacted]")
    text = _SECRET_FIELD_RE.sub("[redacted]", text)
    text = _CANARY_RE.sub("[redacted]", text)
    return text


_SECRET_NAME_RE = re.compile(r"(?i)(api[_-]?key|api[_-]?secret|access[_-]?token|secret|password|bearer|canary|credential)")


def name_looks_secret(name: str) -> bool:
    return _SECRET_NAME_RE.search(str(name)) is not None


def text_contains_secret(value, extra: tuple[str, ...] | list[str] = ()) -> bool:
    """Detect secret-like values. Do not use this on ordinary field names such as secrets_read."""
    raw = value if isinstance(value, str) else str(value)
    if _CANARY_RE.search(raw) or _SECRET_FIELD_RE.search(raw):
        return True
    return any(secret in raw for secret in secret_values(extra))


class RedactingLogFilter:
    """Drop secret material from log records. This does not control other processes."""

    def filter(self, record) -> bool:
        record.msg = scrub_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {key: scrub_text(item) if isinstance(item, str) else item for key, item in record.args.items()}
            else:
                record.args = tuple(scrub_text(item) if isinstance(item, str) else item for item in record.args)
        return True


def install_log_redactor(logger) -> None:
    redactor = RedactingLogFilter()
    if not any(isinstance(item, RedactingLogFilter) for item in logger.filters):
        logger.addFilter(redactor)
    for handler in logger.handlers:
        if not any(isinstance(item, RedactingLogFilter) for item in handler.filters):
            handler.addFilter(redactor)


class PythonSocketBlock:
    """Block new socket connections inside this process.

    This is not OS-enforced isolation. Native code, another process, or a
    process started before the patch can still use the network. Deployment
    isolation has to be applied by the OS or the container runtime.
    """

    def __enter__(self):
        import socket

        self._socket = socket
        self._original = socket.socket

        class _BlockedSocket(socket.socket):
            def connect(self, *args, **kwargs):
                raise SafetyError("python_network_block", ISOLATION_BOUNDARY)

            def connect_ex(self, *args, **kwargs):
                raise SafetyError("python_network_block", ISOLATION_BOUNDARY)

        socket.socket = _BlockedSocket
        return self

    def __exit__(self, exc_type, exc, tb):
        self._socket.socket = self._original
        return False


def reject_active_promotion(relative_path: str) -> None:
    """Learning or evolution must not replace active security or model code."""
    normalized = relative_path.replace("\\", "/").lstrip("./")
    if normalized in ACTIVE_POLICY_FILES or Path(normalized).name in {item.split("/")[-1] for item in ACTIVE_POLICY_FILES}:
        raise SafetyError(
            "policy_frozen",
            "Learning cannot mutate security policy or promote candidate code into the active path.",
        )


def redact_mapping(value):
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in ("key", "token", "secret", "password", "credential")):
                redacted[key] = "***"
            else:
                redacted[key] = redact_mapping(item)
        return redacted
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    return value


def comet_experiment_kwargs(comet_config: dict) -> dict | None:
    """Return Comet client kwargs only when the operator opted in for this process."""
    if os.environ.get("KRONOS_ALLOW_COMET") != "1":
        return None
    api_key = os.environ.get("COMET_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_COMET_API_KEY":
        return None
    project = os.environ.get("COMET_PROJECT_NAME") or (comet_config or {}).get("project_name")
    workspace = os.environ.get("COMET_WORKSPACE") or (comet_config or {}).get("workspace")
    if not project or not workspace or "your_comet" in str(workspace):
        return None
    return {"api_key": api_key, "project_name": project, "workspace": workspace}


def _validate_config_file(config_path: Path) -> None:
    if config_path.is_symlink() or not config_path.is_file():
        raise SafetyError("invalid_model", "config.json must be a regular file inside the model directory")
    if config_path.stat().st_size > MAX_CONFIG_BYTES:
        raise SafetyError("invalid_model", "config.json exceeds the engineering size limit")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SafetyError("invalid_model", "config.json is not valid JSON") from None
    if not isinstance(config, dict):
        raise SafetyError("invalid_model", "config.json must be an object")
    if "auto_map" in config or config.get("trust_remote_code") or "pytorch_model.bin" in json.dumps(config):
        raise SafetyError("invalid_model", "remote code in model config is disabled")
    for key, cap in _ARCH_CAPS.items():
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > cap:
            raise SafetyError("invalid_model", "model config exceeds the engineering architecture limit")


def _validate_local_model_dir(directory: Path) -> str:
    if directory.is_symlink():
        raise SafetyError("symlink_rejected", "symlinked model directories are disabled")
    resolved = directory.resolve()
    if not resolved.is_dir():
        raise SafetyError("invalid_model", "model path must be a local directory")
    if (resolved / "pytorch_model.bin").exists() or (resolved / "model.bin").exists():
        raise SafetyError("unsafe_weight_format", "pickle weight files are disabled; provide model.safetensors only")
    weights = resolved / "model.safetensors"
    if weights.is_symlink() or not weights.is_file():
        raise SafetyError("invalid_model", "model.safetensors must be a regular file in the model directory")
    try:
        weights.resolve(strict=True).relative_to(resolved)
    except ValueError:
        raise SafetyError("path_escape", "model weights are outside the model directory") from None
    if weights.stat().st_size <= 0 or weights.stat().st_size > MAX_WEIGHT_BYTES:
        raise SafetyError("invalid_model", "model.safetensors size is outside the engineering limit")
    _validate_config_file(resolved / "config.json")
    return str(resolved)


def prepare_pretrained_load(model_id, kwargs: dict | None = None) -> tuple[str, dict]:
    """Validate a from_pretrained source. Does not download weights."""
    apply_hub_environment()
    if not isinstance(model_id, (str, Path)):
        raise SafetyError("invalid_model", "model source must be a local directory or a reviewed repo id")
    text = str(model_id).strip()
    _reject_url(text)
    options = dict(kwargs or {})
    candidate = Path(os.path.expanduser(text))
    if candidate.is_dir():
        options["local_files_only"] = True
        options.pop("force_download", None)
        return _validate_local_model_dir(candidate), options

    if text in PINNED_REVISIONS:
        if os.environ.get("KRONOS_ALLOW_HUB_DOWNLOAD") != "1":
            raise SafetyError(
                "hub_download_disabled",
                "Hub download is disabled. Use a local model.safetensors directory, "
                "or set KRONOS_ALLOW_HUB_DOWNLOAD=1 before importing Kronos.",
            )
        if os.environ.get("HF_HUB_OFFLINE") == "1" and not options.get("local_files_only"):
            raise SafetyError(
                "hub_offline",
                "Hub download was locked off at startup. Set KRONOS_ALLOW_HUB_DOWNLOAD=1 before importing Kronos.",
            )
        expected = PINNED_REVISIONS[text]
        given = options.get("revision")
        if given is not None and given != expected:
            raise SafetyError("unpinned_revision", "revision does not match the reviewed commit")
        options["revision"] = expected
        options.setdefault("local_files_only", False)
        return text, options

    raise SafetyError(
        "invalid_model",
        "model source must be an existing local directory with model.safetensors, or a reviewed repo id",
    )


def webui_runtime_settings() -> dict:
    host = os.environ.get("KRONOS_WEBUI_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if host == "localhost":
        host = "127.0.0.1"
    raw_port = os.environ.get("KRONOS_WEBUI_PORT", "7070")
    try:
        port = int(raw_port)
    except ValueError:
        raise SafetyError("invalid_bind", "KRONOS_WEBUI_PORT must be an integer") from None
    if port < 1 or port > 65535:
        raise SafetyError("invalid_bind", "KRONOS_WEBUI_PORT is outside 1-65535")
    if host not in {"127.0.0.1", "::1"}:
        token = os.environ.get("KRONOS_WEBUI_TOKEN", "")
        if os.environ.get("KRONOS_ALLOW_NONLOCAL_BIND") != "1" or len(token) < 16:
            raise SafetyError(
                "nonlocal_bind_disabled",
                "refusing a non-local bind without KRONOS_ALLOW_NONLOCAL_BIND=1 and a KRONOS_WEBUI_TOKEN of at least 16 characters",
            )
    # The Werkzeug debugger stays off even if a debug variable is present.
    return {"host": host, "port": port, "debug": False, "use_reloader": False, "load_dotenv": False}


def loopback_addr(addr: str | None) -> bool:
    return addr in {"127.0.0.1", "::1"}


def secrets_equal(left: str, right: str) -> bool:
    if not isinstance(left, str) or not isinstance(right, str) or len(left) != len(right) or not left:
        return False
    return hmac.compare_digest(left, right)

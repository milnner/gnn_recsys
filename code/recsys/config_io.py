# recsys/config_io.py
from __future__ import annotations

import enum
import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

from .config import Config


def _json_default(obj: Any) -> Any:
    """Serializa tipos que `json.dumps` não conhece. Falha explícita para
    tipos inesperados."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    raise TypeError(f"Tipo não serializável em Config: {type(obj).__name__}")


def _coerce(value: Any, hint: Any, field_path: str) -> Any:
    """Coage valor JSON-decoded ao tipo do hint. Cobre tuple/Enum/Literal/Path."""
    if value is None:
        return value
    origin = get_origin(hint)

    if origin is tuple and isinstance(value, list):
        args = get_args(hint)
        if args and args[-1] is Ellipsis:  # tuple[T, ...]
            return tuple(_coerce(v, args[0], field_path) for v in value)
        if args:  # tuple[T1, T2, ...]
            return tuple(_coerce(v, a, field_path) for v, a in zip(value, args))
        return tuple(value)

    if origin is Literal:
        allowed = get_args(hint)
        if value not in allowed:
            raise ValueError(
                f"{field_path}: {value!r} não está em {allowed} "
                f"(esperado um destes valores)"
            )
        return value

    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return hint(value)

    if hint is Path and isinstance(value, str):
        return Path(value)

    return value


def _from_dict(cls: type, data: dict, path: str = "") -> Any:
    """Reconstrói recursivamente um dataclass a partir de dict JSON-decoded."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} não é dataclass")
    hints = get_type_hints(cls)  # resolve strings
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{path or cls.__name__}: campos desconhecidos {sorted(unknown)}. "
            f"Esperado: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        f_type = hints[f.name]
        value = data[f.name]
        field_path = f"{path}.{f.name}" if path else f.name
        if is_dataclass(f_type) and isinstance(value, dict):
            value = _from_dict(f_type, value, field_path)
        else:
            value = _coerce(value, f_type, field_path)
        kwargs[f.name] = value

    try:
        return cls(**kwargs)  # __post_init__ valida aqui
    except TypeError as e:
        raise TypeError(
            f"{path or cls.__name__}: falha ao construir. {e}. "
            f"Recebido: {sorted(kwargs)}. "
            f"Esperado: {sorted(f.name for f in fields(cls))}"
        ) from e


def save_config(cfg: Config, path: str | Path) -> Path:
    """Serializa Config para JSON de forma atômica."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        asdict(cfg),
        indent=2,
        allow_nan=False,  # rejeita inf/nan
        default=_json_default,  # Path/Enum
        sort_keys=False,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)  # atômico
    return path


def load_config(path: str | Path) -> Config:
    """Carrega Config de JSON com validação de tipo, Literal, tuple e estritos."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: esperado objeto JSON, recebido {type(data).__name__}"
        )
    return _from_dict(Config, data, path=path.stem)

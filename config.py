"""Load shared JSON settings before applying command-line overrides."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
PROVIDERS = ("lmstudio", "ollama", "openai", "claude")
PATH_SETTINGS = {
    "output_root", "model", "speaker_model", "pdf_font_file", "summary_prompt_file",
    "prompt_file", "output",
}


def load_config(path, section):
    path = path.expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as error:
        raise ValueError(f"Файл конфигурации не найден: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Некорректный JSON в {path}: строка {error.lineno}, столбец {error.colno}") from error
    if not isinstance(data, dict):
        raise ValueError("Корень конфигурации должен быть JSON-объектом.")
    allowed = {"provider", "base_url", "api_key", "context_length", "transcribe", "summarize"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Неизвестные параметры конфигурации: {', '.join(sorted(unknown))}")
    values = data.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"Раздел {section} должен быть JSON-объектом.")
    settings = {key: data[key] for key in ("provider", "base_url", "api_key", "context_length") if key in data}
    settings.update(values)
    if "base_url" in settings:
        settings["lm_url"] = settings.pop("base_url")
    for key in PATH_SETTINGS:
        if key == "model" and section == "summarize":
            continue
        value = settings.get(key)
        if isinstance(value, str) and value:
            candidate = Path(value).expanduser()
            settings[key] = candidate if candidate.is_absolute() else path.parent / candidate
    return settings


def apply_config(parser, settings):
    actions = {action.dest: action for action in parser._actions}
    unknown = set(settings) - set(actions)
    if unknown:
        raise ValueError(f"Неизвестные параметры раздела конфигурации: {', '.join(sorted(unknown))}")
    converted = {}
    for key, value in settings.items():
        action = actions[key]
        if action.choices is not None and value is not None and value not in action.choices:
            raise ValueError(f"Недопустимое значение {key}: {value!r}. Доступны: {', '.join(action.choices)}")
        if action.type is not None and value is not None and not isinstance(value, action.type):
            try:
                value = action.type(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Некорректное значение параметра {key}: {value!r}") from error
        if isinstance(action, argparse.BooleanOptionalAction) and not isinstance(value, bool):
            raise ValueError(f"Параметр {key} должен быть true или false.")
        converted[key] = value
    parser.set_defaults(**converted)


def parse_args(parser_factory, section, argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_args, _ = preliminary.parse_known_args(argv)
    config_path = config_args.config.expanduser().resolve()
    settings = load_config(config_path, section)
    parser = parser_factory()
    apply_config(parser, settings)
    args = parser.parse_args(argv)
    args.config = config_path
    return parser, args

"""Load grouped JSON settings before applying command-line overrides."""
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
BOTH = ("transcribe", "summarize")
SCHEMA = {
    "models": {
        "provider": dict.fromkeys(BOTH, "provider"),
        "base_url": dict.fromkeys(BOTH, "lm_url"),
        "api_key": dict.fromkeys(BOTH, "api_key"),
        "context_length": dict.fromkeys(BOTH, "context_length"),
        "whisper": {"transcribe": "model"},
        "speaker": {"transcribe": "speaker_model"},
        "translation": {"transcribe": "translation_model"},
        "summary": {"transcribe": "summary_model", "summarize": "model"},
        "vision": {"transcribe": "vision_model"},
    },
    "transcription": {
        "language": {"transcribe": "language"},
        "prompt": {"transcribe": "prompt"},
        "device": {"transcribe": "device"},
        "speakers": {"transcribe": "speakers"},
        "num_speakers": {"transcribe": "num_speakers"},
        "reuse_transcript": {"transcribe": "reuse_transcript"},
    },
    "translation": {
        "chunk_chars": {"transcribe": "translation_chunk_chars"},
        "max_tokens": {"transcribe": "translation_max_tokens"},
        "timeout": {"transcribe": "translation_timeout"},
        "reasoning": {"transcribe": "translation_reasoning"},
    },
    "text": {
        "output_language": dict.fromkeys(BOTH, "output_language"),
        "timestamps": {"transcribe": "timestamps"},
        "paragraph_seconds": {"transcribe": "paragraph_seconds"},
        "timestamp_interval": {"transcribe": "timestamp_interval"},
    },
    "slides": {
        "enabled": {"transcribe": "slides"},
        "mode": {"transcribe": "slides_mode"},
        "vision_max_tokens": {"transcribe": "vision_max_tokens"},
        "sample_interval": {"transcribe": "sample_interval"},
        "scene_threshold": {"transcribe": "scene_threshold"},
        "settle": {"transcribe": "settle"},
        "min_interval": {"transcribe": "min_slide_interval"},
        "require_speech": {"transcribe": "require_speech"},
        "speech_window": {"transcribe": "speech_window"},
    },
    "outputs": {
        "output_root": {"transcribe": "output_root"},
        "pptx": {"transcribe": "pptx"},
        "pdf": {"transcribe": "pdf"},
        "pdf_font_file": {"transcribe": "pdf_font_file"},
        "obsidian": {"transcribe": "obsidian"},
        "summary_file": {"summarize": "output"},
    },
    "summary": {
        "enabled": {"transcribe": "summarize"},
        "list_models": {"summarize": "list_models"},
        "prompt_file": {"transcribe": "summary_prompt_file", "summarize": "prompt_file"},
        "type": dict.fromkeys(BOTH, "summary_type"),
        "chunk_chars": {"transcribe": "summary_chunk_chars", "summarize": "chunk_chars"},
        "max_tokens": {"transcribe": "summary_max_tokens", "summarize": "max_tokens"},
        "timeout": {"transcribe": "summary_timeout", "summarize": "timeout"},
        "reasoning": {"transcribe": "summary_reasoning", "summarize": "reasoning"},
        "force": {"summarize": "force"},
    },
}


def flatten_config(data, command):
    if not isinstance(data, dict):
        raise ValueError("Корень конфигурации должен быть JSON-объектом.")
    unknown_sections = set(data) - set(SCHEMA)
    if unknown_sections:
        raise ValueError(f"Неизвестные разделы конфигурации: {', '.join(sorted(unknown_sections))}")
    settings = {}
    for section, fields in SCHEMA.items():
        values = data.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Раздел {section} должен быть JSON-объектом.")
        unknown = set(values) - set(fields)
        if unknown:
            raise ValueError(f"Неизвестные параметры раздела {section}: {', '.join(sorted(unknown))}")
        for key, value in values.items():
            target = fields[key].get(command)
            if target:
                settings[target] = value
    return settings


def load_config(path, command):
    path = path.expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as error:
        raise ValueError(f"Файл конфигурации не найден: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Некорректный JSON в {path}: строка {error.lineno}, столбец {error.colno}") from error
    settings = flatten_config(data, command)
    for key in PATH_SETTINGS:
        if key == "model" and command == "summarize":
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


def parse_args(parser_factory, command, argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_args, _ = preliminary.parse_known_args(argv)
    config_path = config_args.config.expanduser().resolve()
    settings = load_config(config_path, command)
    parser = parser_factory()
    apply_config(parser, settings)
    args = parser.parse_args(argv)
    args.config = config_path
    return parser, args

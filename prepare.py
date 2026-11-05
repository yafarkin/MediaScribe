#!/usr/bin/env python3
"""Explicit online step. Inference itself is always offline."""
import argparse
import os
import sys
from transcribe import ROOT, MLX_MODEL, CUDA_MODEL, SPEAKER_MODEL, backend_name


def main():
    p = argparse.ArgumentParser(description="Скачать модели для последующей работы без сети")
    p.add_argument("--speakers", action="store_true")
    p.add_argument("--model", help="Другой Hugging Face репозиторий модели Whisper")
    args = p.parse_args()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        if error.name != "huggingface_hub":
            raise
        print("Не установлены зависимости. Выполните:\n"
              f'  "{sys.executable}" -m ensurepip --upgrade\n'
              f'  "{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"',
              file=sys.stderr)
        return 1
    backend = backend_name()
    repo = args.model or (MLX_MODEL if backend == "mlx" else CUDA_MODEL)
    folder = ROOT / "models" / ("whisper-mlx" if backend == "mlx" else "whisper-faster")
    print(f"Скачивание {repo} в {folder}", flush=True)
    snapshot_download(repo_id=repo, local_dir=str(folder))
    if args.speakers:
        print("Для модели голосов нужны принятые условия на Hugging Face и HF_TOKEN.", flush=True)
        snapshot_download(repo_id=SPEAKER_MODEL, local_dir=str(ROOT / "models" / "speakers"),
                          token=os.environ.get("HF_TOKEN"))
    print("Модели скачаны. transcribe.py можно запускать без сети.")


if __name__ == "__main__":
    sys.exit(main())

"""Small, loopback-only client for LM Studio's OpenAI-compatible API."""
import ipaddress
import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("Перенаправление запроса LM Studio запрещено.")


def local_endpoint(url):
    parts = urlsplit(url)
    try:
        local = ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        local = False
    if parts.scheme != "http" or not local or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("LM Studio URL должен быть локальным, например http://127.0.0.1:1234/v1")
    return url.rstrip("/")


def request_json(base_url, path, payload=None, timeout=600):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LM_STUDIO_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["LM_STUDIO_API_KEY"]
    request = urllib.request.Request(local_endpoint(base_url) + path,
                                     data=json.dumps(payload).encode() if payload is not None else None,
                                     headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"LM Studio HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"LM Studio недоступна: {error}. Проверьте локальный сервер и --lm-url.") from error


def chat(base_url, model, messages, max_tokens=4096, timeout=600, reasoning="off"):
    # Native API actually applies reasoning=off to MLX Qwen. The compatibility
    # endpoint may silently ignore chat_template_kwargs on some LM Studio builds.
    payload = {
        "model": model, "temperature": 0.1, "max_output_tokens": max_tokens,
        "stream": False, "store": False,
        "system_prompt": "\n\n".join(m["content"] for m in messages if m["role"] == "system"),
        "input": "\n\n".join(m["content"] for m in messages if m["role"] == "user"),
    }
    if reasoning != "auto":
        payload["reasoning"] = reasoning
    result = request_json(local_endpoint(base_url).removesuffix("/v1"), "/api/v1/chat", payload, timeout=timeout)
    try:
        text = "\n\n".join(item["content"] for item in result["output"] if item["type"] == "message")
    except (KeyError, TypeError) as error:
        raise RuntimeError("LM Studio вернула ответ без текста сообщения.") from error
    if result.get("stats", {}).get("total_output_tokens", 0) >= max_tokens:
        raise RuntimeError("Ответ модели обрезан лимитом токенов. Увеличьте --max-tokens или уменьшите --chunk-chars.")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("LM Studio вернула пустой текст. Проверьте модель и лимит токенов.")
    return text.strip()

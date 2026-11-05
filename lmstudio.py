"""Small loopback-only client for LM Studio, including streaming progress."""
from contextlib import contextmanager
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


@contextmanager
def open_response(base_url, path, payload=None, timeout=600):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LM_STUDIO_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["LM_STUDIO_API_KEY"]
    request = urllib.request.Request(local_endpoint(base_url) + path,
                                     data=json.dumps(payload).encode() if payload is not None else None,
                                     headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            yield response
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"LM Studio HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"LM Studio недоступна: {error}. Проверьте локальный сервер и --lm-url.") from error


def request_json(base_url, path, payload=None, timeout=600):
    with open_response(base_url, path, payload, timeout) as response:
        return json.load(response)


def iter_events(response):
    """Read SSE blocks, including comments and multiline data fields."""
    event, data = "", []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                value = json.loads("\n".join(data))
                value.setdefault("type", event)
                yield value
            event, data = "", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        value = json.loads("\n".join(data))
        value.setdefault("type", event)
        yield value


def request_stream(base_url, path, payload, timeout, on_event):
    result = None
    with open_response(base_url, path, payload, timeout) as response:
        for event in iter_events(response):
            if event["type"] == "error":
                raise RuntimeError(f"LM Studio: {event.get('error', {}).get('message', 'ошибка потока')}")
            on_event(event)
            if event["type"] == "chat.end":
                result = event["result"]
                break
    if result is None:
        raise RuntimeError("Поток LM Studio прерван до завершения ответа. Незавершённый конспект не сохранён.")
    return result


def chat(base_url, model, messages, max_tokens=4096, timeout=600, reasoning="off", on_event=None):
    # Native API actually applies reasoning=off to MLX Qwen. The compatibility
    # endpoint may silently ignore chat_template_kwargs on some LM Studio builds.
    payload = {
        "model": model, "temperature": 0.1, "max_output_tokens": max_tokens,
        "stream": on_event is not None, "store": False,
        "system_prompt": "\n\n".join(m["content"] for m in messages if m["role"] == "system"),
        "input": "\n\n".join(m["content"] for m in messages if m["role"] == "user"),
    }
    if reasoning != "auto":
        payload["reasoning"] = reasoning
    base = local_endpoint(base_url).removesuffix("/v1")
    if on_event is None:
        result = request_json(base, "/api/v1/chat", payload, timeout=timeout)
    else:
        result = request_stream(base, "/api/v1/chat", payload, timeout, on_event)
    try:
        text = "\n\n".join(item["content"] for item in result["output"] if item["type"] == "message")
    except (KeyError, TypeError) as error:
        raise RuntimeError("LM Studio вернула ответ без текста сообщения.") from error
    if result.get("stats", {}).get("total_output_tokens", 0) >= max_tokens:
        raise RuntimeError("Ответ модели обрезан лимитом токенов. Увеличьте --max-tokens или уменьшите --chunk-chars.")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("LM Studio вернула пустой текст. Проверьте модель и лимит токенов.")
    return text.strip()

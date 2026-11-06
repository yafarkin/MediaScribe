"""Small HTTP clients for LM Studio, Ollama, OpenAI and Claude."""
from contextlib import contextmanager
import ipaddress
import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit

PROVIDER_NAMES = {"lmstudio": "LM Studio", "ollama": "Ollama", "openai": "OpenAI", "claude": "Claude"}
API_KEY_ENV = {
    "lmstudio": "LM_STUDIO_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}


class StreamInterruptedError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("Перенаправление запроса к модели запрещено.")


def local_endpoint(url):
    parts = urlsplit(url)
    try:
        local = ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        local = False
    if parts.scheme != "http" or not local or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("URL локального сервера должен использовать HTTP и числовой loopback-адрес, например http://127.0.0.1:1234/v1")
    return url.rstrip("/")


def provider_endpoint(url, provider="lmstudio"):
    if provider in {"lmstudio", "ollama"}:
        return local_endpoint(url)
    if provider not in {"openai", "claude"}:
        raise ValueError(f"Неизвестный провайдер: {provider}")
    parts = urlsplit(url)
    try:
        local = ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        local = False
    if (parts.scheme != "https" and not (parts.scheme == "http" and local)) or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("URL внешнего провайдера должен использовать HTTPS без учётных данных, параметров и якоря.")
    return url.rstrip("/")


def api_key(provider, configured=None):
    return os.environ.get(API_KEY_ENV[provider]) or configured


@contextmanager
def open_response(base_url, path, payload=None, timeout=600, provider="lmstudio", api_key_value=None):
    headers = {"Content-Type": "application/json"}
    key = api_key(provider, api_key_value)
    if provider == "claude":
        if not key:
            raise RuntimeError("Для Claude задайте ANTHROPIC_API_KEY или параметр api_key в config.json.")
        headers["x-api-key"] = key
        headers["anthropic-version"] = "2023-06-01"
    elif key:
        headers["Authorization"] = "Bearer " + key
    if provider == "openai" and not key:
        raise RuntimeError("Для OpenAI задайте OPENAI_API_KEY или параметр api_key в config.json.")
    request = urllib.request.Request(provider_endpoint(base_url, provider) + path,
                                     data=json.dumps(payload).encode() if payload is not None else None,
                                     headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    name = PROVIDER_NAMES[provider]
    try:
        with opener.open(request, timeout=timeout) as response:
            yield response
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"{name} HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"{name} недоступен: {error}. Проверьте сервер, base_url и api_key.") from error


def request_json(base_url, path, payload=None, timeout=600, provider="lmstudio", api_key_value=None):
    with open_response(base_url, path, payload, timeout=timeout, provider=provider, api_key_value=api_key_value) as response:
        return json.load(response)


def model_info(base_url, model):
    base = local_endpoint(base_url).removesuffix("/v1")
    result = request_json(base, "/api/v1/models")
    models = result.get("models", result.get("data", []))
    for item in models:
        if model in {item.get("key"), item.get("id")}:
            return item
    raise RuntimeError(f"Модель {model} не найдена в LM Studio.")


def validate_context_length(info, model, context_length):
    maximum = info.get("max_context_length")
    if isinstance(maximum, int) and context_length > maximum:
        raise ValueError(f"Контекст {context_length} превышает максимум {maximum} для модели {model}.")


def model_context_length(base_url, model):
    info = model_info(base_url, model)
    contexts = [instance.get("config", {}).get("context_length")
                for instance in info.get("loaded_instances", [])]
    contexts = [value for value in contexts if isinstance(value, int) and value > 0]
    return min(contexts) if contexts else None


def load_model(base_url, model, context_length, timeout=600, provider="lmstudio", api_key_value=None):
    if provider != "lmstudio":
        return model
    info = model_info(base_url, model)
    validate_context_length(info, model, context_length)
    instances = info.get("loaded_instances", [])
    matching = [instance for instance in instances
                if instance.get("config", {}).get("context_length") == context_length]
    if matching:
        return matching[0].get("id", model)
    base = local_endpoint(base_url).removesuffix("/v1")
    result = request_json(base, "/api/v1/models/load", {
        "model": model, "context_length": context_length, "echo_load_config": True,
    }, timeout=timeout)
    actual = result.get("load_config", {}).get("context_length")
    instance_id = result.get("instance_id")
    if actual is None or instance_id is None:
        info = model_info(base_url, model)
        matching = [instance for instance in info.get("loaded_instances", [])
                    if instance.get("config", {}).get("context_length") == context_length]
        if matching:
            actual = context_length
            instance_id = matching[0].get("id")
    if actual != context_length or not instance_id:
        raise RuntimeError(f"LM Studio загрузила модель {model} с контекстом {actual}, ожидался {context_length}.")
    return instance_id


def list_models(base_url, provider="lmstudio", api_key_value=None):
    result = request_json(base_url, "/v1/models" if provider == "claude" else "/models",
                          provider=provider, api_key_value=api_key_value)
    return result.get("data", result.get("models", []))


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
        raise StreamInterruptedError("Поток LM Studio прерван до завершения ответа. Незавершённый ответ не сохранён.")
    return result


def openai_text(result, provider):
    try:
        choice = result["choices"][0]
        content = choice["message"]["content"]
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if item.get("type") == "text")
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"{PROVIDER_NAMES[provider]} вернул ответ без текста сообщения.") from error
    if choice.get("finish_reason") == "length":
        raise RuntimeError("Ответ модели обрезан лимитом токенов. Увеличьте max_tokens или уменьшите размер части.")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{PROVIDER_NAMES[provider]} вернул пустой текст. Проверьте модель и лимит токенов.")
    return content.strip()


def claude_text(result):
    try:
        text = "\n\n".join(item["text"] for item in result["content"] if item["type"] == "text")
    except (KeyError, TypeError) as error:
        raise RuntimeError("Claude вернул ответ без текстового содержимого.") from error
    if result.get("stop_reason") == "max_tokens":
        raise RuntimeError("Ответ модели обрезан лимитом токенов. Увеличьте max_tokens или уменьшите размер части.")
    if not text.strip():
        raise RuntimeError("Claude вернул пустой текст. Проверьте модель и лимит токенов.")
    return text.strip()


def compatible_chat(base_url, model, messages, max_tokens, timeout, provider, api_key_value,
                    context_length=None):
    system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    if provider == "claude":
        payload = {"model": model, "temperature": 0.1, "max_tokens": max_tokens,
                   "system": system,
                   "messages": [message for message in messages if message["role"] != "system"]}
        result = request_json(base_url, "/v1/messages", payload, timeout=timeout, provider=provider, api_key_value=api_key_value)
        return claude_text(result)
    payload = {"model": model, "temperature": 0.1, "max_tokens": max_tokens, "messages": messages}
    if provider == "ollama" and context_length:
        payload["options"] = {"num_ctx": context_length}
    result = request_json(base_url, "/chat/completions", payload, timeout=timeout, provider=provider, api_key_value=api_key_value)
    return openai_text(result, provider)


def chat(base_url, model, messages, max_tokens=4096, timeout=600, reasoning="off", on_event=None,
         provider="lmstudio", api_key_value=None, context_length=None):
    if provider != "lmstudio":
        if on_event:
            on_event({"type": "prompt_processing.start"})
        text = compatible_chat(base_url, model, messages, max_tokens, timeout, provider, api_key_value,
                               context_length)
        if on_event:
            on_event({"type": "message.start"})
            on_event({"type": "message.delta", "content": text})
        return text
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
        raise RuntimeError("Ответ модели обрезан лимитом токенов. Увеличьте max_tokens для этого этапа или уменьшите размер части.")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("LM Studio вернула пустой текст. Проверьте модель и лимит токенов.")
    return text.strip()


def vision_chat(base_url, model, content, timeout=180, provider="lmstudio", api_key_value=None,
                max_tokens=250):
    if provider == "claude":
        converted = []
        for part in content:
            if part["type"] == "text":
                converted.append(part)
            else:
                header, data = part["image_url"]["url"].split(",", 1)
                converted.append({"type": "image", "source": {
                    "type": "base64", "media_type": header.split(":", 1)[1].split(";", 1)[0], "data": data,
                }})
        payload = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": converted}]}
        return claude_text(request_json(base_url, "/v1/messages", payload, timeout=timeout, provider=provider, api_key_value=api_key_value))
    payload = {"model": model, "temperature": 0, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": content}]}
    if provider == "lmstudio":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    result = request_json(base_url, "/chat/completions", payload,
                          timeout=timeout, provider=provider, api_key_value=api_key_value)
    return openai_text(result, provider)

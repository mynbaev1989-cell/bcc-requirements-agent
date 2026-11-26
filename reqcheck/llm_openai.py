"""Клиент для локальной модели внутри контура (OpenAI-совместимый API: Ollama, vLLM, LM Studio).

Документ уходит только на адрес REQCHECK_LLM_BASE_URL, в Google ничего не отправляется.
Только стандартная библиотека (urllib). Ключ необязателен: REQCHECK_LLM_API_KEY, в отчёт не попадает.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from reqcheck.llm import (DEFAULT_MAX_CALLS, DEFAULT_RETRIES, DEFAULT_TIMEOUT_SEC, MAX_BAD_JSON, LLMError,
                          LLMStats, LLMUnavailable, parse_json_loose)

ENV_BASE_URL = "REQCHECK_LLM_BASE_URL"
ENV_MODEL = "REQCHECK_LLM_MODEL"
ENV_API_KEY = "REQCHECK_LLM_API_KEY"


def safe_base_url(url: str) -> str:
    """Адрес для паспорта: схема + хост[:порт], без пути, логина/пароля и параметров."""
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname or ""
        if ":" in host:  # IPv6
            host = f"[{host}]"
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "(адрес скрыт)"
    if not host:
        return "(адрес скрыт)"
    return f"{parts.scheme}://{host}{port}" if parts.scheme else f"{host}{port}"


class OpenAICompatClient:
    """Тот же интерфейс, что у GeminiClient: generate_json, stats, model. Температура 0, JSON-режим."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        retries: int = DEFAULT_RETRIES,
        max_calls: int = DEFAULT_MAX_CALLS,
        cache_dir: str | None = ".reqcheck_cache",
    ):
        self.base_url = (base_url or os.environ.get(ENV_BASE_URL) or "").strip().rstrip("/")
        if not self.base_url:
            raise LLMUnavailable(f"Не задан {ENV_BASE_URL} (например http://localhost:11434/v1).")
        self.model = (model or os.environ.get(ENV_MODEL) or "").strip()
        if not self.model:
            raise LLMUnavailable(f"Не задана модель: укажите --model или {ENV_MODEL} (например llama3.1:8b).")
        self._api_key = os.environ.get(ENV_API_KEY) or ""
        self.provider = f"openai-compatible: {safe_base_url(self.base_url)}"
        self.timeout, self.retries, self.max_calls = timeout, retries, max_calls
        self.stats = LLMStats(model_version=self.model)
        self.stats.models_used.append(self.model)
        self._json_mode = True  # сервер не принял response_format — дальше шлём без него
        self._exhausted = ""  # сервер не принимает соединения: дальше в сеть не ходим, шаги берут запасные варианты
        self._parallel = threading.Semaphore(max(1, int(os.environ.get("REQCHECK_MAX_PARALLEL", "3") or 3)))
        self.cache_dir = Path(cache_dir) if cache_dir else None

    # --- кэш: ключ включает провайдера (base_url) и модель, ответы Gemini не смешиваются ---
    def _cache_path(self, payload: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = f"openai-compat\n{self.base_url}\n{self.model}\n{payload}"
        return self.cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    def _from_cache(self, role: str, payload: str) -> tuple[bool, Any]:
        cache = self._cache_path(payload)
        if not (cache and cache.exists()):
            return False, None
        try:
            parsed = json.loads(cache.read_text(encoding="utf-8"))["parsed"]
        except (OSError, ValueError, KeyError):
            return False, None
        self.stats.add(role, cache_hits=1)
        return True, parsed

    # --- запрос ---
    def _post(self, system: str, prompt: str) -> tuple[str, dict]:
        body: dict[str, Any] = {"model": self.model, "temperature": 0,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": prompt}]}
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with self._parallel, urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"неожиданная структура ответа ({exc})") from exc
        return text, data.get("usage") or {}

    def generate_json(self, role: str, system: str, prompt: str) -> Any:
        payload = system + "\n\n" + prompt
        hit, cached = self._from_cache(role, payload)
        if hit:
            return cached
        if self._exhausted:
            raise LLMError(self._exhausted)
        if self.stats.calls >= self.max_calls:
            raise LLMError(f"Исчерпан лимит вызовов LLM на прогон ({self.max_calls}).")
        self.stats.add(role, calls=1)
        last_error, bad_json, content = "", 0, prompt
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                text, usage = self._post(system, content)
                self.stats.add(role, seconds=time.perf_counter() - started,
                               input_tokens=int(usage.get("prompt_tokens") or 0),
                               output_tokens=int(usage.get("completion_tokens") or 0))
                parsed = parse_json_loose(text)
                cache = self._cache_path(payload)
                if cache:
                    try:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps({"role": role, "model": self.model, "parsed": parsed},
                                                    ensure_ascii=False), encoding="utf-8")
                    except OSError:
                        pass
                return parsed
            except (ValueError, json.JSONDecodeError) as exc:
                self.stats.add(seconds=time.perf_counter() - started)
                last_error = f"невалидный JSON ({exc})"
                bad_json += 1
                if bad_json >= MAX_BAD_JSON:
                    break
                content = prompt + "\n\nВАЖНО: верни ТОЛЬКО валидный JSON по схеме, без пояснений и без обрезки."
            except urllib.error.HTTPError as exc:
                self.stats.add(seconds=time.perf_counter() - started)
                last_error = f"HTTP {exc.code}"
                if exc.code == 400 and self._json_mode:
                    self._json_mode = False  # сервер не знает response_format — один раз повторяем без него
                    continue
                if exc.code in (400, 401, 403, 404):
                    self.stats.add(role, failures=1)
                    hint = " (проверьте имя модели и REQCHECK_LLM_BASE_URL)" if exc.code == 404 else ""
                    raise LLMError(f"Сервер модели отклонил запрос: {last_error}{hint}") from exc
                if attempt < self.retries:
                    self.stats.add(retries=1)
                    time.sleep(self._backoff(exc, attempt))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self.stats.add(seconds=time.perf_counter() - started)
                reason = getattr(exc, "reason", exc)
                last_error = f"{type(exc).__name__}: {reason}"[:200]
                if isinstance(reason, (ConnectionRefusedError, socket.gaierror)):
                    # Сервер не запущен или адрес не существует: повторы только тянут время (в Windows каждая
                    # попытка к закрытому порту — около 4 с). Помечаем сервер недоступным для всего прогона.
                    self._exhausted = (f"Сервер модели {self.provider.split(': ', 1)[-1]} не принимает соединения "
                                       f"({type(reason).__name__}): запустите его или проверьте {ENV_BASE_URL}.")
                    self.stats.add(role, failures=1)
                    self.stats.note(error=f"{role}: {self._exhausted}")
                    raise LLMError(self._exhausted) from exc
                if attempt < self.retries:
                    self.stats.add(retries=1)
                    time.sleep(self._backoff(exc, attempt))
        self.stats.add(role, failures=1)
        self.stats.note(error=f"{role}: {last_error}")
        raise LLMError(f"Модель не ответила корректно после {self.retries + 1} попыток: {last_error}")

    @staticmethod
    def _backoff(exc: Exception, attempt: int) -> float:
        headers = getattr(exc, "headers", None)
        ra = headers.get("Retry-After") if headers else None
        if ra and re.fullmatch(r"\d+(\.\d+)?", str(ra)):
            return min(float(ra), 60.0)
        return min(2 ** attempt * 2 + random.uniform(0, 1.0), 60.0)

"""Клиент LLM (Gemini API) с таймаутами, ретраями, лимитом вызовов, кэшем и разбором JSON.

Ключ берётся только из переменной окружения GEMINI_API_KEY (или файла .env рядом с запуском).
Ключ никогда не логируется и не попадает в отчёт.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MODEL = "gemini-3.5-flash"
# Если модель по умолчанию недоступна для ключа (404), пробуем следующие — без падения прогона
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-latest"]
DEFAULT_TIMEOUT_SEC = 240
DEFAULT_RETRIES = 4
DEFAULT_MAX_CALLS = 40
MAX_BAD_JSON = 2  # попыток при невалидном JSON (сетевые ошибки и 429 повторяются по DEFAULT_RETRIES)
DEFAULT_MAX_OUTPUT_TOKENS = 65536  # у моделей с «размышлениями» лимит включает и размышления
# Глубина размышлений по ролям: смысловой анализ — MEDIUM, механические шаги — LOW (быстрее и без таймаутов)
THINKING_BY_ROLE = {"extractor": "LOW", "reviewer": "MEDIUM", "consistency": "MEDIUM", "gaps": "MEDIUM",
                    "critic": "LOW", "analyst_checklist": "LOW", "analyst_tests": "LOW"}


class LLMError(Exception):
    """Модель недоступна или ответила некорректно после всех повторов."""


class LLMUnavailable(LLMError):
    """Нет ключа / нет библиотеки — работаем в режиме «только правила»."""


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    model_version: str = ""
    errors: list[str] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    thinking_tokens: int = 0
    by_role: dict[str, dict[str, float]] = field(default_factory=dict)  # журнал по агентам: вызовы, кэш, токены, время
    # Агенты работают в нескольких потоках на одном клиенте: счётчики меняем только под блокировкой
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def role(self, name: str) -> dict[str, float]:
        return self.by_role.setdefault(name, {"calls": 0, "cache_hits": 0, "failures": 0, "input_tokens": 0,
                                              "output_tokens": 0, "thinking_tokens": 0, "seconds": 0.0})

    def add(self, role: str | None = None, **counts: float) -> None:
        """Потокобезопасно прибавить счётчики к общим итогам и (если задана роль) к строке роли."""
        with self._lock:
            rs = self.role(role) if role else None
            for key, val in counts.items():
                setattr(self, key, getattr(self, key) + val)
                if rs is not None and key in rs:
                    rs[key] += val

    def note(self, error: str = "", model_version: str = "") -> None:
        with self._lock:
            if error:
                self.errors.append(error)
            if model_version:
                self.model_version = model_version
                if model_version not in self.models_used:
                    self.models_used.append(model_version)


class JsonLLM(Protocol):
    model: str
    stats: LLMStats

    def generate_json(self, role: str, system: str, prompt: str) -> Any: ...


def parse_json_loose(text: str) -> Any:
    """Разобрать JSON из ответа модели: снимает ```-обёртки, обрезает мусор до/после, чинит висячие запятые."""
    if text is None:
        raise ValueError("пустой ответ")
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("в ответе нет JSON")
    start = min(starts)
    end = max(t.rfind("}"), t.rfind("]"))
    candidate = t[start:end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return json.loads(candidate)


def _load_dotenv() -> None:
    if os.environ.get("REQCHECK_NO_DOTENV"):  # тесты и CI: не подхватывать локальный .env
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


class GeminiClient:
    """Обёртка над google-genai. Температура 0 и JSON-режим для воспроизводимости."""

    def __init__(
        self,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        retries: int = DEFAULT_RETRIES,
        max_calls: int = DEFAULT_MAX_CALLS,
        cache_dir: str | None = ".reqcheck_cache",
        min_interval: float = 0.0,
    ):
        _load_dotenv()
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMUnavailable("Не задан GEMINI_API_KEY (см. README, раздел «Ключ LLM»).")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMUnavailable(f"Не установлена библиотека google-genai: {exc}") from exc
        self._types = types
        explicit = model or os.environ.get("GEMINI_MODEL")
        self.model = explicit or DEFAULT_MODEL
        # Явно выбранную модель не подменяем; для модели по умолчанию разрешён запасной список
        self._fallbacks = [] if explicit else [m for m in FALLBACK_MODELS if m != self.model]
        self._chain = [self.model] + self._fallbacks  # порядок поиска в кэше: основная модель, затем запасные
        self.timeout = timeout
        self.retries = retries
        self.max_calls = max_calls
        self.min_interval = float(os.environ.get("REQCHECK_MIN_INTERVAL", min_interval) or 0)
        self.stats = LLMStats()
        self._thinking_ok = True
        self._switch_lock = threading.Lock()  # Reviewer работает в нескольких потоках на одном клиенте
        # Не больше N запросов одновременно: агенты идут параллельно, а бесплатный тариф ограничивает запросы в минуту
        self._parallel = threading.Semaphore(max(1, int(os.environ.get("REQCHECK_MAX_PARALLEL", "3") or 3)))
        self._exhausted = ""  # квота исчерпана и запасных моделей нет: дальнейшие запросы не отправляем
        self._last_call = 0.0
        self.cache_dir = Path(cache_dir) if cache_dir else None
        # Повторы делаем сами (с учётом retryDelay), встроенные повторы SDK отключаем, чтобы паузы не умножались
        self._client = genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=int(timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1)))

    # --- кэш ответов: повторный прогон того же документа не тратит квоту и даёт тот же результат ---
    def _cache_path(self, payload: str, model: str | None = None) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(f"{model or self.model}\n{payload}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _from_cache(self, role: str, payload: str) -> tuple[bool, Any]:
        """Ответ из кэша. Ключ — модель + промпт; запись с меткой другой модели не принимается.

        Модели проверяются по цепочке (основная, затем запасные), а не только текущая: тогда повторный прогон
        берёт те же ответы независимо от того, в какой момент параллельные шаги переключили модель."""
        for model in self._chain:
            cache = self._cache_path(payload, model)
            if not (cache and cache.exists()):
                continue
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                if data.get("model", model) != model:
                    continue  # старые записи кэша могли лечь под ключом одной модели с ответом другой
                parsed = data["parsed"]
            except (OSError, ValueError, KeyError):
                continue
            self.stats.add(role, cache_hits=1)
            # Ответ из кэша тоже дала модель: она попадает в паспорт отчёта (иначе смесь моделей не видна)
            self.stats.note(model_version=model)
            return True, parsed
        return False, None

    def generate_json(self, role: str, system: str, prompt: str) -> Any:
        payload = system + "\n\n" + prompt
        hit, cached = self._from_cache(role, payload)
        if hit:
            return cached
        if self._exhausted:
            raise LLMError(self._exhausted)
        if self.stats.calls >= self.max_calls:
            raise LLMError(f"Исчерпан лимит вызовов LLM на прогон ({self.max_calls}).")

        types = self._types

        def make_config():
            kw = dict(system_instruction=system, temperature=0.0, response_mime_type="application/json",
                      max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS)
            level = os.environ.get("REQCHECK_THINKING") or THINKING_BY_ROLE.get(role, "LOW")
            if self._thinking_ok and level.upper() != "OFF":
                kw["thinking_config"] = types.ThinkingConfig(thinking_level=level.upper())
            return types.GenerateContentConfig(**kw)

        config = make_config()
        last_error = ""
        bad_json = 0
        contents = prompt
        self.stats.add(role, calls=1)  # один запрос — один вызов; повторы считаются отдельно (retries)
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            started = time.perf_counter()
            self._last_call = time.monotonic()
            try:
                model = self.model
                with self._parallel:
                    resp = self._client.models.generate_content(model=model, contents=contents, config=config)
                spent = time.perf_counter() - started
                self.stats.add(role, seconds=spent)
                usage = getattr(resp, "usage_metadata", None)
                if usage is not None:
                    tin = getattr(usage, "prompt_token_count", 0) or 0
                    tout = getattr(usage, "candidates_token_count", 0) or 0
                    tthink = getattr(usage, "thoughts_token_count", 0) or 0
                    self.stats.add(role, input_tokens=tin, output_tokens=tout, thinking_tokens=tthink)
                self.stats.note(model_version=getattr(resp, "model_version", "") or "")
                text = resp.text or ""
                finish = str(getattr((resp.candidates or [None])[0], "finish_reason", "") or "")
                if "MAX_TOKENS" in finish:
                    # Тот же запрос снова обрежется: повторы только тратят квоту — сразу отдаём ошибку шагу
                    self.stats.add(role, failures=1)
                    self.stats.note(error=f"{role}: ответ обрезан по лимиту токенов")
                    raise LLMError("Ответ модели обрезан по лимиту токенов; шаг выполнен без LLM.")
                parsed = parse_json_loose(text)
                cache = self._cache_path(payload, model)  # ключ — модель, которая ответила (даже если другой поток уже переключил)
                if cache:
                    try:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        cache.write_text(json.dumps({"role": role, "model": model, "parsed": parsed},
                                                    ensure_ascii=False), encoding="utf-8")
                    except OSError:
                        pass
                return parsed
            except (ValueError, json.JSONDecodeError) as exc:
                # Невалидный JSON: повторяем с явным напоминанием
                self.stats.add(seconds=time.perf_counter() - started)
                last_error = f"невалидный JSON ({exc})"
                bad_json += 1
                if bad_json >= MAX_BAD_JSON:  # третий раз тот же запрос вряд ли даст другой ответ — бережём квоту
                    break
                contents = prompt + "\n\nВАЖНО: верни ТОЛЬКО валидный JSON по схеме, без пояснений и без обрезки."
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001 — сетевые ошибки, 429, 5xx, таймауты SDK
                self.stats.add(seconds=time.perf_counter() - started)
                last_error = self._describe(exc)
                status = self._status(exc)
                if status == 400 and self._thinking_ok and "thinking" in str(exc).lower():
                    self._thinking_ok = False  # модель не поддерживает уровень размышлений — повторяем без него
                    config = make_config()
                    continue
                if status in (400, 401, 403):
                    self.stats.add(failures=1)
                    raise LLMError(last_error) from exc
                if status == 429 and self._is_daily_quota(exc):
                    # Дневная квота не восстановится за минуты: ждать бессмысленно, сразу берём запасную модель
                    if self._switch_model(model, "дневная квота {} исчерпана"):
                        continue
                    self.stats.add(failures=1)
                    self._exhausted = (f"Исчерпана дневная квота бесплатного ключа для «{self.model}». "
                                       "Повторите после сброса квоты или укажите другую модель через --model.")
                    raise LLMError(self._exhausted) from exc
                if status == 404:
                    if self._switch_model(model, "модель {} недоступна"):
                        continue
                    self.stats.add(failures=1)
                    raise LLMError(f"Модель «{self.model}» не найдена. Укажите другую через GEMINI_MODEL или --model.") from exc
                if attempt < self.retries:
                    self.stats.add(retries=1)
                    time.sleep(self._backoff(exc, attempt))
                continue
            if attempt < self.retries:
                self.stats.add(retries=1)
        self.stats.add(role, failures=1)
        self.stats.note(error=f"{role}: {last_error}")
        raise LLMError(f"Модель не ответила корректно после {self.retries + 1} попыток: {last_error}")

    @staticmethod
    def _status(exc: Exception) -> int | None:
        for attr in ("code", "status_code"):
            val = getattr(exc, attr, None)
            if isinstance(val, int):
                return val
        m = re.search(r"\b(4\d\d|5\d\d)\b", str(exc)[:200])
        return int(m.group(1)) if m else None

    def _switch_model(self, failed: str, reason: str) -> bool:
        """Перейти на запасную модель. Потокобезопасно: если другой поток уже переключил — просто повторяем."""
        with self._switch_lock:
            if self.model != failed:
                return True
            if not self._fallbacks:
                return False
            self.model = self._fallbacks.pop(0)
            self.stats.note(error=f"{reason.format(failed)}, переключаюсь на {self.model}")
            return True

    @staticmethod
    def _is_daily_quota(exc: Exception) -> bool:
        s = str(exc)
        return bool(re.search(r"per ?day|PerDay|requests_per_day", s, re.I))

    @staticmethod
    def _describe(exc: Exception) -> str:
        msg = str(exc).replace("\n", " ")
        msg = re.sub(r"key=[\w-]+", "key=***", msg)  # на всякий случай не светим ключ
        return f"{type(exc).__name__}: {msg[:300]}"

    def _backoff(self, exc: Exception, attempt: int) -> float:
        # Уважаем подсказку сервера («retry in 21s» / retryDelay), иначе экспоненциальная пауза с джиттером
        m = re.search(r"retry(?:Delay)?[\"': ]+(?:in )?(\d+(?:\.\d+)?)s", str(exc), re.I)
        if m:
            return min(float(m.group(1)) + 1.0, 90.0)
        return min(2 ** attempt * 3 + random.uniform(0, 1.5), 60.0)


class FakeLLM:
    """Подставная модель для тестов: возвращает заранее заданные ответы по роли."""

    def __init__(self, responses: dict[str, Any] | None = None, model: str = "fake-model", fail_roles: set[str] | None = None):
        self.responses = responses or {}
        self.model = model
        self.stats = LLMStats(model_version=model)
        self.fail_roles = fail_roles or set()
        self.prompts: list[tuple[str, str]] = []

    def generate_json(self, role: str, system: str, prompt: str) -> Any:
        self.stats.add(role, calls=1)
        self.prompts.append((role, prompt))
        if role in self.fail_roles:
            self.stats.add(role, failures=1)
            raise LLMError(f"имитация отказа роли {role}")
        value = self.responses.get(role, {})
        return value(prompt) if callable(value) else value

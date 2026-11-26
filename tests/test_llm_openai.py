"""Тесты клиента OpenAI-совместимого API (Ollama, vLLM, LM Studio). Сеть подменена, живых запросов нет."""

import io
import json
import urllib.error
import urllib.request

import pytest

from reqcheck.cli import main
from reqcheck.llm import GeminiClient, LLMError, LLMUnavailable
from reqcheck.llm_openai import OpenAICompatClient, safe_base_url

BASE = "http://localhost:11434/v1"


class FakeResp:
    def __init__(self, content, usage=None):
        self._body = json.dumps({"choices": [{"message": {"content": content}}],
                                 "usage": usage or {"prompt_tokens": 5, "completion_tokens": 3}}).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_err(code, headers=None):
    return urllib.error.HTTPError(BASE, code, "err", headers or {}, io.BytesIO(b""))


@pytest.fixture
def net(monkeypatch):
    """Очередь ответов urlopen: объект исключения — бросается, иное — возвращается. Запросы копятся в calls."""
    state = {"queue": [], "calls": []}

    def fake(req, timeout=None):
        state["calls"].append(req)
        item = state["queue"].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return state


def client(tmp_path, **kw):
    kw.setdefault("cache_dir", str(tmp_path / "cache"))
    return OpenAICompatClient(base_url=BASE, model="llama3.1:8b", **kw)


def body(req):
    return json.loads(req.data)


def test_success_request_shape(net, tmp_path, monkeypatch):
    monkeypatch.delenv("REQCHECK_LLM_API_KEY", raising=False)
    net["queue"] = [FakeResp('{"a": 1}')]
    c = client(tmp_path)
    assert c.generate_json("extractor", "sys", "usr") == {"a": 1}
    req = net["calls"][0]
    assert req.full_url == BASE + "/chat/completions"
    b = body(req)
    assert b["temperature"] == 0 and b["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in b["messages"]] == ["system", "user"]
    assert not req.has_header("Authorization")
    assert c.stats.calls == 1 and c.stats.input_tokens == 5 and c.stats.output_tokens == 3


def test_api_key_header_only_when_set(net, tmp_path, monkeypatch):
    monkeypatch.setenv("REQCHECK_LLM_API_KEY", "k-test")
    net["queue"] = [FakeResp("{}")]
    client(tmp_path).generate_json("r", "s", "p")
    assert net["calls"][0].get_header("Authorization") == "Bearer k-test"


def test_json_fence_stripped(net, tmp_path):
    net["queue"] = [FakeResp('```json\n{"x": [1, 2]}\n```')]
    assert client(tmp_path).generate_json("r", "s", "p") == {"x": [1, 2]}


def test_400_response_format_retries_without_it(net, tmp_path):
    net["queue"] = [http_err(400), FakeResp("{}"), FakeResp("{}")]
    c = client(tmp_path)
    assert c.generate_json("r", "s", "p1") == {}
    assert "response_format" in body(net["calls"][0]) and "response_format" not in body(net["calls"][1])
    c.generate_json("r", "s", "p2")  # режим без response_format запоминается
    assert "response_format" not in body(net["calls"][2])


def test_400_twice_is_error(net, tmp_path):
    net["queue"] = [http_err(400), http_err(400)]
    with pytest.raises(LLMError):
        client(tmp_path).generate_json("r", "s", "p")


def test_429_retried(net, tmp_path):
    net["queue"] = [http_err(429, {"Retry-After": "1"}), FakeResp('{"ok": true}')]
    c = client(tmp_path)
    assert c.generate_json("r", "s", "p") == {"ok": True}
    assert c.stats.retries == 1


def test_5xx_exhausted(net, tmp_path):
    net["queue"] = [http_err(503)] * 3
    c = client(tmp_path, retries=2)
    with pytest.raises(LLMError, match="3 попыток"):
        c.generate_json("r", "s", "p")
    assert c.stats.failures == 1 and len(net["calls"]) == 3


def test_network_error_retried(net, tmp_path):
    net["queue"] = [urllib.error.URLError("refused"), FakeResp("{}")]
    assert client(tmp_path).generate_json("r", "s", "p") == {}


def test_404_message_without_url(net, tmp_path):
    net["queue"] = [http_err(404)]
    with pytest.raises(LLMError) as e:
        client(tmp_path).generate_json("r", "s", "p")
    assert "404" in str(e.value) and "localhost" not in str(e.value)


def test_bad_json_then_ok_and_bad_structure(net, tmp_path):
    net["queue"] = [FakeResp("не json"), FakeResp('{"ok": 1}')]
    assert client(tmp_path).generate_json("r", "s", "p") == {"ok": 1}
    net["queue"] = [FakeResp("нет"), FakeResp("опять нет")]
    with pytest.raises(LLMError):
        client(tmp_path).generate_json("r", "s", "p2")


def test_unexpected_structure(net, tmp_path):
    class Odd(FakeResp):
        def __init__(self):
            self._body = b'{"foo": 1}'
    net["queue"] = [Odd(), Odd()]
    with pytest.raises(LLMError):
        client(tmp_path).generate_json("r", "s", "p")


def test_call_limit(net, tmp_path):
    net["queue"] = [FakeResp("{}")]
    c = client(tmp_path, max_calls=1)
    c.generate_json("r", "s", "p1")
    with pytest.raises(LLMError, match="лимит"):
        c.generate_json("r", "s", "p2")
    assert len(net["calls"]) == 1


def test_cache_second_call_no_network(net, tmp_path):
    net["queue"] = [FakeResp('{"v": 7}')]
    c = client(tmp_path)
    assert c.generate_json("r", "s", "p") == {"v": 7}
    c2 = client(tmp_path)  # новый клиент, тот же кэш на диске
    assert c2.generate_json("r", "s", "p") == {"v": 7}
    assert len(net["calls"]) == 1 and c2.stats.cache_hits == 1 and c2.stats.calls == 0


def test_cache_corrupt_file_ignored(net, tmp_path):
    c = client(tmp_path)
    path = c._cache_path("s\n\np")
    path.parent.mkdir(parents=True)
    path.write_text("{битый", encoding="utf-8")
    net["queue"] = [FakeResp("{}")]
    assert c.generate_json("r", "s", "p") == {}


def test_cache_key_differs_from_gemini_and_other_url(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    g = GeminiClient(model="llama3.1:8b", cache_dir=str(tmp_path / "cache"))
    a = client(tmp_path)
    b = OpenAICompatClient(base_url="http://other:8000/v1", model="llama3.1:8b", cache_dir=str(tmp_path / "cache"))
    assert g._cache_path("p") != a._cache_path("p") != b._cache_path("p")
    assert a._cache_path("p") != b._cache_path("p")


def test_no_cache_dir(net, tmp_path):
    net["queue"] = [FakeResp("{}"), FakeResp("{}")]
    c = client(tmp_path, cache_dir=None)
    c.generate_json("r", "s", "p")
    c.generate_json("r", "s", "p")
    assert len(net["calls"]) == 2


def test_missing_base_url_and_model(monkeypatch):
    monkeypatch.delenv("REQCHECK_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("REQCHECK_LLM_MODEL", raising=False)
    with pytest.raises(LLMUnavailable, match="REQCHECK_LLM_BASE_URL"):
        OpenAICompatClient()
    with pytest.raises(LLMUnavailable, match="REQCHECK_LLM_MODEL"):
        OpenAICompatClient(base_url=BASE)
    monkeypatch.setenv("REQCHECK_LLM_MODEL", "m")
    assert OpenAICompatClient(base_url=BASE + "/").model == "m"


def test_safe_base_url():
    assert safe_base_url("http://user:pa55@host.local:8000/v1/x?k=1") == "http://host.local:8000"
    assert safe_base_url("http://[::1]:11434/v1") == "http://[::1]:11434"
    assert safe_base_url("localhost:11434/v1") == "(адрес скрыт)"  # без схемы не печатаем
    assert safe_base_url("") == "(адрес скрыт)"
    assert safe_base_url("http://h:99999999/v1") == "(адрес скрыт)"


# ---- выбор клиента и сквозной прогон ----

def reply_for_roles(req):
    """Корректный пустой ответ для любой роли: проходит разбор JSON, находок нет."""
    return FakeResp(json.dumps({"requirements": [], "findings": [], "items": [], "questions": [],
                                "checklist": [], "tests": [], "verdicts": []}))


@pytest.fixture
def e2e_net(monkeypatch):
    calls = []

    def fake(req, timeout=None):
        calls.append(req)
        return reply_for_roles(req)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return calls


def test_cli_selects_openai_compat_and_passport(e2e_net, sample_md, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REQCHECK_LLM_BASE_URL", "http://user:secret@localhost:11434/v1")
    monkeypatch.setenv("REQCHECK_LLM_MODEL", "llama3.1:8b")
    out = tmp_path / "r.md"
    assert main([str(sample_md), str(out), "--formats", "md,json"]) == 0
    assert e2e_net, "запросы должны идти на локальный сервер"
    assert all(r.full_url == "http://user:secret@localhost:11434/v1/chat/completions" for r in e2e_net)
    text = out.read_text(encoding="utf-8")
    data = (tmp_path / "r.json").read_text(encoding="utf-8")
    assert "llama3.1:8b" in text and "openai-compatible: http://localhost:11434" in text
    for blob in (text, data, capsys.readouterr().out):
        assert "secret" not in blob and "user:" not in blob


def test_cli_model_flag_overrides_env(e2e_net, sample_md, tmp_path, monkeypatch):
    monkeypatch.setenv("REQCHECK_LLM_BASE_URL", BASE)
    monkeypatch.setenv("REQCHECK_LLM_MODEL", "env-model")
    assert main([str(sample_md), str(tmp_path / "r.md"), "--model", "flag-model", "--formats", "md"]) == 0
    assert json.loads(e2e_net[0].data)["model"] == "flag-model"


def test_cli_missing_model_is_diagnostic(sample_md, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REQCHECK_LLM_BASE_URL", BASE)
    monkeypatch.delenv("REQCHECK_LLM_MODEL", raising=False)
    assert main([str(sample_md), str(tmp_path / "r.md"), "--formats", "md"]) == 0
    out = capsys.readouterr().out
    assert "REQCHECK_LLM_MODEL" in out and "GEMINI_API_KEY" not in out


def test_offline_ignores_base_url(sample_md, tmp_path, monkeypatch):
    monkeypatch.setenv("REQCHECK_LLM_BASE_URL", BASE)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("сеть не нужна"))
    assert main([str(sample_md), str(tmp_path / "r.md"), "--offline", "--formats", "md"]) == 0


def test_without_base_url_gemini_path_unchanged(sample_md, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("REQCHECK_LLM_BASE_URL", raising=False)
    assert main([str(sample_md), str(tmp_path / "r.md"), "--formats", "md"]) == 0
    assert "GEMINI_API_KEY" in capsys.readouterr().out

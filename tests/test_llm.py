"""Тесты llm.py: паспорт моделей и кэш GeminiClient."""


def test_passport_lists_every_model_including_cached_answers(tmp_path, monkeypatch):
    import json as _json
    from reqcheck.llm import GeminiClient
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    llm = GeminiClient(model="model-a", retries=0, cache_dir=str(tmp_path))
    llm._chain = ["model-a", "model-b"]  # как у модели по умолчанию с запасной
    for model, prompt in (("model-a", "p1"), ("model-b", "p2")):
        llm.model = model
        llm._cache_path("sys\n\n" + prompt).write_text(_json.dumps({"model": model, "parsed": {"ok": 1}}), encoding="utf-8")
        assert llm.generate_json("reviewer", "sys", prompt) == {"ok": 1}
    assert llm.stats.models_used == ["model-a", "model-b"]


def test_answer_after_model_switch_is_cached_under_the_model_that_answered(tmp_path, monkeypatch):
    import json as _json
    from types import SimpleNamespace
    from reqcheck.llm import GeminiClient
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    class Quota429(Exception):
        code = 429

    class Client:
        class models:  # noqa: N801
            @staticmethod
            def generate_content(model, contents, config):
                calls.append(model)
                if model == "model-a":
                    raise Quota429("429 RESOURCE_EXHAUSTED: Generate content requests per day limit exceeded")
                return SimpleNamespace(text='{"ok": 1}', usage_metadata=None, model_version=model,
                                       candidates=[SimpleNamespace(finish_reason="STOP")])

    def client(model):
        c = GeminiClient(model=None, retries=1, cache_dir=str(tmp_path))
        c.model, c._fallbacks, c._client, c.min_interval = model, ["model-b"], Client(), 0
        c._chain = [model, "model-b"]
        return c

    first = client("model-a")
    assert first.generate_json("critic", "sys", "p") == {"ok": 1} and calls == ["model-a", "model-b"]
    # Ответ model-b лежит под ключом model-b, а не под ключом модели, с которой начинали
    assert not first._cache_path("sys\n\np", "model-a").exists()
    assert _json.loads(first._cache_path("sys\n\np", "model-b").read_text(encoding="utf-8"))["model"] == "model-b"
    # Следующий прогон: поиск по цепочке моделей находит ответ запасной сразу, без запроса к основной
    calls.clear()
    second = client("model-a")
    assert second.generate_json("critic", "sys", "p") == {"ok": 1} and calls == []
    assert second.stats.models_used == ["model-b"]

"""Общие хелперы для тестов."""

from reqcheck.loaders import load_document


def _doc(tmp_path, text):
    path = tmp_path / "doc.md"
    # Нейтральная строка в конце: загрузчик отклоняет документы почти без текста
    path.write_text(text + "\n\nПримечание: тестовый документ для проверки правил агента.\n", encoding="utf-8")
    doc, _ = load_document(path)
    return doc

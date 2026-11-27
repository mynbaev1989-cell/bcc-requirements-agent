"""Экспорт вопросов в CSV для Jira: одна строка на вопрос, приоритет и роль адресата, цитаты со строками."""
import csv
import io

from reqcheck.models import Finding, Question, Quote, RunResult
from reqcheck.pipeline import write_outputs
from reqcheck.report_jira import render_jira_csv


def _result() -> RunResult:
    r = RunResult(document=None)
    r.passport["document_title"] = "Тест"
    r.findings = [Finding(fid="F-001", scope="requirement", criterion="unambiguous", severity="critical",
                          title="Метод GET меняет данные", explanation="", quotes=[Quote(text="REST type GET", line=104, found=True)])]
    r.questions = [Question(qid="Q-01", text="Почему GET?\nНужен POST?", priority="blocker", addressee="Бизнес заказчик",
                            options=["POST", "PUT"], assumption="POST", fids=["F-001"], lines=[104]),
                   Question(qid="Q-02", text="x" * 400, priority="clarify")]
    return r


def test_one_row_per_question_with_priority_and_role():
    text = render_jira_csv(_result())
    assert text.startswith("\ufeff")
    rows = list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))
    assert rows[0][0] == "Summary" and len(rows) == 3
    q1 = rows[1]
    assert q1[0] == "[Q-01] Почему GET? Нужен POST?" and q1[2] == "Highest" and q1[4] == "Бизнес_заказчик"
    assert "строки 104" in q1[5] and "«REST type GET»" in q1[5] and "(1) POST; (2) PUT" in q1[5]
    assert len(rows[2][0]) <= 250 and rows[2][2] == "Medium"


def test_no_questions_no_file(tmp_path):
    r = _result()
    written, _ = write_outputs(r, str(tmp_path / "r.md"), ("md", "csv"))
    assert any(p.endswith("r_jira.csv") for p in written)
    r.questions = []
    written, _ = write_outputs(r, str(tmp_path / "s.md"), ("md", "csv"))
    assert not any(p.endswith("_jira.csv") for p in written)

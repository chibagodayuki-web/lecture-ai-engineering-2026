"""ツールのテスト（LLMもAPIキーも使わない）。"""

import json
import runpy
import subprocess
import sys

import app.ask as ask_module
from app import tools


def test_glossary_ids_are_unique_and_related_links_exist():
    terms = tools.load_glossary()
    ids = [t["id"] for t in terms]
    assert len(ids) == len(set(ids))
    for t in terms:
        for r in t["related"]:
            assert r in ids, f"{t['id']} の related に存在しない id: {r}"


def test_search_finds_term_by_name_and_alias():
    for query in ["LoRA", "lora", "Low-Rank Adaptation"]:
        hits = json.loads(tools.search_glossary(query))
        assert hits[0]["term"] == "LoRA", query


def test_search_finds_course_specific_term():
    hits = json.loads(tools.search_glossary("lecNN-start"))
    assert hits[0]["term"] == "lecNN-start タグ"


def test_search_limits_number_of_results():
    hits = json.loads(tools.search_glossary("エージェント"))
    assert 1 <= len(hits) <= tools.MAX_RESULTS


def test_search_reports_not_found():
    assert "見つかりませんでした" in tools.search_glossary("存在しない用語xyz")


def test_lecture_summary():
    assert "エージェント" in tools.get_lecture_summary(3)
    assert "まだありません" in tools.get_lecture_summary(99)


def test_default_question_of_lecture1_is_answerable_from_glossary():
    """第1回の既定の質問（app.ask）は、第3回のエージェントが用語集で正しく答えられること。"""
    from app.ask import DEFAULT_QUESTION

    assert "3レベル" in DEFAULT_QUESTION
    hits = json.loads(tools.search_glossary("3レベル"))
    assert hits[0]["term"] == "宿題の3レベル"


def test_ask_prints_input_and_output_token_usage(capsys, monkeypatch):
    class FakeUsageMetadata:
        prompt_token_count = 123
        candidates_token_count = 45
        total_token_count = 168

    class FakeResponse:
        text = "テスト応答"
        usage_metadata = FakeUsageMetadata()

    class FakeModels:
        @staticmethod
        def generate_content(model, contents):
            assert model == ask_module.MODEL
            assert contents == "テスト質問"
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "fake-key"
            self.models = FakeModels()

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "fake-key")

    result = ask_module.ask("テスト質問")

    out = capsys.readouterr().out
    assert result == "テスト応答"
    assert "入力トークン数: 123" in out
    assert "出力トークン数: 45" in out


def test_ask_module_can_be_loaded_by_path_without_package_install():
    namespace = runpy.run_path("/workspaces/lecture-ai-engineering-2026/app/ask.py")
    assert "ask" in namespace


def test_ask_script_runs_when_executed_directly():
    script_path = "/workspaces/lecture-ai-engineering-2026/app/ask.py"
    python_code = f"""
import os
import runpy
import sys
import types

repo_root = '/workspaces/lecture-ai-engineering-2026'
sys.path = [p for p in sys.path if p and p != repo_root]

fake_google = types.ModuleType('google')
fake_genai = types.ModuleType('google.genai')

class FakeClient:
    def __init__(self, api_key):
        assert api_key == 'fake-key'

    @property
    def models(self):
        return types.SimpleNamespace(
            generate_content=lambda model, contents: types.SimpleNamespace(
                text='ok',
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=4,
                ),
            )
        )

fake_genai.Client = FakeClient
fake_genai.errors = types.SimpleNamespace(APIError=Exception)
fake_google.genai = fake_genai
sys.modules['google'] = fake_google
sys.modules['google.genai'] = fake_genai
os.environ['GEMINI_API_KEY'] = 'fake-key'
runpy.run_path({script_path!r}, run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", python_code],
        capture_output=True,
        text=True,
        cwd="/workspaces/lecture-ai-engineering-2026",
    )
    assert result.returncode == 0, result.stderr
    assert "[질문]" in result.stdout or "[質問]" in result.stdout
    assert "ok" in result.stdout

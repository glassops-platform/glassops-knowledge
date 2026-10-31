from pathlib import Path

import pytest
from google.genai import errors

from knowledge.generation import Generator
from knowledge.llm import GenerationError, LLMClient
from knowledge.settings import PROMPTS_PATH, ConfigError, generation_settings


def client_for(cfg, fakes, script, **kwargs):
    models = fakes.FakeModels(generate=script)
    llm = LLMClient(settings=generation_settings(cfg), client=fakes.client(models),
                    sleep=lambda s: None, **kwargs)
    return llm, models


# LLM client


def test_missing_api_key_raises(cfg):
    with pytest.raises(GenerationError, match="GOOGLE_API_KEY"):
        LLMClient(settings=generation_settings(cfg))


def test_default_model_comes_from_config(cfg, fakes):
    llm, models = client_for(cfg, fakes, [fakes.text_response("ok")])
    assert llm.generate("p") == "ok"
    assert models.generate_calls[0]["model"] == "test-docs-model"
    assert models.generate_calls[0]["config"].max_output_tokens == 256


def test_api_error_raises(cfg, fakes):
    err = errors.ClientError(404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}})
    llm, models = client_for(cfg, fakes, [err])
    with pytest.raises(GenerationError, match="call failed"):
        llm.generate("p")
    assert len(models.generate_calls) == 1


def test_retryable_error_then_success(cfg, fakes):
    err = errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
    llm, models = client_for(cfg, fakes, [err, fakes.text_response("ok")])
    assert llm.generate("p") == "ok"
    assert len(models.generate_calls) == 2


def test_retries_exhausted_raise(cfg, fakes):
    err = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    llm, _ = client_for(cfg, fakes, [err, err])
    with pytest.raises(GenerationError):
        llm.generate("p", max_retries=1)


def test_empty_text_raises(cfg, fakes):
    llm, _ = client_for(cfg, fakes, [fakes.text_response("", finish_reason="MAX_TOKENS")])
    with pytest.raises(GenerationError, match="MAX_TOKENS"):
        llm.generate("p")


# Generator


@pytest.fixture
def src_root(tmp_path):
    root = tmp_path / "repo"
    (root / "packages" / "pkg").mkdir(parents=True)
    (root / "packages" / "pkg" / "one.py").write_text("def one():\n    return 1\n", encoding="utf-8")
    (root / "packages" / "pkg" / "two.py").write_text("def two():\n    return 2\n", encoding="utf-8")
    return root


def test_prompts_load_from_package_not_scan_root(src_root):
    gen = Generator(str(src_root), llm=object())
    assert gen.prompts_path == PROMPTS_PATH
    assert not (src_root / "config" / "prompts.yml").exists()
    gen._load_prompts()
    assert "py" in gen.prompts and "_shared_rules" in gen.prompts


def test_shipped_prompts_are_used_for_python(src_root):
    gen = Generator(str(src_root), llm=object())
    gen._load_prompts()
    prompt = gen._get_prompt_for_file(src_root / "packages" / "pkg" / "one.py", "CODE")
    assert "STRICT RULES" in prompt and "CODE" in prompt


def test_missing_prompts_file_raises(src_root, tmp_path):
    gen = Generator(str(src_root), llm=object(), prompts_path=tmp_path / "none.yml")
    with pytest.raises(ConfigError, match="prompts file not found"):
        gen.run(["packages/**/*.py"])


def test_empty_prompts_file_raises(src_root, tmp_path):
    p = tmp_path / "p.yml"
    p.write_text("prompts: {}\n", encoding="utf-8")
    gen = Generator(str(src_root), llm=object(), prompts_path=p)
    with pytest.raises(ConfigError, match="no \"prompts\""):
        gen.run(["packages/**/*.py"])


def test_failed_file_is_counted_and_not_written(cfg, fakes, src_root):
    err = errors.ClientError(404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}})
    llm, _ = client_for(cfg, fakes, [err, fakes.text_response("Doc for two.")])
    gen = Generator(str(src_root), llm=llm)
    failures = gen.run(["packages/**/*.py"])
    assert failures == 1
    assert gen.failures == ["packages/pkg/one.py"]
    docs = src_root / "packages" / "pkg" / "docs"
    assert not (docs / "one.md").exists()
    assert (docs / "two.md").exists()
    import json
    cache = json.loads((src_root / "config" / "doc-cache.json").read_text(encoding="utf-8"))
    assert "packages/pkg/one.py" not in cache and "packages/pkg/two.py" in cache


def test_missing_key_fails_every_file(src_root):
    gen = Generator(str(src_root))
    assert gen.run(["packages/**/*.py"]) == 2


def test_all_success_returns_zero(cfg, fakes, src_root):
    llm, _ = client_for(cfg, fakes, [fakes.text_response("A."), fakes.text_response("B.")])
    assert Generator(str(src_root), llm=llm).run(["packages/**/*.py"]) == 0

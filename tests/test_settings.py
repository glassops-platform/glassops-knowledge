import json

import pytest

from knowledge.settings import (
    ConfigError,
    embedding_settings,
    generation_settings,
    load_config,
)


def test_shipped_config_loads():
    # The ids themselves live only in config/config.json. This checks that they load and that
    # the embedding size is the one the index was planned for.
    cfg = load_config()
    emb = embedding_settings(cfg)
    gen = generation_settings(cfg)
    assert emb.model and gen.docs_model and gen.query_model
    assert emb.dimensions == 768


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="could not read"):
        load_config(p)


def test_missing_models_section_raises():
    with pytest.raises(ConfigError, match='"models"'):
        embedding_settings({})


@pytest.mark.parametrize("dims", [None, 0, -1, "768", True])
def test_embedding_dimensions_must_be_explicit_positive_int(cfg, dims):
    if dims is None:
        cfg["models"]["embedding"].pop("dimensions")
    else:
        cfg["models"]["embedding"]["dimensions"] = dims
    with pytest.raises(ConfigError, match="dimensions"):
        embedding_settings(cfg)


@pytest.mark.parametrize("value", [None, "", "   ", 5])
def test_embedding_id_required(cfg, value):
    if value is None:
        cfg["models"]["embedding"].pop("id")
    else:
        cfg["models"]["embedding"]["id"] = value
    with pytest.raises(ConfigError, match="models.embedding.id"):
        embedding_settings(cfg)


@pytest.mark.parametrize("key", ["docs", "query"])
def test_generation_ids_required(cfg, key):
    cfg["models"]["generation"].pop(key)
    with pytest.raises(ConfigError, match=f"models.generation.{key}"):
        generation_settings(cfg)


def test_config_round_trips(tmp_path, cfg):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    assert embedding_settings(load_config(p)).model == "test-embed"

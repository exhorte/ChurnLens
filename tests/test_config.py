from dataclasses import replace

import pytest

from churnlens.config import Config, ConfigError, ModelsConfig, load_config


def test_toml_file_mirrors_dataclass_defaults():
    """configs/config.toml and the code defaults must never drift apart."""
    assert load_config() == Config()


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[split]\ntest_size = 0.2\ntypo_key = 1\n")
    with pytest.raises(ConfigError, match="typo_key"):
        load_config(path)


@pytest.mark.parametrize(
    "content",
    [
        "[split]\ntest_size = 0.9\n",
        "[cv]\nn_splits = 1\n",
        "[selection]\nrule = 'magic'\n",
        "[threshold]\ncriterion = 'accuracy'\n",
        "[segments]\nlow_below = 0.6\nhigh_from = 0.5\n",
    ],
)
def test_invalid_values_are_rejected(tmp_path, content):
    path = tmp_path / "c.toml"
    path.write_text(content)
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_explicit_file_and_bad_toml(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[split\n")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(bad)


def test_partial_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[split]\nrandom_state = 7\n")
    cfg = load_config(path)
    assert cfg.split.random_state == 7
    assert cfg.split.test_size == Config().split.test_size


def test_fingerprint_tracks_modelling_choices_not_paths():
    base = Config()
    other_paths = replace(base, paths=replace(base.paths, models_dir="elsewhere"))
    assert base.fingerprint() == other_paths.fingerprint()
    other_economics = replace(base, economics=replace(base.economics, cost_per_contact=99.0))
    assert base.fingerprint() == other_economics.fingerprint()      # simulator defaults do not change a model
    other_split = replace(base, split=replace(base.split, random_state=7))
    assert base.fingerprint() != other_split.fingerprint()
    other_model = replace(base, models=ModelsConfig(logistic_regression={"C": 0.1, "max_iter": 2000}))
    assert base.fingerprint() != other_model.fingerprint()

"""Config loading: `extends` resolution and deep-merge (SPEC.md §5)."""

import pytest

from xdr.config import load_config


def test_base_config_loads():
    config = load_config("base.yaml")
    assert config.seed == 20260807
    assert config.data.competitions.train[0].competition_id == 55
    assert config.config_name == "base"
    assert len(config.config_hash) == 12


def test_extends_deep_merges_over_base():
    config = load_config("transfer_euro24_to_weuro25.yaml")
    # Overridden by the child config.
    assert config.data.competitions.test[0].competition_id == 53
    assert config.evaluation.recalibrate_on_target is True
    # Inherited from base.yaml, not repeated in the child.
    assert config.split.unit == "match"
    assert config.model.lightgbm.n_estimators == 500


def test_unknown_config_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_different_configs_hash_differently():
    base = load_config("base.yaml")
    transfer = load_config("transfer_euro24_to_weuro25.yaml")
    assert base.config_hash != transfer.config_hash

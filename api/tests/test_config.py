"""Config loading: `extends` resolution and deep-merge (SPEC.md §5), plus the
repo-root resolution every path in the project is built on."""

import importlib

import pytest

import xdr.config
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


def test_repo_root_finds_configs_by_default():
    """The no-env-var path: walking up from the installed source has to land on
    a checkout that actually contains configs/."""
    assert (xdr.config.REPO_ROOT / "configs" / "base.yaml").exists()
    assert xdr.config.CONFIG_DIR == xdr.config.REPO_ROOT / "configs"


def test_xdr_root_env_var_overrides_repo_root(tmp_path, monkeypatch):
    """The Docker path (api/Dockerfile). pip installs the package into
    site-packages, where walking up from __file__ lands in the interpreter's
    lib/ directory instead of a checkout -- artifacts/ then silently resolves
    to nothing and every route serves empty. XDR_ROOT is the override, so it
    has to actually win over the computed default.
    """
    monkeypatch.setenv("XDR_ROOT", str(tmp_path))
    reloaded = importlib.reload(xdr.config)
    try:
        assert reloaded.REPO_ROOT == tmp_path.resolve()
        assert reloaded.CONFIG_DIR == tmp_path.resolve() / "configs"
    finally:
        # Module-level constants: leaving the reloaded module in place would
        # point every later test at tmp_path.
        monkeypatch.delenv("XDR_ROOT")
        importlib.reload(xdr.config)

"""Parameter-directory resolution."""

import warnings

import pytest

import experiment
import ste_vanilla
import vani_env
from vani_env import default_params_dir, resolve_params_dir


def _clear_params_env(monkeypatch):
    monkeypatch.delenv("ADAPR_PARAMS_DIR", raising=False)
    monkeypatch.delenv("ADAPR_VANILLA_ENV_DIR", raising=False)
    monkeypatch.delenv("ADAPR_EXPERIMENT_PARAMS_DIR", raising=False)
    monkeypatch.delenv("ADAPR_EST_PRIOR_PARAMS_DIR", raising=False)


def test_default_is_repo_env_para_vanilla(monkeypatch):
    _clear_params_env(monkeypatch)
    assert resolve_params_dir() == default_params_dir()
    assert default_params_dir() == (vani_env.PROJECT_ROOT / "env_para_vanilla").resolve()


def test_adapr_params_dir_selects_the_folder(monkeypatch):
    _clear_params_env(monkeypatch)
    monkeypatch.setenv("ADAPR_PARAMS_DIR", "env_para_ste0.5")
    assert resolve_params_dir() == (vani_env.PROJECT_ROOT / "env_para_ste0.5").resolve()


def test_mismatched_vanilla_alias_names_both_directories(monkeypatch):
    _clear_params_env(monkeypatch)
    monkeypatch.setenv("ADAPR_PARAMS_DIR", "env_para_ste0.2")
    monkeypatch.setenv("ADAPR_VANILLA_ENV_DIR", "env_para_ste0.8")
    with pytest.raises(ValueError, match="ADAPR_VANILLA_ENV_DIR") as exc:
        resolve_params_dir()
    message = str(exc.value)
    assert "env_para_ste0.2" in message
    assert "env_para_ste0.8" in message
    assert "ADAPR_PARAMS_DIR" in message


def test_mismatched_alias_against_the_default_names_both(monkeypatch):
    _clear_params_env(monkeypatch)
    monkeypatch.setenv("ADAPR_VANILLA_ENV_DIR", "env_para_ste0.5")
    with pytest.raises(ValueError, match="default parameter directory") as exc:
        resolve_params_dir()
    message = str(exc.value)
    assert "env_para_ste0.5" in message
    assert str(default_params_dir()) in message


def test_matching_alias_warns_and_uses_that_directory(monkeypatch):
    _clear_params_env(monkeypatch)
    monkeypatch.setenv("ADAPR_PARAMS_DIR", str(vani_env.PROJECT_ROOT / "env_para_ste0.5"))
    monkeypatch.setenv("ADAPR_VANILLA_ENV_DIR", "env_para_ste0.5")
    with pytest.warns(DeprecationWarning, match="ADAPR_VANILLA_ENV_DIR"):
        got = resolve_params_dir()
    assert got == (vani_env.PROJECT_ROOT / "env_para_ste0.5").resolve()


def test_wrappers_follow_the_shared_resolver(monkeypatch):
    _clear_params_env(monkeypatch)
    monkeypatch.setenv("ADAPR_PARAMS_DIR", "env_para_ste0.8")
    # The array launcher still passes its absolute path. The experiment
    # variable is not a second resolver.
    monkeypatch.setenv(
        "ADAPR_EXPERIMENT_PARAMS_DIR",
        str(vani_env.PROJECT_ROOT / "env_para_ste0.2"),
    )
    selected = (vani_env.PROJECT_ROOT / "env_para_ste0.8").resolve()
    assert experiment.resolve_params_dir() == selected
    assert ste_vanilla._resolve_params_dir() == selected
    explicit = vani_env.PROJECT_ROOT / "env_para_ste0.2"
    assert experiment.resolve_params_dir(explicit) == explicit.resolve()
    assert ste_vanilla._resolve_params_dir("env_para_ste0.2") == explicit.resolve()


def test_launchers_and_tune_ste_keep_ste_directories(monkeypatch):
    """run_array.sh, run_ste.sh, run_tune_ste.sh, and run_aggregate.sh.

    Each launcher cds to the repository and points at a folder name
    (``ENV_VARIANT``, ``ADAPR_PARAMS_DIR``, or ``TUNE_PARAMS_DIR``).
    ``tune_ste.py`` resolves ``--params-dir`` from that same working directory.
    """
    _clear_params_env(monkeypatch)
    repo = vani_env.PROJECT_ROOT
    for name in (
        "env_para_vanilla",
        "env_para_ste0.2",
        "env_para_ste0.5",
        "env_para_ste0.8",
    ):
        monkeypatch.setenv("ADAPR_PARAMS_DIR", name)
        resolved = resolve_params_dir()
        via_repo = (repo / name).resolve()
        assert resolved == via_repo
        # run_array.sh --params-dir "$SLURM_SUBMIT_DIR/$ENV_VARIANT"
        assert experiment.resolve_params_dir(via_repo) == via_repo
        # run_tune_ste.sh --params-dir "$TUNE_PARAMS_DIR" after cd to the repo.
        assert (repo / name).resolve() == via_repo


def test_alias_warning_is_a_deprecation(monkeypatch):
    _clear_params_env(monkeypatch)
    monkeypatch.setenv("ADAPR_VANILLA_ENV_DIR", "env_para_vanilla")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_params_dir()
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)

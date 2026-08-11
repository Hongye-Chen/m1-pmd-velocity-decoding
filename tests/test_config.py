"""Tests for configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_decoding.config import (
    DEFAULT_CONFIG_PATH,
    NWB_PATH_ENV_VAR,
    ConfigError,
    load_config,
    project_root,
)


@pytest.fixture
def no_local_config(tmp_path: Path) -> Path:
    """A path to a local config file that does not exist."""
    return tmp_path / "absent-local.yaml"


def test_tracked_defaults_are_machine_independent(no_local_config: Path) -> None:
    config = load_config(local_config_path=no_local_config, use_env=False)

    assert config.sources == (DEFAULT_CONFIG_PATH,)
    assert config.paths.nwb_file is None
    assert config.paths.results_dir == project_root() / "results"
    assert config.paths.reports_dir == project_root() / "reports"
    assert config.paths.log_dir == project_root() / "logs"
    assert isinstance(config.run.seed, int)
    assert config.run.log_level == "INFO"


def test_local_file_overrides_defaults(tmp_path: Path) -> None:
    local = tmp_path / "local.yaml"
    local.write_text(
        f'paths:\n  nwb_file: "{(tmp_path / "session.nwb").as_posix()}"\nrun:\n  seed: 7\n',
        encoding="utf-8",
    )

    config = load_config(local_config_path=local, use_env=False)

    assert config.paths.nwb_file == tmp_path / "session.nwb"
    assert config.run.seed == 7
    assert config.paths.results_dir == project_root() / "results"
    assert local in config.sources


def test_environment_variable_overrides_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local.yaml"
    local.write_text(
        f'paths:\n  nwb_file: "{(tmp_path / "from-file.nwb").as_posix()}"\n', encoding="utf-8"
    )
    env_path = tmp_path / "from-env.nwb"
    monkeypatch.setenv(NWB_PATH_ENV_VAR, str(env_path))

    config = load_config(local_config_path=local)

    assert config.paths.nwb_file == env_path


def test_explicit_overrides_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NWB_PATH_ENV_VAR, str(tmp_path / "from-env.nwb"))
    override = tmp_path / "explicit.nwb"

    config = load_config(
        local_config_path=tmp_path / "absent.yaml",
        overrides={"paths": {"nwb_file": str(override)}, "run": {"log_level": "debug"}},
    )

    assert config.paths.nwb_file == override
    assert config.run.log_level == "DEBUG"


def test_relative_paths_resolve_against_an_explicit_root(tmp_path: Path) -> None:
    config = load_config(
        local_config_path=tmp_path / "absent.yaml",
        overrides={"paths": {"nwb_file": "data/session.nwb"}},
        use_env=False,
        root=tmp_path,
    )

    assert config.paths.nwb_file == tmp_path / "data" / "session.nwb"
    assert config.paths.results_dir == tmp_path / "results"


def test_require_nwb_file_reports_an_unset_path(no_local_config: Path) -> None:
    config = load_config(local_config_path=no_local_config, use_env=False)

    with pytest.raises(ConfigError, match=NWB_PATH_ENV_VAR):
        config.require_nwb_file()


def test_require_nwb_file_reports_a_missing_file(tmp_path: Path) -> None:
    absent = tmp_path / "absent.nwb"
    config = load_config(
        local_config_path=tmp_path / "absent.yaml",
        overrides={"paths": {"nwb_file": str(absent)}},
        use_env=False,
    )

    with pytest.raises(ConfigError, match="does not exist"):
        config.require_nwb_file()


def test_require_nwb_file_returns_an_existing_file(tmp_path: Path) -> None:
    present = tmp_path / "session.nwb"
    present.write_bytes(b"")
    config = load_config(
        local_config_path=tmp_path / "absent.yaml",
        overrides={"paths": {"nwb_file": str(present)}},
        use_env=False,
    )

    assert config.require_nwb_file() == present


def test_ensure_directories_creates_output_locations(tmp_path: Path) -> None:
    config = load_config(
        local_config_path=tmp_path / "absent.yaml", use_env=False, root=tmp_path
    )

    config.ensure_directories()

    assert config.paths.results_dir.is_dir()
    assert config.paths.reports_dir.is_dir()
    assert config.paths.log_dir.is_dir()


def test_missing_defaults_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Missing configuration file"):
        load_config(config_path=tmp_path / "absent.yaml", use_env=False)


def test_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must contain a mapping"):
        load_config(config_path=bad, use_env=False)


def test_non_integer_seed_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="run.seed"):
        load_config(
            local_config_path=tmp_path / "absent.yaml",
            overrides={"run": {"seed": "not-a-number"}},
            use_env=False,
        )

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import dcp_invariant.cli as cli
from dcp_invariant.artifact import EvidenceArtifactError


def fake_artifact() -> SimpleNamespace:
    return SimpleNamespace(
        manifest_sha256="a" * 64,
        summary={
            "overall_status": "pass",
            "passed_scenarios": 10,
        },
    )


def test_verify_prints_only_normalized_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "verify_evidence_artifact",
        lambda root: fake_artifact(),
    )
    assert cli.main(["verify", "--artifact-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out == (
        '{"manifest_sha256":"'
        + ("a" * 64)
        + '","overall_status":"pass","passed_scenarios":10}\n'
    )


def test_run_imports_suite_lazily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = types.ModuleType("dcp_invariant.suite")
    module.run_suite = lambda *args, **kwargs: SimpleNamespace(artifact=fake_artifact())
    monkeypatch.setitem(sys.modules, "dcp_invariant.suite", module)

    assert (
        cli.main(
            [
                "run",
                "--output-dir",
                str(tmp_path / "evidence"),
                "--source-revision",
                "a" * 40,
            ]
        )
        == 0
    )
    assert '"passed_scenarios":10' in capsys.readouterr().out


def test_failure_output_does_not_echo_sensitive_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(root: Path) -> None:
        raise EvidenceArtifactError("sensitive absolute location")

    monkeypatch.setattr(cli, "verify_evidence_artifact", fail)
    with pytest.raises(SystemExit) as raised:
        cli.main(["verify", "--artifact-dir", str(tmp_path)])
    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "dcp-invariant: command failed\n"
    assert "sensitive" not in captured.err


def test_invalid_revision_is_rejected_before_live_import(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "run",
                "--output-dir",
                str(tmp_path / "evidence"),
                "--source-revision",
                "not-a-revision",
            ]
        )
    assert raised.value.code == 2
    assert "40-hex" in capsys.readouterr().err


def test_offline_verify_path_does_not_import_torch(tmp_path: Path) -> None:
    script = """
import sys
from dcp_invariant.cli import main
try:
    main(["verify", "--artifact-dir", sys.argv[1]])
except SystemExit:
    pass
raise SystemExit(1 if "torch" in sys.modules else 0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0

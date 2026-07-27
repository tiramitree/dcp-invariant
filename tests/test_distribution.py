from __future__ import annotations

import hashlib
import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest


def load_verifier():
    path = Path(__file__).parents[1] / "scripts" / "verify_distribution.py"
    spec = importlib.util.spec_from_file_location("verify_distribution", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_wheel(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def write_sdist(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
def test_clean_source_only_distribution_passes(tmp_path: Path, suffix: str) -> None:
    verifier = load_verifier()
    path = tmp_path / f"dcp_invariant{suffix}"
    members = {
        "dcp_invariant/__init__.py": b'__version__ = "0.1.0"\n',
        "dcp_invariant/cli.py": b"def main(): return 0\n",
    }
    if suffix == ".whl":
        write_wheel(path, members)
    else:
        write_sdist(path, members)
    result = verifier.verify_distribution(path)
    assert result["status"] == "PASS"
    assert result["member_count"] == 2
    assert result["archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "name",
    [
        "package/checkpoint/.metadata",
        "package/checkpoint/__0_0.distcp",
        "package/torch/__init__.py",
        "package/__pycache__/module.pyc",
    ],
)
def test_native_runtime_or_dependency_payload_is_rejected(
    tmp_path: Path,
    name: str,
) -> None:
    verifier = load_verifier()
    path = tmp_path / "bad.whl"
    write_wheel(path, {name: b"payload"})
    with pytest.raises(verifier.DistributionBoundaryError, match="forbidden"):
        verifier.verify_distribution(path)


def test_absolute_user_path_in_content_is_rejected(tmp_path: Path) -> None:
    verifier = load_verifier()
    path = tmp_path / "bad.whl"
    canary = b"value=C:" + bytes([92]) + b"Us" + b"ers" + bytes([92]) + b"candidate"
    write_wheel(path, {"package/value.txt": canary})
    with pytest.raises(verifier.DistributionBoundaryError, match="forbidden class"):
        verifier.verify_distribution(path)


def test_unsafe_archive_path_is_rejected(tmp_path: Path) -> None:
    verifier = load_verifier()
    path = tmp_path / "bad.whl"
    write_wheel(path, {"../outside.txt": b"no"})
    with pytest.raises(verifier.DistributionBoundaryError, match="unsafe"):
        verifier.verify_distribution(path)


def test_directory_input_expands_registered_archives_only(tmp_path: Path) -> None:
    verifier = load_verifier()
    wheel = tmp_path / "package.whl"
    sdist = tmp_path / "package.tar.gz"
    write_wheel(wheel, {"package/__init__.py": b""})
    write_sdist(sdist, {"package/__init__.py": b""})
    (tmp_path / "ignore.txt").write_text("not an archive", encoding="utf-8")
    assert verifier._expand_inputs([tmp_path]) == sorted(
        [wheel, sdist],
        key=lambda path: path.name,
    )


def test_external_denylist_rejects_content_without_echoing_value(
    tmp_path: Path,
) -> None:
    verifier = load_verifier()
    path = tmp_path / "private.whl"
    canary_literal = "private-publication-canary"
    write_wheel(path, {"package/value.txt": canary_literal.encode("utf-8")})
    with pytest.raises(
        verifier.DistributionBoundaryError,
        match="external denylist",
    ) as raised:
        verifier.verify_distribution(path, denylist=(canary_literal,))
    assert canary_literal not in str(raised.value)


def test_archive_is_snapshotted_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = load_verifier()
    path = tmp_path / "single-snapshot.whl"
    write_wheel(path, {"package/__init__.py": b""})
    original = verifier.Path.read_bytes
    reads = 0

    def counted_read(candidate: Path) -> bytes:
        nonlocal reads
        if candidate == path:
            reads += 1
        return original(candidate)

    monkeypatch.setattr(verifier.Path, "read_bytes", counted_read)
    verifier.verify_distribution(path)
    assert reads == 1


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
def test_total_unpacked_bound_is_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    verifier = load_verifier()
    path = tmp_path / f"bounded{suffix}"
    members = {
        "package/one.txt": b"1234",
        "package/two.txt": b"5678",
    }
    if suffix == ".whl":
        write_wheel(path, members)
    else:
        write_sdist(path, members)
    monkeypatch.setattr(verifier, "MAX_TOTAL_UNPACKED_BYTES", 7)
    with pytest.raises(
        verifier.DistributionBoundaryError,
        match="unpacked bytes",
    ):
        verifier.verify_distribution(path)

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import struct
import sys
import tarfile
import zipfile
import zlib
from pathlib import Path

import pytest


def load_verifier():
    path = Path(__file__).parents[1] / "scripts" / "verify_distribution.py"
    spec = importlib.util.spec_from_file_location("verify_distribution", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scanner():
    path = Path(__file__).parents[1] / "scripts" / "privacy_scan.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "privacy_scan_for_distribution",
            path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_distribution_patterns_match_source_gate() -> None:
    verifier = load_verifier()
    scanner = load_scanner()
    actual = {
        key: (value.pattern, value.flags)
        for key, value in verifier._SENSITIVE_PATTERNS.items()
    }
    expected = {
        key: (value.pattern, value.flags) for key, value in scanner._PATTERNS.items()
    }
    assert actual == expected


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


def write_manual_wheel(
    path: Path,
    *,
    name: str,
    flags: int = 0,
    local_extra: bytes = b"",
) -> None:
    name_bytes = name.encode("utf-8")
    data = b"safe"
    crc = zlib.crc32(data) & 0xFFFFFFFF
    local = struct.pack(
        "<4s5H3L2H",
        b"PK\x03\x04",
        20,
        flags,
        0,
        0,
        0,
        crc,
        len(data),
        len(data),
        len(name_bytes),
        len(local_extra),
    )
    local_record = local + name_bytes + local_extra + data
    central = struct.pack(
        "<4s6H3L5H2L",
        b"PK\x01\x02",
        20,
        20,
        flags,
        0,
        0,
        0,
        crc,
        len(data),
        len(data),
        len(name_bytes),
        0,
        0,
        0,
        0,
        0,
        0,
    )
    central_record = central + name_bytes
    eocd = struct.pack(
        "<4s4H2LH",
        b"PK\x05\x06",
        0,
        0,
        1,
        1,
        len(central_record),
        len(local_record),
        0,
    )
    path.write_bytes(local_record + central_record + eocd)


def test_wheel_rejects_non_ascii_name_without_utf8_flag(tmp_path: Path) -> None:
    verifier = load_verifier()
    path = tmp_path / "ambiguous.whl"
    write_manual_wheel(
        path,
        name="package/" + chr(0x2603) + ".txt",
        flags=0,
    )

    with pytest.raises(
        verifier.DistributionBoundaryError,
        match="member name is invalid",
    ):
        verifier.verify_distribution(path)


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


def test_snapshot_api_preserves_directory_type(tmp_path: Path) -> None:
    verifier = load_verifier()
    path = tmp_path / "evidence.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        directory = tarfile.TarInfo("evidence")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        payload = b"safe\n"
        info = tarfile.TarInfo("evidence/summary.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    result, members = verifier.snapshot_verified_distribution(path)

    assert result["status"] == "PASS"
    assert [(member.name, member.is_directory) for member in members] == [
        ("evidence", True),
        ("evidence/summary.json", False),
    ]


@pytest.mark.parametrize(
    "name",
    [
        "package/checkpoint/.metadata",
        "package/checkpoint/__0_0.distcp",
        "package/torch/__init__.py",
        "package/numpy/__init__.py",
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


def test_distribution_denylist_utf8_bom_is_rejected(tmp_path: Path) -> None:
    verifier = load_verifier()
    denylist = tmp_path / "denylist.txt"
    denylist.write_bytes(b"\xef\xbb\xbfprivate-canary\n")

    with pytest.raises(verifier.DistributionBoundaryError, match="BOM"):
        verifier._load_denylist(denylist, required=True)


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
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff\xfe\x00\x01", "opaque"),
        (
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:" + (b"a" * 64) + b"\nsize 1\n",
            "opaque",
        ),
    ],
)
def test_opaque_or_lfs_archive_member_fails_closed(
    tmp_path: Path,
    suffix: str,
    payload: bytes,
    message: str,
) -> None:
    verifier = load_verifier()
    path = tmp_path / f"opaque{suffix}"
    members = {"package/opaque.bin": payload}
    if suffix == ".whl":
        write_wheel(path, members)
    else:
        write_sdist(path, members)

    with pytest.raises(verifier.DistributionBoundaryError, match=message):
        verifier.verify_distribution(path)


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
@pytest.mark.parametrize(
    "payload",
    [
        b"sk-" + (b"A" * 24),
        b"https://"
        + b"bounded-user:bounded-secret"
        + bytes((64,))
        + b"example.invalid/path",
    ],
)
def test_distribution_rejects_registered_credential_patterns(
    tmp_path: Path,
    suffix: str,
    payload: bytes,
) -> None:
    verifier = load_verifier()
    path = tmp_path / f"credential{suffix}"
    members = {"package/value.txt": payload}
    if suffix == ".whl":
        write_wheel(path, members)
    else:
        write_sdist(path, members)
    with pytest.raises(verifier.DistributionBoundaryError, match="forbidden class"):
        verifier.verify_distribution(path)


def test_wheel_archive_comment_is_rejected(tmp_path: Path) -> None:
    verifier = load_verifier()
    path = tmp_path / "metadata.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("package/value.txt", b"safe")
        archive.comment = b"private-metadata-canary"

    with pytest.raises(
        verifier.DistributionBoundaryError,
        match="unregistered ZIP metadata",
    ):
        verifier.verify_distribution(
            path,
            denylist=("private-metadata-canary",),
        )


@pytest.mark.parametrize("field", ["comment", "extra"])
def test_wheel_member_text_metadata_is_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    verifier = load_verifier()
    path = tmp_path / "metadata.whl"
    info = zipfile.ZipInfo("package/value.txt")
    if field == "comment":
        info.comment = b"private-metadata-canary"
    else:
        payload = b"private-metadata-canary"
        info.extra = b"\xfe\xca" + len(payload).to_bytes(2, "little") + payload
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, b"safe")

    with pytest.raises(
        verifier.DistributionBoundaryError,
        match="unregistered ZIP metadata",
    ):
        verifier.verify_distribution(
            path,
            denylist=("private-metadata-canary",),
        )


@pytest.mark.parametrize("field", ["uname", "gname", "pax_headers"])
def test_sdist_member_text_metadata_is_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    verifier = load_verifier()
    path = tmp_path / "metadata.tar.gz"
    payload = b"safe"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("package/value.txt")
        info.size = len(payload)
        if field == "pax_headers":
            info.pax_headers = {"comment": "private-metadata-canary"}
        else:
            setattr(info, field, "private-metadata-canary")
        archive.addfile(info, io.BytesIO(payload))

    with pytest.raises(
        verifier.DistributionBoundaryError,
        match="external denylist|unregistered TAR metadata",
    ):
        verifier.verify_distribution(
            path,
            denylist=("private-metadata-canary",),
        )


def test_sdist_gzip_filename_metadata_is_rejected(tmp_path: Path) -> None:
    verifier = load_verifier()
    path = tmp_path / "metadata.tar.gz"
    tar_payload = io.BytesIO()
    with tarfile.open(fileobj=tar_payload, mode="w") as archive:
        payload = b"safe"
        info = tarfile.TarInfo("package/value.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with (
        path.open("wb") as raw_output,
        gzip.GzipFile(
            filename="private-metadata-canary",
            mode="wb",
            fileobj=raw_output,
            mtime=0,
        ) as compressed,
    ):
        compressed.write(tar_payload.getvalue())

    with pytest.raises(
        verifier.DistributionBoundaryError,
        match="unregistered gzip metadata",
    ):
        verifier.verify_distribution(
            path,
            denylist=("private-metadata-canary",),
        )


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
def test_archive_directory_names_are_privacy_scanned(
    tmp_path: Path,
    suffix: str,
) -> None:
    verifier = load_verifier()
    path = tmp_path / f"directory{suffix}"
    canary = "private-directory-canary"
    if suffix == ".whl":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{canary}/", b"")
    else:
        with tarfile.open(path, "w:gz") as archive:
            info = tarfile.TarInfo(canary)
            info.type = tarfile.DIRTYPE
            info.size = 0
            archive.addfile(info)

    with pytest.raises(
        verifier.DistributionBoundaryError,
        match="external denylist",
    ):
        verifier.verify_distribution(
            path,
            denylist=(canary,),
        )


@pytest.mark.parametrize("location", ["prefix", "suffix", "internal-gap"])
def test_wheel_rejects_unregistered_container_bytes(
    tmp_path: Path,
    location: str,
) -> None:
    verifier = load_verifier()
    path = tmp_path / "container.whl"
    canary = b"archive-container-private-canary"
    write_wheel(path, {"package/value.txt": b"safe"})
    raw = path.read_bytes()
    if location == "prefix":
        mutated = canary + raw
    elif location == "suffix":
        mutated = raw + canary
    else:
        eocd_offset = len(raw) - 22
        central_offset = struct.unpack_from("<L", raw, eocd_offset + 16)[0]
        value = bytearray(raw[:central_offset] + canary + raw[central_offset:])
        struct.pack_into(
            "<L",
            value,
            eocd_offset + len(canary) + 16,
            central_offset + len(canary),
        )
        mutated = bytes(value)
    path.write_bytes(mutated)

    with pytest.raises(verifier.DistributionBoundaryError) as raised:
        verifier.verify_distribution(
            path,
            denylist=(canary.decode("ascii"),),
        )
    assert canary.decode("ascii") not in str(raised.value)


def test_wheel_rejects_local_only_extra_bytes(tmp_path: Path) -> None:
    verifier = load_verifier()
    path = tmp_path / "local-extra.whl"
    canary = "local-extra-private-canary"
    write_manual_wheel(
        path,
        name="package/value.txt",
        local_extra=canary.encode("ascii"),
    )

    with pytest.raises(verifier.DistributionBoundaryError) as raised:
        verifier.verify_distribution(path, denylist=(canary,))
    assert canary not in str(raised.value)


def test_wheel_encryption_rejection_never_echoes_member_name(tmp_path: Path) -> None:
    verifier = load_verifier()
    path = tmp_path / "encrypted.whl"
    canary = "encrypted-name-private-canary"
    write_manual_wheel(path, name=f"package/{canary}.txt", flags=1)

    with pytest.raises(verifier.DistributionBoundaryError) as raised:
        verifier.verify_distribution(path, denylist=(canary,))
    assert canary not in str(raised.value)


@pytest.mark.parametrize("suffix_kind", ["plain", "second-gzip-member"])
def test_sdist_rejects_bytes_after_single_gzip_member(
    tmp_path: Path,
    suffix_kind: str,
) -> None:
    verifier = load_verifier()
    path = tmp_path / "trailing.tar.gz"
    canary = b"archive-trailing-private-canary"
    write_sdist(path, {"package/value.txt": b"safe"})
    suffix = canary if suffix_kind == "plain" else gzip.compress(canary, mtime=0)
    path.write_bytes(path.read_bytes() + suffix)

    with pytest.raises(verifier.DistributionBoundaryError) as raised:
        verifier.verify_distribution(
            path,
            denylist=(canary.decode("ascii"),),
        )
    assert canary.decode("ascii") not in str(raised.value)


def test_sdist_rejects_nonzero_member_padding(tmp_path: Path) -> None:
    verifier = load_verifier()
    path = tmp_path / "padding.tar.gz"
    canary = b"tar-padding-private-canary"
    tar_payload = io.BytesIO()
    with tarfile.open(fileobj=tar_payload, mode="w") as archive:
        info = tarfile.TarInfo("package/value.txt")
        info.size = 2
        archive.addfile(info, io.BytesIO(b"ok"))
    raw_tar = bytearray(tar_payload.getvalue())
    raw_tar[514 : 514 + len(canary)] = canary
    with (
        path.open("wb") as output,
        gzip.GzipFile(
            filename=path.name[:-3],
            mode="wb",
            fileobj=output,
            mtime=0,
        ) as compressed,
    ):
        compressed.write(raw_tar)

    with pytest.raises(verifier.DistributionBoundaryError) as raised:
        verifier.verify_distribution(
            path,
            denylist=(canary.decode("ascii"),),
        )
    assert canary.decode("ascii") not in str(raised.value)


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

import hashlib
import io
import zipfile

import pytest

import quantum_lake_student.stages.register_sources as register_sources


def _fake_manifest(objects: list[dict]) -> dict:
    return {"bundle_version": 3, "release_name": "test", "objects": objects}


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _manifest_object(path: str, data: bytes) -> dict:
    return {
        "path": path,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mandatory": True,
        "source": path.split("/")[1].removeprefix("source="),
        "source_member": None,
        "teaching_role": "test fixture",
    }


def _patch_platform(monkeypatch, *, bronze_sizes: dict[str, int], object_bytes: dict[str, bytes]) -> None:
    monkeypatch.setattr(register_sources, "bronze_inventory", lambda settings: list(bronze_sizes.items()))
    monkeypatch.setattr(register_sources, "minio_client", lambda settings: object())
    monkeypatch.setattr(register_sources, "get_object_bytes", lambda client, bucket, key: object_bytes[key])


def test_bronze_key_translates_raw_prefix() -> None:
    assert register_sources._bronze_key("raw/source=qec_syndromes/x.zip") == "bronze/source=qec_syndromes/x.zip"


def test_bronze_key_rejects_unexpected_prefix() -> None:
    with pytest.raises(ValueError):
        register_sources._bronze_key("bronze/source=qec_syndromes/x.zip")


def test_register_sources_happy_path(monkeypatch) -> None:
    data = _zip_bytes({"a.csv": b"hello"})
    manifest = _fake_manifest([_manifest_object("raw/source=qec_syndromes/x.zip", data)])
    key = "bronze/source=qec_syndromes/x.zip"

    monkeypatch.setattr(register_sources, "_load_manifest", lambda: manifest)
    _patch_platform(monkeypatch, bronze_sizes={key: len(data)}, object_bytes={key: data})

    result = register_sources.run("test-run")

    assert result.input_count == 1
    assert result.output_count == 1
    assert result.issue_count == 0
    assert result.finished_at is not None


def test_register_sources_raises_on_missing_object(monkeypatch) -> None:
    data = _zip_bytes({"a.csv": b"hello"})
    manifest = _fake_manifest([_manifest_object("raw/source=qec_syndromes/x.zip", data)])

    monkeypatch.setattr(register_sources, "_load_manifest", lambda: manifest)
    _patch_platform(monkeypatch, bronze_sizes={}, object_bytes={})

    with pytest.raises(RuntimeError, match="Missing required Bronze object"):
        register_sources.run("test-run")


def test_register_sources_raises_on_checksum_mismatch(monkeypatch) -> None:
    data = _zip_bytes({"a.csv": b"hello"})
    tampered = _zip_bytes({"a.csv": b"tampered"})
    manifest = _fake_manifest([_manifest_object("raw/source=qec_syndromes/x.zip", data)])
    key = "bronze/source=qec_syndromes/x.zip"

    monkeypatch.setattr(register_sources, "_load_manifest", lambda: manifest)
    # size matches (same length by construction below) but content -- and therefore hash -- does not
    tampered = tampered[: len(data)] if len(tampered) > len(data) else tampered + b"\x00" * (len(data) - len(tampered))
    _patch_platform(monkeypatch, bronze_sizes={key: len(data)}, object_bytes={key: tampered})

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        register_sources.run("test-run")


def test_register_sources_raises_on_size_mismatch(monkeypatch) -> None:
    data = _zip_bytes({"a.csv": b"hello"})
    manifest = _fake_manifest([_manifest_object("raw/source=qec_syndromes/x.zip", data)])
    key = "bronze/source=qec_syndromes/x.zip"

    monkeypatch.setattr(register_sources, "_load_manifest", lambda: manifest)
    _patch_platform(monkeypatch, bronze_sizes={key: len(data) + 1}, object_bytes={key: data})

    with pytest.raises(RuntimeError, match="size mismatch"):
        register_sources.run("test-run")


def test_register_sources_raises_on_unsafe_archive_member(monkeypatch) -> None:
    data = _zip_bytes({"../evil.txt": b"payload"})
    manifest = _fake_manifest([_manifest_object("raw/source=qec_syndromes/x.zip", data)])
    key = "bronze/source=qec_syndromes/x.zip"

    monkeypatch.setattr(register_sources, "_load_manifest", lambda: manifest)
    _patch_platform(monkeypatch, bronze_sizes={key: len(data)}, object_bytes={key: data})

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        register_sources.run("test-run")


def test_register_sources_reports_unexpected_object_without_raising(monkeypatch) -> None:
    data = _zip_bytes({"a.csv": b"hello"})
    manifest = _fake_manifest([_manifest_object("raw/source=qec_syndromes/x.zip", data)])
    key = "bronze/source=qec_syndromes/x.zip"
    extra_key = "bronze/source=qec_syndromes/unexpected.zip"

    monkeypatch.setattr(register_sources, "_load_manifest", lambda: manifest)
    _patch_platform(
        monkeypatch,
        bronze_sizes={key: len(data), extra_key: 10},
        object_bytes={key: data},
    )

    result = register_sources.run("test-run")

    assert result.issue_count == 1
    assert result.input_count == 1

"""Tests for memboot.licensing."""

from __future__ import annotations

from pathlib import Path

from memboot.licensing import (
    TIER_DEFINITIONS,
    Tier,
    TierConfig,
    _compute_check_segment,
    _find_license_key,
    _validate_key_checksum,
    _validate_key_format,
    get_license_info,
    get_upgrade_message,
    has_feature,
    is_pro,
)


def _make_valid_key() -> str:
    """Generate a valid MMBT license key."""
    body = "TEST-ABCD"
    check = _compute_check_segment(body)
    return f"MMBT-{body}-{check}"


class TestTiers:
    def test_tier_values(self):
        assert Tier.FREE == "free"
        assert Tier.PRO == "pro"

    def test_free_features(self):
        config = TIER_DEFINITIONS[Tier.FREE]
        assert "init" in config.features
        assert "query" in config.features
        assert "remember" in config.features
        assert "serve" not in config.features

    def test_pro_features(self):
        config = TIER_DEFINITIONS[Tier.PRO]
        assert "serve" in config.features
        assert "ingest_pdf" in config.features
        assert "ingest_web" in config.features

    def test_tier_config_fields(self):
        config = TIER_DEFINITIONS[Tier.FREE]
        assert isinstance(config, TierConfig)
        assert config.name == "Free"
        assert config.price_label


class TestValidateKeyFormat:
    def test_valid_key(self):
        assert _validate_key_format("MMBT-ABCD-EFGH-IJKL")

    def test_wrong_prefix(self):
        assert not _validate_key_format("XXXX-ABCD-EFGH-IJKL")

    def test_too_few_parts(self):
        assert not _validate_key_format("MMBT-ABCD-EFGH")

    def test_too_many_parts(self):
        assert not _validate_key_format("MMBT-ABCD-EFGH-IJKL-MNOP")

    def test_lowercase_fails(self):
        assert not _validate_key_format("MMBT-abcd-EFGH-IJKL")

    def test_wrong_segment_length(self):
        assert not _validate_key_format("MMBT-ABC-EFGH-IJKL")

    def test_with_whitespace(self):
        assert _validate_key_format("  MMBT-ABCD-EFGH-IJKL  ")


class TestValidateKeyChecksum:
    def test_valid_checksum(self):
        key = _make_valid_key()
        assert _validate_key_checksum(key)

    def test_tampered_key(self):
        key = _make_valid_key()
        # Change the last segment
        parts = key.split("-")
        parts[3] = "ZZZZ"
        tampered = "-".join(parts)
        assert not _validate_key_checksum(tampered)

    def test_wrong_part_count(self):
        assert not _validate_key_checksum("MMBT-ABCD")


class TestComputeCheckSegment:
    def test_deterministic(self):
        seg1 = _compute_check_segment("TEST-ABCD")
        seg2 = _compute_check_segment("TEST-ABCD")
        assert seg1 == seg2

    def test_uppercase(self):
        seg = _compute_check_segment("TEST-ABCD")
        assert seg == seg.upper()

    def test_length_four(self):
        seg = _compute_check_segment("TEST-ABCD")
        assert len(seg) == 4

    def test_different_bodies(self):
        seg1 = _compute_check_segment("AAAA-BBBB")
        seg2 = _compute_check_segment("CCCC-DDDD")
        assert seg1 != seg2


class TestFindLicenseKey:
    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("MEMBOOT_LICENSE", "MMBT-ABCD-EFGH-IJKL")
        key = _find_license_key()
        assert key == "MMBT-ABCD-EFGH-IJKL"

    def test_env_var_empty(self, monkeypatch):
        monkeypatch.setenv("MEMBOOT_LICENSE", "")
        monkeypatch.setattr("memboot.licensing._LICENSE_LOCATIONS", [])
        key = _find_license_key()
        assert key is None

    def test_file_location(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("MEMBOOT_LICENSE", raising=False)
        license_file = tmp_path / ".memboot-license"
        license_file.write_text("MMBT-FILE-KEYS-HERE")
        monkeypatch.setattr(
            "memboot.licensing._LICENSE_LOCATIONS",
            [str(license_file)],
        )
        key = _find_license_key()
        assert key == "MMBT-FILE-KEYS-HERE"

    def test_no_key_found(self, monkeypatch):
        monkeypatch.delenv("MEMBOOT_LICENSE", raising=False)
        monkeypatch.setattr("memboot.licensing._LICENSE_LOCATIONS", [])
        key = _find_license_key()
        assert key is None


class TestGetLicenseInfo:
    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("MEMBOOT_LICENSE", raising=False)
        monkeypatch.setattr("memboot.licensing._LICENSE_LOCATIONS", [])
        info = get_license_info()
        assert info.tier == Tier.FREE
        assert info.license_key is None

    def test_valid_key(self, monkeypatch):
        key = _make_valid_key()
        monkeypatch.setenv("MEMBOOT_LICENSE", key)
        info = get_license_info()
        assert info.tier == Tier.PRO
        assert info.valid is True
        assert info.license_key == key

    def test_invalid_format(self, monkeypatch):
        monkeypatch.setenv("MEMBOOT_LICENSE", "bad-key")
        info = get_license_info()
        assert info.tier == Tier.FREE
        assert info.valid is False

    def test_bad_checksum(self, monkeypatch):
        monkeypatch.setenv("MEMBOOT_LICENSE", "MMBT-ABCD-EFGH-ZZZZ")
        info = get_license_info()
        assert info.tier == Tier.FREE
        assert info.valid is False


class TestHasFeature:
    def test_free_feature_without_key(self, monkeypatch):
        monkeypatch.delenv("MEMBOOT_LICENSE", raising=False)
        monkeypatch.setattr("memboot.licensing._LICENSE_LOCATIONS", [])
        assert has_feature("init") is True
        assert has_feature("query") is True

    def test_pro_feature_without_key(self, monkeypatch):
        monkeypatch.delenv("MEMBOOT_LICENSE", raising=False)
        monkeypatch.setattr("memboot.licensing._LICENSE_LOCATIONS", [])
        assert has_feature("serve") is False
        assert has_feature("ingest_pdf") is False

    def test_pro_feature_with_key(self, monkeypatch):
        key = _make_valid_key()
        monkeypatch.setenv("MEMBOOT_LICENSE", key)
        assert has_feature("serve") is True
        assert has_feature("ingest_pdf") is True


class TestIsPro:
    def test_without_key(self, monkeypatch):
        monkeypatch.delenv("MEMBOOT_LICENSE", raising=False)
        monkeypatch.setattr("memboot.licensing._LICENSE_LOCATIONS", [])
        assert is_pro() is False

    def test_with_valid_key(self, monkeypatch):
        key = _make_valid_key()
        monkeypatch.setenv("MEMBOOT_LICENSE", key)
        assert is_pro() is True


class TestGetUpgradeMessage:
    def test_contains_feature(self):
        msg = get_upgrade_message("serve")
        assert "serve" in msg

    def test_contains_env_var(self):
        msg = get_upgrade_message("serve")
        assert "MEMBOOT_LICENSE" in msg

    def test_contains_price(self):
        msg = get_upgrade_message("serve")
        assert "$8/mo" in msg

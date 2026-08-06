"""
Tests for §1.8A — load_asp_config (backend/src/animation/config.py).
Tests for §1.8B — validate_asp_config schema validation.
Tests for §1.8C — dump_asp_config TOML serialization.
Tests for §1.8D — dump_asp_config typed schema comments.

Verifies TOML loading, env-var injection, setdefault precedence, multi-section
flattening, the override_env=False dry-run mode, config schema validation, and
TOML dump with machine-readable type/range annotations.
"""

import os
import warnings

import pytest
from backend.src.animation.core.config import (
    dump_asp_config,
    load_asp_config,
    validate_asp_config,
)
from backend.src.errors import ConfigError


class TestLoadAspConfig:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        result = load_asp_config(str(tmp_path / "nonexistent.toml"))
        assert result == {}

    def test_valid_config_sets_env_var(self, tmp_path, monkeypatch):
        cfg = tmp_path / "asp_config.toml"
        cfg.write_text("[frame_selection]\nASP_NEAR_DUP_LUMA = 7.5\n")
        monkeypatch.delenv("ASP_NEAR_DUP_LUMA", raising=False)
        result = load_asp_config(str(cfg))
        assert result.get("ASP_NEAR_DUP_LUMA") == pytest.approx(7.5)
        assert os.environ.get("ASP_NEAR_DUP_LUMA") == "7.5"

    def test_existing_env_var_not_overwritten(self, tmp_path, monkeypatch):
        cfg = tmp_path / "asp_config.toml"
        cfg.write_text("[frame_selection]\nASP_NEAR_DUP_LUMA = 7.5\n")
        monkeypatch.setenv("ASP_NEAR_DUP_LUMA", "3.0")
        load_asp_config(str(cfg))
        assert os.environ.get("ASP_NEAR_DUP_LUMA") == "3.0"

    def test_multi_section_keys_flattened(self, tmp_path, monkeypatch):
        cfg = tmp_path / "asp_config.toml"
        cfg.write_text(
            "[frame_selection]\nASP_HOLD_THRESHOLD = 0.03\n\n"
            "[compositing]\nASP_SP_SOFT_PX = 8\n"
        )
        monkeypatch.delenv("ASP_HOLD_THRESHOLD", raising=False)
        monkeypatch.delenv("ASP_SP_SOFT_PX", raising=False)
        result = load_asp_config(str(cfg))
        assert "ASP_HOLD_THRESHOLD" in result
        assert "ASP_SP_SOFT_PX" in result
        assert result["ASP_HOLD_THRESHOLD"] == pytest.approx(0.03)
        assert result["ASP_SP_SOFT_PX"] == 8

    def test_override_env_false_does_not_write_env(self, tmp_path, monkeypatch):
        cfg = tmp_path / "asp_config.toml"
        cfg.write_text("[pipeline]\nASP_COV_MIN_MULTI_PCT = 0.45\n")
        monkeypatch.delenv("ASP_COV_MIN_MULTI_PCT", raising=False)
        result = load_asp_config(str(cfg), override_env=False)
        assert result.get("ASP_COV_MIN_MULTI_PCT") == pytest.approx(0.45)
        assert "ASP_COV_MIN_MULTI_PCT" not in os.environ


# ---------------------------------------------------------------------------
# TestValidateAspConfig — §1.8B schema validation (S42)
# ---------------------------------------------------------------------------


class TestValidateAspConfig:
    """
    validate_asp_config(config, strict=False) checks known ASP keys against
    _CONFIG_SCHEMA: type correctness and min/max bounds.  Unknown keys emit a
    UserWarning but are not violations.  strict=True raises ValueError.
    """

    def test_valid_known_keys_return_empty_violations(self):
        """A well-formed config with schema-valid values must produce no violations."""
        config = {
            "ASP_HOLD_THRESHOLD": 0.03,
            "ASP_COV_MIN_MULTI_PCT": 0.30,
            "ASP_SP_SOFT_PX": 6,
            "ASP_HOLD_AVERAGE": 0,
        }
        assert validate_asp_config(config) == []

    def test_wrong_type_produces_violation(self):
        """A string value where float is expected must produce a violation message."""
        config = {"ASP_HOLD_THRESHOLD": "not-a-number"}
        violations = validate_asp_config(config)
        assert len(violations) == 1
        assert "ASP_HOLD_THRESHOLD" in violations[0]
        assert "float" in violations[0]

    def test_out_of_range_value_produces_violation(self):
        """A value exceeding the schema maximum must produce a violation message."""
        config = {"ASP_HIGH_HOLD_RESPONSE": 1.5}  # max is 1.0
        violations = validate_asp_config(config)
        assert len(violations) == 1
        assert "ASP_HIGH_HOLD_RESPONSE" in violations[0]
        assert "1.0" in violations[0]

    def test_strict_raises_on_violations(self):
        """strict=True must raise ValueError listing all violations."""
        config = {
            "ASP_HOLD_THRESHOLD": -0.1,  # below minimum 0.0
            "ASP_HOLD_AVERAGE": 2,  # exceeds maximum 1
        }
        with pytest.raises(
            (ValueError, ConfigError), match="ASP config validation failed"
        ):
            validate_asp_config(config, strict=True)

    def test_unknown_key_warns_but_not_a_violation(self):
        """An unrecognised key must emit UserWarning but not add a violation."""
        config = {"ASP_FUTURE_FLAG": 42}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            violations = validate_asp_config(config)
        assert violations == []
        assert any("ASP_FUTURE_FLAG" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# TestDumpAspConfig — §1.8C config dump (S126)
# ---------------------------------------------------------------------------


class TestDumpAspConfig:
    """dump_asp_config() serialises active ASP env-var state to a TOML file."""

    def test_creates_file(self, tmp_path, monkeypatch):
        out = tmp_path / "out.toml"
        monkeypatch.setenv("ASP_GRAPHCUT_SEAM", "1")
        result = dump_asp_config(str(out))
        assert out.exists()
        assert result == str(out.resolve())

    def test_set_env_var_appears_in_output(self, tmp_path, monkeypatch):
        out = tmp_path / "dump.toml"
        monkeypatch.setenv("ASP_FG_MAX_RESIDUAL", "35.0")
        dump_asp_config(str(out))
        content = out.read_text()
        assert "ASP_FG_MAX_RESIDUAL" in content
        assert "35.0" in content

    def test_unset_env_var_not_in_output_by_default(self, tmp_path, monkeypatch):
        out = tmp_path / "dump.toml"
        monkeypatch.delenv("ASP_HOLD_AVERAGE", raising=False)
        dump_asp_config(str(out))
        content = out.read_text()
        assert "ASP_HOLD_AVERAGE" not in content

    def test_include_defaults_includes_all_schema_keys(self, tmp_path, monkeypatch):
        out = tmp_path / "full.toml"
        for k in ["ASP_GRAPHCUT_SEAM", "ASP_FG_MAX_RESIDUAL"]:
            monkeypatch.delenv(k, raising=False)
        dump_asp_config(str(out), include_defaults=True)
        content = out.read_text()
        assert "ASP_GRAPHCUT_SEAM" in content
        assert "ASP_FG_MAX_RESIDUAL" in content

    def test_output_is_valid_toml(self, tmp_path, monkeypatch):
        """The dumped file must be parseable by tomllib."""
        import tomllib

        out = tmp_path / "valid.toml"
        monkeypatch.setenv("ASP_SEAM_SMOOTH_WINDOW", "5")
        dump_asp_config(str(out))
        with open(out, "rb") as fh:
            data = tomllib.load(fh)
        flat = {}
        for v in data.values():
            if isinstance(v, dict):
                flat.update(v)
        assert "ASP_SEAM_SMOOTH_WINDOW" in flat
        assert flat["ASP_SEAM_SMOOTH_WINDOW"] == 5


# ---------------------------------------------------------------------------
# §1.8D — Typed TOML schema comments in dump_asp_config (S131)
# ---------------------------------------------------------------------------


class TestDumpAspConfigSchemaComments:
    """§1.8D (S131): dump_asp_config emits type/range annotations as TOML comments."""

    def test_type_comment_present_for_float_key(self, tmp_path, monkeypatch):
        """Float keys must have a '# type: float' comment line in the output."""
        out = tmp_path / "typed.toml"
        monkeypatch.setenv("ASP_SEAM_NCC_GATE", "0.45")
        dump_asp_config(str(out))
        content = out.read_text()
        assert "# type: float" in content, (
            "Expected '# type: float' comment for float key"
        )

    def test_range_comment_present(self, tmp_path, monkeypatch):
        """Keys in _CONFIG_SCHEMA with non-None range must have '# … range: [' annotation."""
        out = tmp_path / "range.toml"
        monkeypatch.setenv("ASP_SEAM_NCC_GATE", "0.45")
        dump_asp_config(str(out))
        content = out.read_text()
        assert "range:" in content, "Expected 'range:' in schema comment"

    def test_type_annotation_precedes_key_line(self, tmp_path, monkeypatch):
        """The type comment must appear on the line immediately before the key = value line."""
        out = tmp_path / "order.toml"
        monkeypatch.setenv("ASP_DY_CV_MAX", "0.5")
        dump_asp_config(str(out))
        lines = out.read_text().splitlines()
        for idx, line in enumerate(lines):
            if line.startswith("ASP_DY_CV_MAX"):
                # Previous non-empty lines should include the type annotation
                prev_lines = [line for line in lines[:idx] if line.strip()]
                assert any("type:" in line for line in prev_lines[-3:]), (
                    "type: comment should appear within 3 lines before the key"
                )
                break

    def test_include_defaults_includes_type_comments(self, tmp_path, monkeypatch):
        """include_defaults=True also emits type/range comments for every schema key."""
        out = tmp_path / "defaults.toml"
        monkeypatch.delenv("ASP_SEAM_NCC_GATE", raising=False)
        dump_asp_config(str(out), include_defaults=True)
        content = out.read_text()
        assert "# type:" in content

    def test_int_key_annotated_as_int_type(self, tmp_path, monkeypatch):
        """Integer schema keys must emit '# type: int' not '# type: float'."""
        out = tmp_path / "int_type.toml"
        monkeypatch.setenv("ASP_DHASH_EXACT_DROP", "1")
        dump_asp_config(str(out))
        content = out.read_text()
        assert "# type: int" in content, "Expected '# type: int' for integer schema key"

"""Unit tests for Phase 1 config and infrastructure changes.

Covers:
- REQ-15: strategies/RSIBollingerStrategy.json declares api_mode='rest'
- REQ-6:  strategies/RSIBollingerStrategyV2.json exists with all required keys
- REQ-5:  requirements.txt pins lightstreamer-client-lib
"""

import json
from pathlib import Path

import pytest

# Project root is two levels above this test file (tests/ -> project root)
_PROJECT_ROOT = Path(__file__).parent.parent
_V1_JSON = _PROJECT_ROOT / "strategies" / "RSIBollingerStrategy.json"
_V2_JSON = _PROJECT_ROOT / "strategies" / "RSIBollingerStrategyV2.json"
_REQUIREMENTS = _PROJECT_ROOT / "requirements.txt"


# --------------------------------------------------------------------------- #
# REQ-15: V1 JSON declares api_mode=rest                                       #
# --------------------------------------------------------------------------- #


class TestV1JsonApiMode:
    """REQ-15: strategies/RSIBollingerStrategy.json must declare api_mode='rest'."""

    def test_v1_json_exists(self):
        """The V1 strategy JSON file must be present."""
        assert _V1_JSON.exists(), f"File not found: {_V1_JSON}"

    def test_v1_json_has_api_mode_key(self):
        """REQ-15: api_mode key must be present in V1 JSON."""
        data = json.loads(_V1_JSON.read_text(encoding="utf-8"))
        assert "api_mode" in data, "api_mode key missing from RSIBollingerStrategy.json"

    def test_v1_json_api_mode_is_rest(self):
        """REQ-15 scenario: api_mode value must be 'rest'."""
        data = json.loads(_V1_JSON.read_text(encoding="utf-8"))
        assert (
            data["api_mode"] == "rest"
        ), f"Expected api_mode='rest', got {data.get('api_mode')!r}"


# --------------------------------------------------------------------------- #
# REQ-6: V2 JSON exists with all required keys                                 #
# --------------------------------------------------------------------------- #

_V2_REQUIRED_KEYS = {
    "api_mode",
    "bb_period",
    "bb_std",
    "rsi_period",
    "rsi_oversold",
    "rsi_overbought",
    "max_long_positions",
    "max_short_positions",
    "contract_size",
    "min_dist_between_entries_ticks",
    "take_profit_ticks",
}


class TestV2Json:
    """REQ-6: strategies/RSIBollingerStrategyV2.json must exist and be complete."""

    def test_v2_json_exists(self):
        """REQ-6: The V2 strategy JSON file must be present."""
        assert _V2_JSON.exists(), f"File not found: {_V2_JSON}"

    def test_v2_json_is_valid(self):
        """REQ-6: The V2 JSON file must be valid JSON."""
        try:
            json.loads(_V2_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            pytest.fail(f"RSIBollingerStrategyV2.json is not valid JSON: {e}")

    def test_v2_json_has_all_required_keys(self):
        """REQ-6 scenario: all required keys must be present."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        missing = _V2_REQUIRED_KEYS - set(data.keys())
        assert not missing, f"Missing keys in RSIBollingerStrategyV2.json: {missing}"

    def test_v2_json_api_mode_is_streaming(self):
        """REQ-6: api_mode must be 'streaming' in V2 JSON."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert (
            data["api_mode"] == "streaming"
        ), f"Expected api_mode='streaming', got {data.get('api_mode')!r}"

    def test_v2_json_bb_period_is_20(self):
        """REQ-6: bb_period must be 20 per spec."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert data["bb_period"] == 20

    def test_v2_json_bb_std_is_2(self):
        """REQ-6: bb_std must be 2.0 per spec."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert data["bb_std"] == 2.0

    def test_v2_json_rsi_period_is_14(self):
        """REQ-6: rsi_period must be 14 per spec."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert data["rsi_period"] == 14

    def test_v2_json_rsi_oversold_is_30(self):
        """REQ-6: rsi_oversold must be 30 per spec."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert data["rsi_oversold"] == 30

    def test_v2_json_rsi_overbought_is_70(self):
        """REQ-6: rsi_overbought must be 70 per spec."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert data["rsi_overbought"] == 70


# --------------------------------------------------------------------------- #
# REQ-5: requirements.txt pins lightstreamer-client-lib                        #
# --------------------------------------------------------------------------- #


class TestRequirementsPin:
    """REQ-5: lightstreamer-client-lib must be explicitly pinned in requirements.txt."""

    def test_requirements_exists(self):
        """requirements.txt must exist at the project root."""
        assert _REQUIREMENTS.exists(), f"File not found: {_REQUIREMENTS}"

    def test_lightstreamer_is_pinned(self):
        """REQ-5 scenario: requirements.txt must contain an explicit version pin."""
        content = _REQUIREMENTS.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines()]
        pinned = [
            line for line in lines if line.startswith("lightstreamer-client-lib==")
        ]
        assert pinned, (
            "lightstreamer-client-lib is not explicitly pinned in requirements.txt. "
            "Add a line like: lightstreamer-client-lib==1.0.3"
        )

    def test_lightstreamer_pin_has_version(self):
        """The pin must include a specific version number (not just the package name)."""
        content = _REQUIREMENTS.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines()]
        for line in lines:
            if line.startswith("lightstreamer-client-lib=="):
                version_part = line.split("==", 1)[1].strip()
                assert version_part, "Version must not be empty after '=='"
                assert version_part[
                    0
                ].isdigit(), f"Version must start with a digit, got: {version_part!r}"
                return
        pytest.fail("lightstreamer-client-lib pin not found in requirements.txt")

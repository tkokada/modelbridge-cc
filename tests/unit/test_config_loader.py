"""Unit tests for configuration loading utilities."""

from pathlib import Path
import tempfile

import pytest

from modelbridge.types import ConfigDict, ValidationRules
from modelbridge.utils.config_loader import (
    create_param_config,
    load_toml_config,
    validate_config,
)


class TestLoadTomlConfig:
    """Test cases for load_toml_config function."""

    def test_load_valid_config(self, temp_config_file: Path) -> None:
        """Test loading valid TOML configuration."""
        config = load_toml_config(temp_config_file)

        assert isinstance(config, dict)
        assert "generic" in config
        assert "dataset" in config
        assert config["generic"]["seed"] == 42

    def test_load_nonexistent_file(self) -> None:
        """Test loading non-existent file raises FileNotFoundError."""
        nonexistent_path = Path("/nonexistent/config.toml")

        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_toml_config(nonexistent_path)

    def test_load_invalid_toml(self) -> None:
        """Test loading invalid TOML raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("invalid toml content [[[")
            invalid_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid TOML configuration file"):
                load_toml_config(invalid_path)
        finally:
            invalid_path.unlink()


class TestValidateConfig:
    """Test cases for validate_config function."""

    def test_validate_valid_config(self) -> None:
        """Test validation of valid configuration."""
        config: ConfigDict = {
            "string_key": "value",
            "int_key": 42,
            "float_key": 3.14,
        }

        required_keys: ValidationRules = {
            "string_key": str,
            "int_key": int,
            "float_key": float,
        }

        # Should not raise any exception
        validate_config(config, required_keys)

    def test_validate_missing_key(self) -> None:
        """Test validation with missing required key."""
        config: ConfigDict = {"key1": "value1"}
        required_keys: ValidationRules = {
            "key1": str,
            "key2": int,  # Missing in config
        }

        with pytest.raises(
            ValueError, match="Required configuration key missing: key2"
        ):
            validate_config(config, required_keys)

    def test_validate_wrong_type(self) -> None:
        """Test validation with wrong type."""
        config: ConfigDict = {"key1": "string_value"}
        required_keys: ValidationRules = {"key1": int}  # Expecting int, got str

        with pytest.raises(ValueError, match="must be of type int"):
            validate_config(config, required_keys)


class TestCreateParamConfig:
    """Test cases for create_param_config function."""

    def test_create_valid_config(self) -> None:
        """Test creation of valid parameter configuration."""
        param_names = ["x1", "x2", "p1"]
        param_types = ["float", "float", "int"]
        param_ranges = [(-1.0, 1.0), (0.0, 2.0), (1, 10)]

        config = create_param_config(param_names, param_types, param_ranges)

        assert isinstance(config, dict)
        assert len(config) == 3

        # Check structure
        assert config["x1"]["type"] == "float"
        assert config["x1"]["low"] == -1.0
        assert config["x1"]["high"] == 1.0

        assert config["p1"]["type"] == "int"
        assert config["p1"]["low"] == 1.0
        assert config["p1"]["high"] == 10.0

    def test_create_config_mismatched_lengths(self) -> None:
        """Test creation with mismatched list lengths."""
        param_names = ["x1", "x2"]
        param_types = ["float"]  # Wrong length
        param_ranges = [(-1.0, 1.0), (0.0, 2.0)]

        with pytest.raises(
            ValueError, match="Parameter specification lists must have equal lengths"
        ):
            create_param_config(param_names, param_types, param_ranges)

    def test_create_config_invalid_type(self) -> None:
        """Test creation with invalid parameter type."""
        param_names = ["x1"]
        param_types = ["invalid"]
        param_ranges = [(0.0, 1.0)]

        with pytest.raises(ValueError, match="Invalid parameter type: invalid"):
            create_param_config(param_names, param_types, param_ranges)

    def test_create_config_empty_lists(self) -> None:
        """Test creation with empty lists."""
        config = create_param_config([], [], [])
        assert config == {}

    def test_create_config_single_parameter(self) -> None:
        """Test creation with single parameter."""
        config = create_param_config(["param1"], ["float"], [(0.0, 1.0)])

        assert len(config) == 1
        assert "param1" in config
        assert config["param1"]["type"] == "float"

"""Configuration loading utilities."""

from pathlib import Path

import toml

from ..types import (
    ConfigDict,
    FilePath,
    ParamConfig,
    ValidationRules,
)


def load_toml_config(config_path: FilePath) -> ConfigDict:
    """Load configuration from TOML file.

    Args:
        config_path: Path to the TOML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config: ConfigDict = toml.load(f)
        return config
    except Exception as e:
        raise ValueError(f"Invalid TOML configuration file: {e}") from e


def validate_config(config: ConfigDict, required_keys: ValidationRules) -> None:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate
        required_keys: Dictionary of required keys and their expected types

    Raises:
        ValueError: If required keys are missing or have wrong type
    """
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"Required configuration key missing: {key}")

        if not isinstance(config[key], expected_type):
            raise ValueError(
                f"Configuration key '{key}' must be of type {expected_type.__name__}, "
                f"got {type(config[key]).__name__}"
            )


def create_param_config(
    param_names: list[str],
    param_types: list[str],
    param_ranges: list[tuple[float, float]],
) -> ParamConfig:
    """Create parameter configuration dictionary.

    Args:
        param_names: List of parameter names
        param_types: List of parameter types ("float" or "int")
        param_ranges: List of (min, max) tuples for each parameter

    Returns:
        Parameter configuration dictionary
    """
    lengths = [len(param_names), len(param_types), len(param_ranges)]
    if not all(length == lengths[0] for length in lengths):
        # Python 3.12: Enhanced error messages with better formatting
        raise ValueError(
            f"Parameter specification lists must have equal lengths:\n"
            f"  param_names: {len(param_names)}\n"
            f"  param_types: {len(param_types)}\n"
            f"  param_ranges: {len(param_ranges)}"
        )

    from ..types import ParamConfigValue

    config: ParamConfig = {}
    for name, param_type, (low, high) in zip(
        param_names, param_types, param_ranges, strict=False
    ):
        if param_type not in ("float", "int"):
            raise ValueError(f"Invalid parameter type: {param_type}")
        config[name] = ParamConfigValue(type=param_type, low=low, high=high)  # type: ignore[typeddict-item]

    return config

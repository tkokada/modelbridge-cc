"""Data management utilities for model bridging."""

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..types import (
    FilePath,
    NumPyArray,
    ParamDict,
    ParamList,
    ScalingConfig,
)


class DataManager:
    """Utility class for managing parameter data and file I/O."""

    def convert_params_to_array(
        self, params_list: ParamList, param_names: list[str]
    ) -> NumPyArray:
        """Convert list of parameter dictionaries to numpy array.

        Args:
            params_list: List of parameter dictionaries
            param_names: Ordered list of parameter names

        Returns:
            Numpy array with shape (n_samples, n_params)
        """
        return np.array(
            [[param_dict[name] for name in param_names] for param_dict in params_list]
        )

    def convert_array_to_params(
        self, params_array: NumPyArray, param_names: list[str]
    ) -> ParamList:
        """Convert numpy array to list of parameter dictionaries.

        Args:
            params_array: Numpy array with shape (n_samples, n_params)
            param_names: List of parameter names

        Returns:
            List of parameter dictionaries
        """
        return [
            {name: float(value) for name, value in zip(param_names, row, strict=False)}
            for row in params_array
        ]

    def save_params_csv(
        self,
        params_list: ParamList,
        file_path: FilePath,
        param_names: list[str] | None = None,
    ) -> None:
        """Save parameter list to CSV file.

        Args:
            params_list: List of parameter dictionaries
            file_path: Path to save CSV file
            param_names: Parameter names (if None, uses keys from first dict)
        """
        if not params_list:
            raise ValueError("Empty parameter list")

        if param_names is None:
            param_names = list(params_list[0].keys())

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=param_names)
            writer.writeheader()
            writer.writerows(params_list)

    def save_array_csv(
        self, array: NumPyArray, file_path: FilePath, column_names: list[str]
    ) -> None:
        """Save numpy array to CSV file.

        Args:
            array: Numpy array to save
            file_path: Path to save CSV file
            column_names: Column names for CSV
        """
        if array.shape[1] != len(column_names):
            # Python 3.12: Enhanced f-string with debugging info
            raise ValueError(
                f"Array columns ({array.shape[1]}) must match "
                f"column names count ({len(column_names)})"
            )

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(array, columns=column_names)
        df.to_csv(file_path, index=False)

    def load_params_csv(self, file_path: FilePath) -> ParamList:
        """Load parameter list from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            List of parameter dictionaries
        """
        df = pd.read_csv(Path(file_path))
        return [dict(row) for row in df.to_dict("records")]

    def load_array_csv(self, file_path: FilePath) -> NumPyArray:
        """Load numpy array from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            Numpy array
        """
        df = pd.read_csv(Path(file_path))
        return np.asarray(df.values, dtype=np.float64)

    def create_dataset_split(
        self, total_datasets: int, train_ratio: float = 0.7
    ) -> tuple[int, int]:
        """Create train/test split for number of datasets.

        Args:
            total_datasets: Total number of datasets
            train_ratio: Ratio of datasets for training

        Returns:
            Tuple of (n_train, n_test)
        """
        n_train = int(total_datasets * train_ratio)
        n_test = total_datasets - n_train
        return n_train, n_test

    def scale_parameters(
        self, params: ParamDict | ParamList, scaling_config: ScalingConfig
    ) -> Any:  # Return type depends on input type
        """Apply scaling to parameters.

        Args:
            params: Parameter dictionary or list of dictionaries
            scaling_config: Scaling configuration
                Format: {param_name: {"scale": scale_factor, "offset": offset_value}}
                Scaled value = original_value * scale + offset

        Returns:
            Scaled parameters
        """
        is_single = isinstance(params, dict)
        param_list = [params] if is_single else params

        scaled_params = []
        for param_dict in param_list:
            scaled_dict = {}
            for param_name, value in param_dict.items():
                if param_name in scaling_config:
                    config = scaling_config[param_name]
                    scaled_value = value * config.get("scale", 1.0) + config.get(
                        "offset", 0.0
                    )
                    scaled_dict[param_name] = scaled_value
                else:
                    scaled_dict[param_name] = value
            scaled_params.append(scaled_dict)

        return scaled_params[0] if is_single else scaled_params

    def generate_variable_dataset(
        self,
        dim: int,
        max_value: float,
        min_value: float,
        num_samples: int,
        sampler: str = "uniform",
        seed: int | None = None,
    ) -> NumPyArray:
        """Generate random variable dataset.

        Args:
            dim: Number of dimensions
            max_value: Maximum value
            min_value: Minimum value
            num_samples: Number of samples to generate
            sampler: Sampling method ("uniform")
            seed: Random seed

        Returns:
            Generated dataset with shape (num_samples, dim)
        """
        if seed is not None:
            np.random.seed(seed)

        if sampler == "uniform":
            return np.random.uniform(min_value, max_value, size=(num_samples, dim))
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

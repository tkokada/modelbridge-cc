"""Unit tests for DataManager."""

from pathlib import Path

import numpy as np
import pytest

from modelbridge.types import FloatArray, ParamDict, ParamList, ScalingConfig
from modelbridge.utils.data_manager import DataManager


class TestDataManager:
    """Test cases for DataManager class."""

    def test_init(self) -> None:
        """Test initialization."""
        dm = DataManager()
        assert dm is not None

    def test_convert_params_to_array(self, sample_param_list: ParamList) -> None:
        """Test conversion from parameter list to numpy array."""
        dm = DataManager()
        param_names = ["x_1", "x_2", "p_1", "p_2"]

        array = dm.convert_params_to_array(sample_param_list, param_names)

        assert array.shape == (3, 4)
        assert isinstance(array, np.ndarray)

        # Check first row values
        expected_first_row = [1.0, 2.0, 0.5, 1.5]
        np.testing.assert_array_almost_equal(array[0], expected_first_row)

    def test_convert_array_to_params(self, sample_float_array: FloatArray) -> None:
        """Test conversion from numpy array to parameter list."""
        dm = DataManager()
        param_names = ["x_1", "x_2", "p_1", "p_2"]

        param_list = dm.convert_array_to_params(sample_float_array, param_names)

        assert len(param_list) == 3
        assert all(isinstance(params, dict) for params in param_list)
        assert all(set(params.keys()) == set(param_names) for params in param_list)

        # Check first parameter dict
        first_params = param_list[0]
        assert first_params["x_1"] == 1.0
        assert first_params["x_2"] == 2.0

    def test_save_and_load_params_csv(
        self, sample_param_list: ParamList, temp_directory: Path
    ) -> None:
        """Test saving and loading parameter CSV files."""
        dm = DataManager()
        param_names = ["x_1", "x_2", "p_1", "p_2"]

        # Save parameters
        csv_file = temp_directory / "test_params.csv"
        dm.save_params_csv(sample_param_list, csv_file, param_names)

        assert csv_file.exists()

        # Load parameters
        loaded_params = dm.load_params_csv(csv_file)

        assert len(loaded_params) == len(sample_param_list)
        assert all(isinstance(params, dict) for params in loaded_params)

        # Check data integrity
        for original, loaded in zip(sample_param_list, loaded_params, strict=False):
            for key in param_names:
                assert abs(float(original[key]) - float(loaded[key])) < 1e-10

    def test_save_params_csv_empty_list(self, temp_directory: Path) -> None:
        """Test saving empty parameter list raises error."""
        dm = DataManager()
        csv_file = temp_directory / "empty.csv"

        with pytest.raises(ValueError, match="Empty parameter list"):
            dm.save_params_csv([], csv_file)

    def test_save_and_load_array_csv(
        self, sample_float_array: FloatArray, temp_directory: Path
    ) -> None:
        """Test saving and loading numpy arrays as CSV."""
        dm = DataManager()
        column_names = ["col1", "col2", "col3", "col4"]

        # Save array
        csv_file = temp_directory / "test_array.csv"
        dm.save_array_csv(sample_float_array, csv_file, column_names)

        assert csv_file.exists()

        # Load array
        loaded_array = dm.load_array_csv(csv_file)

        assert loaded_array.shape == sample_float_array.shape
        np.testing.assert_array_almost_equal(loaded_array, sample_float_array)

    def test_save_array_csv_wrong_columns(
        self, sample_float_array: FloatArray, temp_directory: Path
    ) -> None:
        """Test saving array with wrong number of column names."""
        dm = DataManager()
        csv_file = temp_directory / "wrong_cols.csv"

        # Array has 4 columns, but only provide 2 column names
        with pytest.raises(ValueError, match="Array columns .* must match"):
            dm.save_array_csv(sample_float_array, csv_file, ["col1", "col2"])

    def test_create_dataset_split(self) -> None:
        """Test dataset splitting."""
        dm = DataManager()

        n_train, n_test = dm.create_dataset_split(10, train_ratio=0.7)
        assert n_train == 7
        assert n_test == 3

        # Test edge cases
        n_train, n_test = dm.create_dataset_split(10, train_ratio=0.0)
        assert n_train == 0
        assert n_test == 10

        n_train, n_test = dm.create_dataset_split(10, train_ratio=1.0)
        assert n_train == 10
        assert n_test == 0

    def test_scale_parameters_single_dict(self, sample_param_dict: ParamDict) -> None:
        """Test parameter scaling with single dictionary."""
        dm = DataManager()

        scaling_config: ScalingConfig = {
            "x_1": {"scale": 2.0, "offset": 1.0},
            "p_1": {"scale": 0.5, "offset": 0.0},
        }

        scaled = dm.scale_parameters(sample_param_dict, scaling_config)

        assert isinstance(scaled, dict)
        # x_1: 1.5 -> 1.5 * 2.0 + 1.0 = 4.0
        assert scaled["x_1"] == 4.0
        # p_1: 1.0 -> 1.0 * 0.5 + 0.0 = 0.5
        assert scaled["p_1"] == 0.5
        # Unchanged parameters
        assert scaled["x_2"] == sample_param_dict["x_2"]
        assert scaled["p_2"] == sample_param_dict["p_2"]

    def test_scale_parameters_list(self, sample_param_list: ParamList) -> None:
        """Test parameter scaling with parameter list."""
        dm = DataManager()

        scaling_config: ScalingConfig = {
            "x_1": {"scale": 2.0, "offset": 0.0},
        }

        scaled = dm.scale_parameters(sample_param_list, scaling_config)

        assert isinstance(scaled, list)
        assert len(scaled) == len(sample_param_list)

        # Check first element scaling
        assert scaled[0]["x_1"] == 2.0  # 1.0 * 2.0 + 0.0

    def test_generate_variable_dataset(self) -> None:
        """Test variable dataset generation."""
        dm = DataManager()

        dataset = dm.generate_variable_dataset(
            dim=3,
            max_value=5.0,
            min_value=-5.0,
            num_samples=10,
            sampler="uniform",
            seed=42,
        )

        assert dataset.shape == (10, 3)
        assert isinstance(dataset, np.ndarray)
        assert dataset.dtype == np.float64

        # Check bounds
        assert np.all(dataset >= -5.0)
        assert np.all(dataset <= 5.0)

    def test_generate_variable_dataset_reproducible(self) -> None:
        """Test that dataset generation is reproducible with same seed."""
        dm = DataManager()

        dataset1 = dm.generate_variable_dataset(
            dim=2, max_value=1.0, min_value=-1.0, num_samples=5, seed=123
        )
        dataset2 = dm.generate_variable_dataset(
            dim=2, max_value=1.0, min_value=-1.0, num_samples=5, seed=123
        )

        np.testing.assert_array_equal(dataset1, dataset2)

    def test_generate_variable_dataset_invalid_sampler(self) -> None:
        """Test invalid sampler raises error."""
        dm = DataManager()

        with pytest.raises(ValueError, match="Unknown sampler: invalid"):
            dm.generate_variable_dataset(
                dim=2, max_value=1.0, min_value=-1.0, num_samples=5, sampler="invalid"
            )

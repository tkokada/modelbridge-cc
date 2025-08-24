"""Property-based tests using Hypothesis for ModelBridge components."""


from hypothesis import assume, given, settings
from hypothesis import strategies as st
import numpy as np

from modelbridge.core.regression import LinearRegressionModel, PolynomialRegressionModel
from modelbridge.utils.config_loader import create_param_config
from modelbridge.utils.data_manager import DataManager


class TestDataManagerProperties:
    """Property-based tests for DataManager."""

    @given(
        dim=st.integers(min_value=1, max_value=10),
        num_samples=st.integers(min_value=1, max_value=100),
        min_val=st.floats(min_value=-10.0, max_value=-0.1),
        max_val=st.floats(min_value=0.1, max_value=10.0),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=10)
    def test_generate_variable_dataset_properties(
        self,
        dim: int,
        num_samples: int,
        min_val: float,
        max_val: float,
        seed: int,
    ) -> None:
        """Test properties of generated variable datasets."""
        assume(min_val < max_val)

        dm = DataManager()

        dataset = dm.generate_variable_dataset(
            dim=dim,
            max_value=max_val,
            min_value=min_val,
            num_samples=num_samples,
            seed=seed,
        )

        # Property: correct shape
        assert dataset.shape == (num_samples, dim)

        # Property: values within bounds
        assert np.all(dataset >= min_val)
        assert np.all(dataset <= max_val)

        # Property: reproducibility with same seed
        dataset2 = dm.generate_variable_dataset(
            dim=dim,
            max_value=max_val,
            min_value=min_val,
            num_samples=num_samples,
            seed=seed,
        )
        np.testing.assert_array_equal(dataset, dataset2)

    @given(
        total_datasets=st.integers(min_value=1, max_value=100),
        train_ratio=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_dataset_split_properties(
        self,
        total_datasets: int,
        train_ratio: float,
    ) -> None:
        """Test properties of dataset splitting."""
        dm = DataManager()

        n_train, n_test = dm.create_dataset_split(total_datasets, train_ratio)

        # Property: sum equals total
        assert n_train + n_test == total_datasets

        # Property: non-negative values
        assert n_train >= 0
        assert n_test >= 0

        # Property: training size is approximately correct
        expected_train = int(total_datasets * train_ratio)
        assert n_train == expected_train

    @given(
        n_samples=st.integers(min_value=1, max_value=10),
        n_params=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=5)
    def test_array_param_conversion_roundtrip(self, n_samples: int, n_params: int) -> None:
        """Test roundtrip conversion between arrays and parameter dictionaries."""
        dm = DataManager()

        # Generate homogeneous array data
        array = np.random.randn(n_samples, n_params).astype(np.float64)
        param_names = [f"param_{i}" for i in range(n_params)]

        # Convert to parameter list
        param_list = dm.convert_array_to_params(array, param_names)

        # Convert back to array
        array_back = dm.convert_params_to_array(param_list, param_names)

        # Property: roundtrip should preserve data
        np.testing.assert_array_almost_equal(array, array_back)


class TestRegressionModelProperties:
    """Property-based tests for regression models."""

    @given(
        n_samples=st.integers(min_value=5, max_value=50),
        n_features=st.integers(min_value=1, max_value=5),
        n_targets=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=5)
    def test_linear_regression_fit_predict_shape(
        self,
        n_samples: int,
        n_features: int,
        n_targets: int,
    ) -> None:
        """Test that linear regression maintains correct shapes."""
        model = LinearRegressionModel()

        # Generate random data
        X = np.random.randn(n_samples, n_features).astype(np.float64)
        y = np.random.randn(n_samples, n_targets).astype(np.float64)

        # Fit and predict
        model.fit(X, y)
        predictions = model.predict(X)

        # Property: prediction shape matches target shape
        assert predictions.shape == y.shape

    @given(
        degree=st.integers(min_value=1, max_value=3),
        n_samples=st.integers(min_value=10, max_value=30),
        n_features=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=5)
    def test_polynomial_regression_degrees(
        self,
        degree: int,
        n_samples: int,
        n_features: int,
    ) -> None:
        """Test polynomial regression with different degrees."""
        model = PolynomialRegressionModel(degree=degree)

        X = np.random.randn(n_samples, n_features).astype(np.float64)
        y = np.random.randn(n_samples, 1).astype(np.float64)

        model.fit(X, y)
        predictions = model.predict(X)

        # Property: can fit and predict without errors
        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))


class TestConfigLoaderProperties:
    """Property-based tests for configuration loader."""

    @given(
        n_params=st.integers(min_value=1, max_value=10),
        param_type=st.sampled_from(["float", "int"]),
    )
    def test_param_config_creation_properties(
        self,
        n_params: int,
        param_type: str,
    ) -> None:
        """Test properties of parameter configuration creation."""
        param_names = [f"param_{i}" for i in range(n_params)]
        param_types = [param_type] * n_params

        if param_type == "float":
            param_ranges = [(0.0, 1.0)] * n_params
        else:
            param_ranges = [(0.0, 10.0)] * n_params

        config = create_param_config(param_names, param_types, param_ranges)

        # Property: correct number of parameters
        assert len(config) == n_params

        # Property: all parameters have required keys
        for param_name in param_names:
            assert param_name in config
            assert "type" in config[param_name]
            assert "low" in config[param_name]
            assert "high" in config[param_name]
            assert config[param_name]["type"] == param_type

    @given(
        scale_factor=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
        offset=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        param_value=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    )
    def test_parameter_scaling_properties(
        self,
        scale_factor: float,
        offset: float,
        param_value: float,
    ) -> None:
        """Test properties of parameter scaling."""
        dm = DataManager()

        params = {"test_param": param_value}
        scaling_config = {
            "test_param": {"scale": scale_factor, "offset": offset}
        }

        scaled = dm.scale_parameters(params, scaling_config)

        # Property: scaling formula is correct
        expected_value = param_value * scale_factor + offset
        assert abs(scaled["test_param"] - expected_value) < 1e-10

        # Property: unscaled parameters remain unchanged
        params_with_extra = {"test_param": param_value, "other_param": 42.0}
        scaled_with_extra = dm.scale_parameters(params_with_extra, scaling_config)
        assert scaled_with_extra["other_param"] == 42.0

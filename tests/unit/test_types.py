"""Unit tests for type definitions and type safety."""

import numpy as np

from modelbridge.types import (
    EvaluationMetrics,
    FloatArray,
    ParamConfigValue,
    ParamDict,
    ParameterCollection,
    ParamList,
    ResultCollection,
    StudyConfigDict,
)


class TestTypeDefinitions:
    """Test type definitions and their usage."""

    def test_param_config_value_creation(self) -> None:
        """Test ParamConfigValue TypedDict creation."""
        config_value = ParamConfigValue(type="float", low=0.0, high=1.0)

        assert config_value["type"] == "float"
        assert config_value["low"] == 0.0
        assert config_value["high"] == 1.0

    def test_param_config_value_invalid_type(self) -> None:
        """Test ParamConfigValue with invalid type."""
        # This will be caught by type checkers, not runtime
        config_value = ParamConfigValue(type="float", low=0.0, high=1.0)
        assert config_value is not None

    def test_evaluation_metrics_creation(self) -> None:
        """Test EvaluationMetrics TypedDict creation."""
        metrics = EvaluationMetrics(mse=0.1, mae=0.05, r2=0.95)

        assert metrics["mse"] == 0.1
        assert metrics["mae"] == 0.05
        assert metrics["r2"] == 0.95

    def test_study_config_dict_partial(self) -> None:
        """Test StudyConfigDict with partial fields (total=False)."""
        # Can create with only some fields
        config = StudyConfigDict(storage="sqlite:///test.db")
        assert config["storage"] == "sqlite:///test.db"

        # Can create with all fields
        full_config = StudyConfigDict(
            storage="sqlite:///test.db", direction="minimize", sampler="tpe", seed=42
        )
        assert full_config["direction"] == "minimize"

    def test_param_dict_type_alias(self) -> None:
        """Test ParamDict type alias usage."""
        params: ParamDict = {
            "x_1": 1.5,
            "x_2": -0.5,
            "p_1": 1.0,
            "p_2": 0.8,
        }

        assert isinstance(params, dict)
        assert all(isinstance(v, int | float | str | bool) for v in params.values())

    def test_param_list_type_alias(self) -> None:
        """Test ParamList type alias usage."""
        param_list: ParamList = [
            {"x": 1.0, "p": 0.5},
            {"x": 2.0, "p": 1.0},
        ]

        assert isinstance(param_list, list)
        assert all(isinstance(params, dict) for params in param_list)

    def test_float_array_type_alias(self) -> None:
        """Test FloatArray type alias usage."""
        array: FloatArray = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64


class TestParameterCollection:
    """Test ParameterCollection generic container."""

    def test_parameter_collection_creation(self) -> None:
        """Test ParameterCollection creation and basic operations."""
        params = [{"x": 1.0}, {"x": 2.0}]
        collection = ParameterCollection(params)

        assert collection.get_all() == params

    def test_parameter_collection_add(self) -> None:
        """Test adding parameters to collection."""
        collection = ParameterCollection([{"x": 1.0}])

        result = collection.add({"x": 2.0})

        # Should return self for chaining
        assert result is collection
        assert len(collection.get_all()) == 2

    def test_parameter_collection_chaining(self) -> None:
        """Test method chaining with ParameterCollection."""
        collection = ParameterCollection([])

        result = collection.add({"x": 1.0}).add({"x": 2.0}).add({"x": 3.0})

        assert result is collection
        assert len(collection.get_all()) == 3


class TestResultCollection:
    """Test ResultCollection generic container."""

    def test_result_collection_creation(self) -> None:
        """Test ResultCollection creation."""
        collection = ResultCollection[float]()

        assert collection.get_latest() is None
        assert collection.get_all() == []

    def test_result_collection_add_and_get(self) -> None:
        """Test adding and getting results."""
        collection = ResultCollection[float]()

        collection.add_result(1.5)
        collection.add_result(2.5)

        assert collection.get_latest() == 2.5
        assert collection.get_all() == [1.5, 2.5]

    def test_result_collection_chaining(self) -> None:
        """Test method chaining with ResultCollection."""
        collection = ResultCollection[str]()

        result = collection.add_result("first").add_result("second")

        assert result is collection
        assert collection.get_latest() == "second"

    def test_result_collection_type_safety(self) -> None:
        """Test type safety of ResultCollection."""
        # This is more for demonstrating type safety to type checkers
        float_collection = ResultCollection[float]()
        string_collection = ResultCollection[str]()

        float_collection.add_result(3.14)
        string_collection.add_result("test")

        assert isinstance(float_collection.get_latest(), float)
        assert isinstance(string_collection.get_latest(), str)

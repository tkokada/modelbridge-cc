"""Optuna-based hyperparameter optimization module."""

from collections.abc import Callable

import optuna
from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler, TPESampler

from ..types import (
    OptimizationDirection,
    OptimizationStudy,
    OptimizerSamplerType,
    ParamConfig,
    ParamDict,
)


class OptunaOptimizer:
    """Wrapper for Optuna hyperparameter optimization."""

    def __init__(
        self,
        storage: str | None = None,
        direction: OptimizationDirection = "minimize",
        sampler: OptimizerSamplerType | BaseSampler | None = None,
        seed: int | None = None,
    ):
        """Initialize Optuna optimizer wrapper.

        Sets up an Optuna optimization study with configurable storage, direction,
        sampler algorithm, and random seed for reproducible results.

        Args:
            storage (str | None, optional): Storage URI for Optuna study persistence.
                Can be SQLite path like "sqlite:///study.db" or None for in-memory.
                Defaults to None.
            direction (OptimizationDirection, optional): Optimization direction, either
                "minimize" or "maximize". Defaults to "minimize".
            sampler (OptimizerSamplerType | BaseSampler | None, optional): Optimization
                sampler algorithm. Can be string name ("random", "tpe", "cmaes") or
                Optuna sampler instance. Defaults to None (uses TPE).
            seed (int | None, optional): Random seed for reproducible optimization
                results. If None, uses random initialization. Defaults to None.

        """
        self.storage = storage or "sqlite:///optuna.db"
        self.direction = direction
        self.seed = seed

        if isinstance(sampler, str):
            self.sampler = self._create_sampler(sampler, seed)  # type: ignore[arg-type]
        elif sampler is None:
            self.sampler = self._create_sampler("tpe", seed)
        else:
            self.sampler = sampler

    def _create_sampler(
        self, sampler_name: OptimizerSamplerType, seed: int | None
    ) -> BaseSampler:
        """Create sampler from string name."""
        # Python 3.12: Using match-case for cleaner control flow
        match sampler_name:
            case "random":
                return RandomSampler(seed=seed)
            case "tpe":
                return TPESampler(seed=seed)
            case "cmaes":
                return CmaEsSampler(seed=seed)
            case _:
                supported = ["random", "tpe", "cmaes"]
                raise ValueError(
                    f"Unknown sampler '{sampler_name}'. "
                    f"Supported samplers: {', '.join(supported)}"
                )

    def create_or_load_study(
        self, study_name: str, load_if_exists: bool = True
    ) -> OptimizationStudy:
        """Create or load an Optuna study.

        Args:
            study_name: Name of the study
            load_if_exists: Whether to load existing study

        Returns:
            Tuple of (study, is_existing)

        """
        try:
            if load_if_exists:
                study = optuna.load_study(study_name=study_name, storage=self.storage)
                return study, True
        except KeyError:
            pass

        study = optuna.create_study(
            sampler=self.sampler,
            study_name=study_name,
            storage=self.storage,
            direction=self.direction,
            load_if_exists=False,
        )
        return study, False

    def optimize_batch(
        self,
        objective_func: Callable[[optuna.Trial], float],
        study_name: str,
        n_trials: int,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        """Run batch optimization.

        Args:
            objective_func: Objective function to optimize
            study_name: Name of the study
            n_trials: Number of trials to run
            load_if_exists: Whether to load existing study

        Returns:
            Completed study

        """
        study, is_existing = self.create_or_load_study(study_name, load_if_exists)

        if not is_existing or len(study.trials) < n_trials:
            remaining_trials = n_trials - len(study.trials)
            study.optimize(objective_func, n_trials=remaining_trials)

        return study

    def suggest_parameters(
        self, trial: optuna.Trial, param_config: ParamConfig
    ) -> ParamDict:
        """Suggest parameters based on configuration.

        Args:
            trial: Optuna trial object
            param_config: Parameter configuration dict
                Format: {param_name: {type: "float/int", low: min_val, high: max_val}}

        Returns:
            Dictionary of suggested parameters

        """
        params: ParamDict = {}
        for param_name, config in param_config.items():
            param_type = config["type"]
            # Python 3.12: Match-case for parameter types
            match param_type:
                case "float":
                    params[param_name] = trial.suggest_float(
                        param_name, config["low"], config["high"]
                    )
                case "int":
                    params[param_name] = trial.suggest_int(
                        param_name, int(config["low"]), int(config["high"])
                    )
                case _:
                    supported_types = ["float", "int"]
                    raise ValueError(
                        f"Unknown parameter type '{param_type}' for '{param_name}'. "
                        f"Supported types: {', '.join(supported_types)}"
                    )

        return params

    def get_best_params(self, study_name: str) -> ParamDict:
        """Get best parameters from completed study.

        Args:
            study_name: Name of the study

        Returns:
            Best parameters dictionary

        """
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        return study.best_trial.params  # type: ignore[no-any-return]

    def get_all_completed_params(self, study_name: str) -> list[ParamDict]:
        """Get all completed trial parameters.

        Args:
            study_name: Name of the study

        Returns:
            List of parameter dictionaries

        """
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        return [
            trial.params
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

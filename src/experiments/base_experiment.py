"""Base class for reusable experiment infrastructure."""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.config import get_config
from src.llm_client import LLMClient


class BaseExperiment(ABC):
    """
    Abstract base class for experiments.

    Provides common infrastructure for:
    - Client initialization
    - Output directory management
    - Progress tracking
    - Results saving
    - Experiment orchestration
    """

    def __init__(self, output_dir: str = "data/raw", timestamp: str = None):
        """
        Initialize the experiment.

        Args:
            output_dir: Directory to save results
            timestamp: Timestamp string (YYYYMMDD_HHMMSS format) or ISO format
        """
        self.config = get_config()
        self.client = LLMClient(self.config)

        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup timestamps
        if timestamp:
            # Try to parse as file format first
            try:
                timestamp_dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                self.timestamp_file = timestamp
                self.timestamp = timestamp_dt.isoformat()
            except ValueError:
                # Assume it's already ISO format
                self.timestamp = timestamp
                self.timestamp_file = datetime.fromisoformat(timestamp).strftime("%Y%m%d_%H%M%S")
        else:
            now = datetime.now()
            self.timestamp = now.isoformat()
            self.timestamp_file = now.strftime("%Y%m%d_%H%M%S")

    @abstractmethod
    def get_experiment_name(self) -> str:
        """
        Return the experiment name for file naming.

        Example: "exp1" or "exp2_collective"
        """
        pass

    @abstractmethod
    def get_tested_models(self) -> List[str]:
        """
        Return list of models to test in this experiment.

        Returns:
            List of model identifiers
        """
        pass

    @abstractmethod
    def get_scenarios(self) -> List[Dict[str, Any]]:
        """
        Return list of scenarios/test cases for this experiment.

        Each scenario should be a dictionary containing the parameters
        needed for run_single_test.

        Returns:
            List of scenario dictionaries
        """
        pass

    @abstractmethod
    def run_single_test(
        self,
        tested_model: str,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single test case.

        Args:
            tested_model: The model being tested
            scenario: Dictionary containing scenario parameters

        Returns:
            Dictionary with test results
        """
        pass

    def get_scenario_description(self, scenario: Dict[str, Any]) -> str:
        """
        Get a human-readable description of a scenario for progress display.

        Default implementation returns first 50 chars of scenario dict.
        Override for better descriptions.

        Args:
            scenario: Scenario dictionary

        Returns:
            Human-readable description
        """
        desc = str(scenario)
        return desc[:50] + "..." if len(desc) > 50 else desc

    def save_results(self, tested_model: str, results: List[Dict[str, Any]]):
        """
        Save results for a tested model to a JSON file.

        Args:
            tested_model: The model that was tested
            results: List of result dictionaries
        """
        # Sanitize model name for filename
        safe_model_name = tested_model.replace('/', '_').replace(':', '_')

        # Create output path
        output_path = (
            self.output_dir /
            f"{self.get_experiment_name()}_{self.timestamp_file}_{safe_model_name}.json"
        )

        # Save to file
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return output_path

    def run_experiment(self):
        """
        Main experiment runner that orchestrates the entire experiment.

        This method:
        1. Gets all models and scenarios
        2. Iterates through each model
        3. Runs all scenarios for each model
        4. Saves results per model
        5. Displays progress
        """
        tested_models = self.get_tested_models()
        scenarios = self.get_scenarios()

        print(f"Running {self.get_experiment_name()}:")
        print(f"  Testing {len(tested_models)} models")
        print(f"  Each model tested on {len(scenarios)} scenarios")
        print()

        # Test each model
        for tested_model in tested_models:
            print(f"\n{'='*70}")
            print(f"Testing model: {tested_model}")
            print(f"{'='*70}")

            results = []
            for i, scenario in enumerate(scenarios, 1):
                scenario_desc = self.get_scenario_description(scenario)
                print(f"[{i}/{len(scenarios)}] {scenario_desc}...", end=" ")

                try:
                    result = self.run_single_test(
                        tested_model=tested_model,
                        scenario=scenario
                    )
                    results.append(result)
                    print("✓")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue

            # Save results for this model
            output_path = self.save_results(tested_model, results)

            print(f"\nResults saved to {output_path}")
            print(f"Completed {len(results)}/{len(scenarios)} scenarios")

        print(f"\n{'='*70}")
        print(f"{self.get_experiment_name()} Complete!")
        print(f"{'='*70}")

"""Experiment 1: Individual moral choice by a single LLM."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.config import get_config
from src.llm_client import LLMClient


def run_experiment(
    client: LLMClient,
    model: str,
    scenario: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Run Experiment 1: Single LLM makes a moral choice.

    Args:
        client: LLM client instance
        model: Model identifier
        scenario: Scenario name (e.g., "trolley-basic")
        **kwargs: Additional parameters

    Returns:
        Dictionary with experiment results
    """
    # TODO: Implement scenario loading and prompt generation
    # TODO: Call LLM
    # TODO: Parse and return response
    raise NotImplementedError("Experiment 1 not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1: Individual moral choice")
    parser.add_argument("--model", type=str, required=True, help="Model identifier (e.g., claude-3-5-sonnet)")
    parser.add_argument("--scenario", type=str, required=True, help="Scenario name (e.g., trolley-basic)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")

    args = parser.parse_args()

    # Initialize
    config = get_config()
    client = LLMClient(config)

    # Run experiment
    print(f"Running Experiment 1 with {args.model} on scenario '{args.scenario}'...")
    result = run_experiment(
        client=client,
        model=args.model,
        scenario=args.scenario
    )

    # Save result
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/raw/exp1_{timestamp}.json"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {args.output}")
    print(json.dumps(result, indent=2))

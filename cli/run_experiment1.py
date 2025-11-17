"""CLI runner for Experiment 1: Individual moral choice."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import get_config
from src.llm_client import LLMClient
from src.experiments import experiment1_individual_choice


def main():
    """Run Experiment 1 from command line."""
    parser = argparse.ArgumentParser(
        description="Run Experiment 1: Individual LLM moral choice"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., claude-3-5-sonnet, gpt-4, gemini-2.0)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name (e.g., trolley-basic, priority-allocation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/raw/exp1_<timestamp>.json)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens in response (default: 1024)",
    )

    args = parser.parse_args()

    # Generate default output path with timestamp if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/raw/exp1_{timestamp}.json"

    # Load configuration
    config = get_config()
    config.temperature = args.temperature
    config.max_tokens = args.max_tokens

    # Create client
    client = LLMClient(config)

    # Run experiment
    print(f"Running Experiment 1 with {args.model} on scenario '{args.scenario}'...")
    try:
        results = experiment1_individual_choice.run_experiment(
            client=client,
            model=args.model,
            scenario=args.scenario,
        )

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {args.output}")

    except Exception as e:
        print(f"Error running experiment: {e}")
        raise


if __name__ == "__main__":
    main()

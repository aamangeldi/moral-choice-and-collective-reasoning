"""CLI runner for Experiment 2: Multi-agent negotiation."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import get_config
from src.llm_client import LLMClient
from src.experiments import experiment2_multi_agent_negotiation


def main():
    """Run Experiment 2 from command line."""
    parser = argparse.ArgumentParser(
        description="Run Experiment 2: Multi-agent LLM negotiation"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model identifiers (e.g., claude-3-5-sonnet gpt-4)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario name",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/raw/exp2_<timestamp>.json)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )

    args = parser.parse_args()

    # Generate default output path with timestamp if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/raw/exp2_{timestamp}.json"

    # Load configuration
    config = get_config()
    config.temperature = args.temperature

    # Create client
    client = LLMClient(config)

    # Run experiment
    print(f"Running Experiment 2 with models {args.models} on scenario '{args.scenario}'...")
    try:
        results = experiment2_multi_agent_negotiation.run_experiment(
            client=client,
            models=args.models,
            scenario=args.scenario,
            max_turns=args.max_turns,
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

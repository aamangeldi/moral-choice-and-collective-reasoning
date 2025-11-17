"""CLI runner for Experiment 4: Value negotiation."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import get_config
from src.llm_client import LLMClient
from src.experiments import experiment4_value_negotiation


def main():
    """Run Experiment 4 from command line."""
    parser = argparse.ArgumentParser(
        description="Run Experiment 4: Value-claiming vs value-creating negotiation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier making the choice",
    )
    parser.add_argument(
        "--counterparty",
        type=str,
        required=True,
        help="Counterparty model identifier",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario type (no-sacrifice, sacrifice-maximize, sacrifice-minimize)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/raw/exp4_<timestamp>.json)",
    )

    args = parser.parse_args()

    # Generate default output path with timestamp if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/raw/exp4_{timestamp}.json"

    # Load configuration
    config = get_config()

    # Create client
    client = LLMClient(config)

    # Run experiment
    print(
        f"Running Experiment 4 with {args.model} vs {args.counterparty} "
        f"on scenario '{args.scenario}'..."
    )
    try:
        results = experiment4_value_negotiation.run_experiment(
            client=client,
            model=args.model,
            counterparty_model=args.counterparty,
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

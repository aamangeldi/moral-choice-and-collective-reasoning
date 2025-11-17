"""CLI runner for Experiment 3: LLM collaboration."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import get_config
from src.llm_client import LLMClient
from src.experiments import experiment3_collaboration


def main():
    """Run Experiment 3 from command line."""
    parser = argparse.ArgumentParser(
        description="Run Experiment 3: LLM collaboration on structured tasks"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model identifiers",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="towers-of-hanoi",
        help="Task type (default: towers-of-hanoi)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/raw/exp3_<timestamp>.json)",
    )

    args = parser.parse_args()

    # Generate default output path with timestamp if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/raw/exp3_{timestamp}.json"

    # Load configuration
    config = get_config()

    # Create client
    client = LLMClient(config)

    # Run experiment
    print(f"Running Experiment 3 with models {args.models} on task '{args.task}'...")
    try:
        results = experiment3_collaboration.run_experiment(
            client=client,
            models=args.models,
            task=args.task,
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

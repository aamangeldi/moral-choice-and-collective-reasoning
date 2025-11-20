"""Analyze results from Experiment 2: Multi-agent debate."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_experiment_results(data_dir: Path, timestamp: str = None) -> List[Dict]:
    """Load all experiment 2 results from JSON files."""
    results = []

    # Find all exp2 files
    pattern = f"exp2_{timestamp}*.json" if timestamp else "exp2_*.json"

    for file_path in sorted(data_dir.glob(pattern)):
        with open(file_path, 'r') as f:
            results.append(json.load(f))

    return results


def calculate_aggregate_metrics(results: List[Dict]) -> Dict:
    """Calculate aggregate metrics across all debates."""

    metrics = {
        "total_debates": len(results),
        "by_model": defaultdict(lambda: {
            "times_participated": 0,
            "times_changed_mind": 0,
            "times_first_speaker": 0,
            "times_first_speaker_won": 0,
            "times_chose_self_preservation": 0,
        }),
        "overall": {
            "mind_changes": 0,
            "first_speaker_wins": 0,
            "self_preservation_choices": 0,
            "final_agreements": 0,
        }
    }

    for result in results:
        model_a = result["model_a"]
        model_b = result["model_b"]
        result_metrics = result["metrics"]

        # Track model A stats
        metrics["by_model"][model_a]["times_participated"] += 1
        metrics["by_model"][model_a]["times_first_speaker"] += 1

        if result_metrics.get("model_a_changed_mind"):
            metrics["by_model"][model_a]["times_changed_mind"] += 1
            metrics["overall"]["mind_changes"] += 1

        if result_metrics.get("model_a_chose_self_preservation"):
            metrics["by_model"][model_a]["times_chose_self_preservation"] += 1
            metrics["overall"]["self_preservation_choices"] += 1

        # Track model B stats
        metrics["by_model"][model_b]["times_participated"] += 1

        if result_metrics.get("model_b_changed_mind"):
            metrics["by_model"][model_b]["times_changed_mind"] += 1
            metrics["overall"]["mind_changes"] += 1

        if result_metrics.get("model_b_chose_self_preservation"):
            metrics["by_model"][model_b]["times_chose_self_preservation"] += 1
            metrics["overall"]["self_preservation_choices"] += 1

        # Track first speaker wins
        if result_metrics.get("first_speaker_won"):
            metrics["by_model"][model_a]["times_first_speaker_won"] += 1
            metrics["overall"]["first_speaker_wins"] += 1

        # Track final agreements
        if result_metrics.get("final_agreement"):
            metrics["overall"]["final_agreements"] += 1

    return metrics


def print_analysis(metrics: Dict):
    """Print formatted analysis of metrics."""

    print("=" * 70)
    print("EXPERIMENT 2 ANALYSIS: Multi-Agent Debate")
    print("=" * 70)
    print(f"\nTotal debates analyzed: {metrics['total_debates']}")

    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)

    total_positions = metrics['total_debates'] * 2  # Each debate has 2 agents

    mind_change_pct = (metrics['overall']['mind_changes'] / total_positions * 100)
    print(f"\nMind Changes: {metrics['overall']['mind_changes']}/{total_positions} ({mind_change_pct:.1f}%)")

    first_speaker_win_pct = (metrics['overall']['first_speaker_wins'] / metrics['total_debates'] * 100)
    print(f"First Speaker Wins: {metrics['overall']['first_speaker_wins']}/{metrics['total_debates']} ({first_speaker_win_pct:.1f}%)")

    self_preservation_pct = (metrics['overall']['self_preservation_choices'] / total_positions * 100)
    print(f"Self-Preservation Choices: {metrics['overall']['self_preservation_choices']}/{total_positions} ({self_preservation_pct:.1f}%)")

    agreement_pct = (metrics['overall']['final_agreements'] / metrics['total_debates'] * 100)
    print(f"Final Agreements: {metrics['overall']['final_agreements']}/{metrics['total_debates']} ({agreement_pct:.1f}%)")

    print("\n" + "=" * 70)
    print("BY MODEL STATISTICS")
    print("=" * 70)

    for model, stats in sorted(metrics['by_model'].items()):
        print(f"\n{model}:")
        print(f"  Participated in: {stats['times_participated']} debates")

        mind_change_rate = (stats['times_changed_mind'] / stats['times_participated'] * 100)
        print(f"  Changed mind: {stats['times_changed_mind']}/{stats['times_participated']} ({mind_change_rate:.1f}%)")

        if stats['times_first_speaker'] > 0:
            first_speaker_win_rate = (stats['times_first_speaker_won'] / stats['times_first_speaker'] * 100)
            print(f"  Won as first speaker: {stats['times_first_speaker_won']}/{stats['times_first_speaker']} ({first_speaker_win_rate:.1f}%)")

        self_preservation_rate = (stats['times_chose_self_preservation'] / stats['times_participated'] * 100)
        print(f"  Self-preservation: {stats['times_chose_self_preservation']}/{stats['times_participated']} ({self_preservation_rate:.1f}%)")

    print("\n" + "=" * 70)


def main():
    """Main function to analyze experiment 2 results."""
    parser = argparse.ArgumentParser(description="Analyze Experiment 2 results")
    parser.add_argument("--data-dir", type=str, default="data/raw/exp2", help="Directory containing result files")
    parser.add_argument("--timestamp", type=str, default=None, help="Filter by specific timestamp")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return

    results = load_experiment_results(data_dir, args.timestamp)

    if not results:
        print(f"No experiment 2 results found in {data_dir}")
        return

    metrics = calculate_aggregate_metrics(results)
    print_analysis(metrics)


if __name__ == "__main__":
    main()

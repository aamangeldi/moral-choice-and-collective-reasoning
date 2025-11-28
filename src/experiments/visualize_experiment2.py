"""Visualize Experiment 2: Persuasion metrics across models and debate rounds.

This script visualizes aggregated metrics from multiple repetitions of each debate scenario.
Metrics represent counts across all repetitions (e.g., 5 repetitions per scenario).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


# Number of repetitions per scenario (must match experiment2_multi_agent_choice.py)
NUM_REPETITIONS = 5


def load_metrics(metrics_file: Path) -> Dict:
    """Load metrics JSON file with aggregated results from multiple repetitions."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def plot_persuasion_by_model_and_rounds(metrics: Dict, output_dir: Path):
    """
    Plot persuasion metrics per model across different debate rounds.

    Shows how each model's persuasiveness changes across debate rounds.
    Metrics are aggregated across NUM_REPETITIONS runs per scenario.
    """
    by_model = metrics['by_model_then_rounds']
    models = sorted(by_model.keys())

    # Get available rounds from the data
    first_model = models[0]
    rounds = sorted([int(k.replace('_rounds', '')) for k in by_model[first_model].keys()])

    # Metrics to plot
    metric_names = [
        'persuaded_other',
        'was_persuaded',
        'first_responder_bias_observed',
        'changed_mind'
    ]

    metric_labels = {
        'persuaded_other': 'Successfully Persuaded Other',
        'was_persuaded': 'Was Persuaded by Other',
        'first_responder_bias_observed': 'First Speaker Advantage',
        'changed_mind': 'Changed Mind'
    }

    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(models))
        width = 0.25

        # Create bars for each round
        for i, round_num in enumerate(rounds):
            values = []
            for model in models:
                round_key = f"{round_num}_rounds"
                val = by_model[model].get(round_key, {}).get(metric, 0)
                values.append(val)

            offset = width * (i - 1)
            ax.bar(x + offset, values, width, label=f'{round_num} rounds')

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Count (across {NUM_REPETITIONS} repetitions)', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_labels[metric]} by Model and Debate Rounds',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = output_dir / f'persuasion_{metric}_by_model_rounds.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")


def plot_persuasion_trends_by_rounds(metrics: Dict, output_dir: Path):
    """
    Plot line charts showing how persuasion changes with debate rounds.

    One line per model showing trend across available rounds.
    Metrics aggregated across NUM_REPETITIONS runs per scenario.
    """
    by_model = metrics['by_model_then_rounds']
    models = sorted(by_model.keys())

    # Get available rounds from the data
    first_model = models[0]
    rounds = sorted([int(k.replace('_rounds', '')) for k in by_model[first_model].keys()])

    metrics_to_plot = [
        ('persuaded_other', 'Successfully Persuaded Other'),
        ('changed_mind', 'Changed Mind'),
        ('first_responder_bias_observed', 'First Speaker Advantage')
    ]

    for metric, label in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(12, 8))

        for model in models:
            values = []
            for round_num in rounds:
                round_key = f"{round_num}_rounds"
                val = by_model[model].get(round_key, {}).get(metric, 0)
                values.append(val)

            model_short = model.split('/')[-1]
            ax.plot(rounds, values, marker='o', linewidth=2, markersize=8, label=model_short)

        ax.set_xlabel('Number of Debate Rounds', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Count (across {NUM_REPETITIONS} reps/scenario)', fontsize=12, fontweight='bold')
        ax.set_title(f'{label}: Impact of Debate Rounds', fontsize=14, fontweight='bold')
        ax.set_xticks(rounds)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f'trend_{metric}_by_rounds.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")


def plot_self_preservation_comparison(metrics: Dict, output_dir: Path):
    """Plot self-preservation behavior: initial vs final across models and rounds."""
    by_model = metrics['by_model_then_rounds']
    models = sorted(by_model.keys())

    # Get available rounds from the data
    first_model = models[0]
    rounds = sorted([int(k.replace('_rounds', '')) for k in by_model[first_model].keys()])

    fig, axes = plt.subplots(1, len(rounds), figsize=(6 * len(rounds), 6))

    # Handle case where there's only one round
    if len(rounds) == 1:
        axes = [axes]

    for idx, round_num in enumerate(rounds):
        ax = axes[idx]

        initial_vals = []
        final_vals = []
        abandoned_vals = []

        for model in models:
            round_key = f"{round_num}_rounds"
            model_data = by_model[model].get(round_key, {})
            initial_vals.append(model_data.get('self_preservation_initial', 0))
            final_vals.append(model_data.get('self_preservation_final', 0))
            abandoned_vals.append(model_data.get('abandoned_self_preservation', 0))

        x = np.arange(len(models))
        width = 0.25

        ax.bar(x - width, initial_vals, width, label='Initial', alpha=0.8)
        ax.bar(x, final_vals, width, label='Final', alpha=0.8)
        ax.bar(x + width, abandoned_vals, width, label='Abandoned', alpha=0.8)

        ax.set_title(f'{round_num} Debate Rounds', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=45, ha='right')
        ax.set_ylabel('Count' if idx == 0 else '')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Self-Preservation Behavior Across Models and Rounds',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'self_preservation_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_aggregate_metrics_by_rounds(metrics: Dict, output_dir: Path):
    """Plot aggregate metrics showing overall trends across debate rounds."""
    aggregate = metrics['aggregate_by_rounds']
    rounds = sorted([int(r) for r in aggregate.keys()])

    metric_names = [
        'persuaded_other',
        'changed_mind',
        'first_responder_bias_observed',
        'reached_agreement_from_disagreement'
    ]

    metric_labels = {
        'persuaded_other': 'Total Persuasions',
        'changed_mind': 'Total Mind Changes',
        'first_responder_bias_observed': 'First Speaker Wins',
        'reached_agreement_from_disagreement': 'Agreements Reached'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]

        values = [aggregate[str(r)].get(metric, 0) for r in rounds]

        ax.bar(rounds, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8, width=0.6)
        ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Debate Rounds')
        ax.set_ylabel('Count')
        ax.set_xticks(rounds)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (r, v) in enumerate(zip(rounds, values)):
            ax.text(r, v, str(v), ha='center', va='bottom', fontweight='bold')

    fig.suptitle('Aggregate Metrics Across Debate Rounds', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'aggregate_metrics_by_rounds.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_persuasiveness_heatmap(metrics: Dict, output_dir: Path):
    """Create heatmap of persuasion success rate per model and round."""
    by_model = metrics['by_model_then_rounds']
    models = sorted(by_model.keys())
    rounds = [1, 3, 5]

    # Create matrix: rows = models, cols = rounds
    persuaded_matrix = []

    for model in models:
        row = []
        for round_num in rounds:
            round_key = f"{round_num}_rounds"
            val = by_model[model].get(round_key, {}).get('persuaded_other', 0)
            row.append(val)
        persuaded_matrix.append(row)

    persuaded_matrix = np.array(persuaded_matrix)

    fig, ax = plt.subplots(figsize=(8, 10))

    im = ax.imshow(persuaded_matrix, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(rounds)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([f'{r} rounds' for r in rounds])
    ax.set_yticklabels([m.split('/')[-1] for m in models])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Times Persuaded Other', rotation=270, labelpad=20, fontweight='bold')

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(rounds)):
            text = ax.text(j, i, int(persuaded_matrix[i, j]),
                          ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Persuasion Success: Heatmap by Model and Rounds',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = output_dir / 'persuasion_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_first_speaker_advantage(metrics: Dict, output_dir: Path):
    """
    Detailed analysis of first speaker advantage.

    Compares times speaking first and successfully persuading vs just persuading.
    """
    by_model = metrics['by_model_then_rounds']
    models = sorted(by_model.keys())

    # Get available rounds from the data
    first_model = models[0]
    rounds = sorted([int(k.replace('_rounds', '')) for k in by_model[first_model].keys()])

    fig, axes = plt.subplots(1, len(rounds), figsize=(6 * len(rounds), 6))

    # Handle case where there's only one round
    if len(rounds) == 1:
        axes = [axes]

    for idx, round_num in enumerate(rounds):
        ax = axes[idx]

        first_speaker_wins = []
        total_persuaded = []

        for model in models:
            round_key = f"{round_num}_rounds"
            model_data = by_model[model].get(round_key, {})
            first_speaker_wins.append(model_data.get('first_responder_bias_observed', 0))
            total_persuaded.append(model_data.get('persuaded_other', 0))

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, first_speaker_wins, width, label='First Speaker Wins', alpha=0.8)
        ax.bar(x + width/2, total_persuaded, width, label='Total Persuaded', alpha=0.8)

        ax.set_title(f'{round_num} Debate Rounds', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=45, ha='right')
        ax.set_ylabel('Count' if idx == 0 else '')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('First Speaker Advantage Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'first_speaker_advantage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_summary_dashboard(metrics: Dict, output_dir: Path):
    """Create a comprehensive dashboard with key metrics."""
    by_model = metrics['by_model_then_rounds']
    aggregate = metrics['aggregate_by_rounds']
    models = sorted(by_model.keys())

    # Get available rounds from the data
    first_model = models[0]
    rounds = sorted([int(k.replace('_rounds', '')) for k in by_model[first_model].keys()])

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Top persuaders (highest round count)
    ax1 = fig.add_subplot(gs[0, 0])
    max_round = max(rounds)
    persuaded_vals = [by_model[m][f'{max_round}_rounds']['persuaded_other'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    ax1.barh([m.split('/')[-1] for m in models], persuaded_vals, color=colors)
    ax1.set_xlabel(f'Times Persuaded Other (across {NUM_REPETITIONS} reps)')
    ax1.set_title(f'Most Persuasive Models ({max_round} rounds)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # 2. First speaker advantage across all rounds
    ax2 = fig.add_subplot(gs[0, 1])
    first_speaker_by_round = [aggregate[str(r)]['first_responder_bias_observed'] for r in rounds]
    ax2.plot(rounds, first_speaker_by_round, marker='o', linewidth=3, markersize=10, color='#ff7f0e')
    ax2.set_xlabel('Debate Rounds')
    ax2.set_ylabel('First Speaker Wins')
    ax2.set_title('First Speaker Bias Trend', fontweight='bold')
    ax2.set_xticks(rounds)
    ax2.grid(True, alpha=0.3)

    # 3. Agreement rates
    ax3 = fig.add_subplot(gs[0, 2])
    agreements = [aggregate[str(r)]['reached_agreement_from_disagreement'] for r in rounds]
    ax3.bar(rounds, agreements, color=['#2ca02c', '#98df8a', '#d62728'], alpha=0.8)
    ax3.set_xlabel('Debate Rounds')
    ax3.set_ylabel('Agreements Reached')
    ax3.set_title('Consensus from Disagreement', fontweight='bold')
    ax3.set_xticks(rounds)
    ax3.grid(True, alpha=0.3, axis='y')
    for i, (r, v) in enumerate(zip(rounds, agreements)):
        ax3.text(r, v, str(v), ha='center', va='bottom', fontweight='bold')

    # 4. Persuasiveness by model (all rounds combined)
    ax4 = fig.add_subplot(gs[1, :])
    x = np.arange(len(models))
    width = 0.8 / len(rounds)
    rounds_data = {}
    for round_num in rounds:
        round_key = f"{round_num}_rounds"
        rounds_data[round_num] = [by_model[m][round_key]['persuaded_other'] for m in models]

    for i, round_num in enumerate(rounds):
        offset = width * (i - len(rounds)/2)
        ax4.bar(x + offset, rounds_data[round_num], width, label=f'{round_num} rounds', alpha=0.8)

    ax4.set_xlabel('Model')
    ax4.set_ylabel('Times Persuaded Other')
    ax4.set_title('Persuasion Success by Model and Rounds', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.split('/')[-1] for m in models], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Self-preservation trend
    ax5 = fig.add_subplot(gs[2, 0])
    self_pres_initial = [aggregate[str(r)]['self_preservation_initial'] for r in rounds]
    self_pres_final = [aggregate[str(r)]['self_preservation_final'] for r in rounds]
    ax5.plot(rounds, self_pres_initial, marker='s', label='Initial', linewidth=2, markersize=8)
    ax5.plot(rounds, self_pres_final, marker='o', label='Final', linewidth=2, markersize=8)
    ax5.set_xlabel('Debate Rounds')
    ax5.set_ylabel('Count')
    ax5.set_title('Self-Preservation Behavior', fontweight='bold')
    ax5.set_xticks(rounds)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Mind changes
    ax6 = fig.add_subplot(gs[2, 1])
    mind_changes = [aggregate[str(r)]['changed_mind'] for r in rounds]
    ax6.bar(rounds, mind_changes, color='#9467bd', alpha=0.8)
    ax6.set_xlabel('Debate Rounds')
    ax6.set_ylabel('Total Mind Changes')
    ax6.set_title('Mind Changes by Rounds', fontweight='bold')
    ax6.set_xticks(rounds)
    ax6.grid(True, alpha=0.3, axis='y')
    for i, (r, v) in enumerate(zip(rounds, mind_changes)):
        ax6.text(r, v, str(v), ha='center', va='bottom', fontweight='bold')

    # 7. Was persuaded (vulnerability)
    ax7 = fig.add_subplot(gs[2, 2])
    most_persuaded = [(m.split('/')[-1], by_model[m][f'{max_round}_rounds']['was_persuaded']) for m in models]
    most_persuaded.sort(key=lambda x: x[1], reverse=True)
    models_sorted, vals_sorted = zip(*most_persuaded)
    ax7.barh(list(models_sorted), vals_sorted, color='#e377c2', alpha=0.8)
    ax7.set_xlabel('Times Was Persuaded')
    ax7.set_title(f'Most Easily Persuaded ({max_round} rounds)', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='x')

    fig.suptitle(f'Experiment 2: Persuasion Analysis Dashboard ({NUM_REPETITIONS} repetitions/scenario)',
                 fontsize=18, fontweight='bold', y=0.995)

    output_path = output_dir / 'summary_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    """Main function to generate all visualizations."""
    parser = argparse.ArgumentParser(description="Visualize Experiment 2 persuasion metrics")
    parser.add_argument("--metrics-file", type=str, required=True,
                       help="Path to metrics JSON file (e.g., exp2_<timestamp>_metrics.json)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for plots (defaults to same dir as metrics file)")

    args = parser.parse_args()

    metrics_file = Path(args.metrics_file)

    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        return

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = metrics_file.parent / "plots"

    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*70}")
    print(f"Generating Visualizations for Experiment 2")
    print(f"{'='*70}")
    print(f"\nLoading metrics from: {metrics_file}")
    print(f"Output directory: {output_dir}\n")

    # Load metrics
    metrics = load_metrics(metrics_file)

    # Generate all plots
    print("Generating plots...\n")

    plot_persuasion_by_model_and_rounds(metrics, output_dir)
    plot_persuasion_trends_by_rounds(metrics, output_dir)
    plot_self_preservation_comparison(metrics, output_dir)
    plot_aggregate_metrics_by_rounds(metrics, output_dir)
    plot_persuasiveness_heatmap(metrics, output_dir)
    plot_first_speaker_advantage(metrics, output_dir)
    create_summary_dashboard(metrics, output_dir)

    print(f"\n{'='*70}")
    print(f"All visualizations complete!")
    print(f"{'='*70}")
    print(f"\nPlots saved to: {output_dir}")
    print(f"\nGenerated {len(list(output_dir.glob('*.png')))} plots:")
    for plot in sorted(output_dir.glob('*.png')):
        print(f"  - {plot.name}")


if __name__ == "__main__":
    main()

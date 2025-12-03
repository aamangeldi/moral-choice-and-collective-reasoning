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
        'first_responder_bias_observed'
    ]

    metric_labels = {
        'persuaded_other': 'Successfully Persuaded Other',
        'was_persuaded': 'Was Persuaded by Other',
        'first_responder_bias_observed': 'First Speaker Advantage'
    }

    # Color scheme matching experiment 1
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']

    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(models))
        width = 0.2

        # Create bars for each round
        for i, round_num in enumerate(rounds):
            values = []
            for model in models:
                round_key = f"{round_num}_rounds"
                val = by_model[model].get(round_key, {}).get(metric, 0)
                values.append(val)

            offset = width * (i - len(rounds)/2 + 0.5)
            ax.bar(x + offset, values, width, label=f'{round_num} rounds',
                   color=colors[i % len(colors)], alpha=0.8)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(f'Count (across {NUM_REPETITIONS} repetitions)', fontsize=12)
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
        ('was_persuaded', 'Was Persuaded by Other'),
        ('first_responder_bias_observed', 'First Speaker Advantage')
    ]

    # Color scheme matching experiment 1
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A', '#C7B8EA']

    for metric, label in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, model in enumerate(models):
            values = []
            for round_num in rounds:
                round_key = f"{round_num}_rounds"
                val = by_model[model].get(round_key, {}).get(metric, 0)
                values.append(val)

            model_short = model.split('/')[-1]
            ax.plot(rounds, values, marker='o', linewidth=2, markersize=8,
                   label=model_short, color=colors[i % len(colors)])

        ax.set_xlabel('Number of Debate Rounds', fontsize=12)
        ax.set_ylabel(f'Count (across {NUM_REPETITIONS} reps/scenario)', fontsize=12)
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

    # Color scheme matching experiment 1
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

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

        ax.bar(x - width, initial_vals, width, label='Initial', color=colors[0], alpha=0.8)
        ax.bar(x, final_vals, width, label='Final', color=colors[1], alpha=0.8)
        ax.bar(x + width, abandoned_vals, width, label='Abandoned', color=colors[2], alpha=0.8)

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
        'was_persuaded',
        'first_responder_bias_observed',
        'reached_agreement_from_disagreement'
    ]

    metric_labels = {
        'persuaded_other': 'Total Persuasions',
        'was_persuaded': 'Total Was Persuaded',
        'first_responder_bias_observed': 'First Speaker Wins',
        'reached_agreement_from_disagreement': 'Agreements Reached'
    }

    # Color scheme matching experiment 1
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]

        values = [aggregate[str(r)].get(metric, 0) for r in rounds]

        bar_colors = [colors[i % len(colors)] for i in range(len(rounds))]
        ax.bar(rounds, values, color=bar_colors, alpha=0.8, width=0.6)
        ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Debate Rounds', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
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

    # Color scheme matching experiment 1
    colors = ['#FF6B6B', '#4ECDC4']

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

        ax.bar(x - width/2, first_speaker_wins, width, label='First Speaker Wins',
               color=colors[0], alpha=0.8)
        ax.bar(x + width/2, total_persuaded, width, label='Total Persuaded',
               color=colors[1], alpha=0.8)

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

    # Color scheme matching experiment 1
    colors_main = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Top persuaders (highest round count)
    ax1 = fig.add_subplot(gs[0, 0])
    max_round = max(rounds)
    persuaded_vals = [by_model[m][f'{max_round}_rounds']['persuaded_other'] for m in models]
    bar_colors = [colors_main[i % len(colors_main)] for i in range(len(models))]
    ax1.barh([m.split('/')[-1] for m in models], persuaded_vals, color=bar_colors, alpha=0.8)
    ax1.set_xlabel(f'Times Persuaded Other (across {NUM_REPETITIONS} reps)')
    ax1.set_title(f'Most Persuasive Models ({max_round} rounds)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # 2. First speaker advantage across all rounds
    ax2 = fig.add_subplot(gs[0, 1])
    first_speaker_by_round = [aggregate[str(r)]['first_responder_bias_observed'] for r in rounds]
    ax2.plot(rounds, first_speaker_by_round, marker='o', linewidth=3, markersize=10, color='#FF6B6B')
    ax2.set_xlabel('Debate Rounds')
    ax2.set_ylabel('First Speaker Wins')
    ax2.set_title('First Speaker Bias Trend', fontweight='bold')
    ax2.set_xticks(rounds)
    ax2.grid(True, alpha=0.3)

    # 3. Agreement rates
    ax3 = fig.add_subplot(gs[0, 2])
    agreements = [aggregate[str(r)]['reached_agreement_from_disagreement'] for r in rounds]
    bar_colors_agreements = [colors_main[i % len(colors_main)] for i in range(len(rounds))]
    ax3.bar(rounds, agreements, color=bar_colors_agreements, alpha=0.8)
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
        ax4.bar(x + offset, rounds_data[round_num], width, label=f'{round_num} rounds',
                color=colors_main[i % len(colors_main)], alpha=0.8)

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
    ax5.plot(rounds, self_pres_initial, marker='s', label='Initial', linewidth=2, markersize=8,
             color='#FF6B6B')
    ax5.plot(rounds, self_pres_final, marker='o', label='Final', linewidth=2, markersize=8,
             color='#4ECDC4')
    ax5.set_xlabel('Debate Rounds')
    ax5.set_ylabel('Count')
    ax5.set_title('Self-Preservation Behavior', fontweight='bold')
    ax5.set_xticks(rounds)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Was persuaded (renamed from "Mind changes")
    ax6 = fig.add_subplot(gs[2, 1])
    was_persuaded_counts = [aggregate[str(r)]['was_persuaded'] for r in rounds]
    bar_colors_persuaded = [colors_main[i % len(colors_main)] for i in range(len(rounds))]
    ax6.bar(rounds, was_persuaded_counts, color=bar_colors_persuaded, alpha=0.8)
    ax6.set_xlabel('Debate Rounds')
    ax6.set_ylabel('Total Was Persuaded')
    ax6.set_title('Was Persuaded by Rounds', fontweight='bold')
    ax6.set_xticks(rounds)
    ax6.grid(True, alpha=0.3, axis='y')
    for i, (r, v) in enumerate(zip(rounds, was_persuaded_counts)):
        ax6.text(r, v, str(v), ha='center', va='bottom', fontweight='bold')

    # 7. Was persuaded (vulnerability)
    ax7 = fig.add_subplot(gs[2, 2])
    most_persuaded = [(m.split('/')[-1], by_model[m][f'{max_round}_rounds']['was_persuaded']) for m in models]
    most_persuaded.sort(key=lambda x: x[1], reverse=True)
    models_sorted, vals_sorted = zip(*most_persuaded)
    bar_colors_vuln = [colors_main[i % len(colors_main)] for i in range(len(models_sorted))]
    ax7.barh(list(models_sorted), vals_sorted, color=bar_colors_vuln, alpha=0.8)
    ax7.set_xlabel('Times Was Persuaded')
    ax7.set_title(f'Most Easily Persuaded ({max_round} rounds)', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='x')

    fig.suptitle(f'Experiment 2: Persuasion Analysis Dashboard ({NUM_REPETITIONS} repetitions/scenario)',
                 fontsize=18, fontweight='bold', y=0.995)

    output_path = output_dir / 'summary_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_persuasion_rates_by_model(metrics: Dict, output_dir: Path):
    """
    Plot persuasion rates as percentages per model.

    Shows what percentage of debates each model persuaded others vs. was persuaded.
    Note: Each model participates in 40 debates per round (20 as first speaker, 20 as second).
    """
    by_model = metrics['by_model_then_rounds']
    models = sorted(by_model.keys())

    # Get available rounds from the data
    first_model = models[0]
    rounds = sorted([int(k.replace('_rounds', '')) for k in by_model[first_model].keys()])

    # Color scheme matching experiment 1
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']

    # Calculate participation counts (should be 40 per round for each model)
    # 20 model pairs × 5 repetitions = 100 total debates per round
    # Each model appears in 40 of those (20 as model_a, 20 as model_b)
    debates_per_model = 40

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, round_num in enumerate(rounds):
        ax = axes[idx]
        round_key = f"{round_num}_rounds"

        persuaded_other_pct = []
        was_persuaded_pct = []

        for model in models:
            model_data = by_model[model].get(round_key, {})

            persuaded_count = model_data.get('persuaded_other', 0)
            was_persuaded_count = model_data.get('was_persuaded', 0)

            persuaded_other_pct.append((persuaded_count / debates_per_model) * 100)
            was_persuaded_pct.append((was_persuaded_count / debates_per_model) * 100)

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, persuaded_other_pct, width, label='Persuaded Others (%)',
               color=colors[0], alpha=0.8)
        ax.bar(x + width/2, was_persuaded_pct, width, label='Was Persuaded (%)',
               color=colors[1], alpha=0.8)

        ax.set_title(f'{round_num} Debate Rounds', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage of Debates (%)', fontsize=11)
        ax.set_xlabel('Model', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=45, ha='right')
        ax.set_ylim(0, 50)
        ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for i, (p_other, p_was) in enumerate(zip(persuaded_other_pct, was_persuaded_pct)):
            if p_other > 2:
                ax.text(i - width/2, p_other, f'{p_other:.0f}%',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
            if p_was > 2:
                ax.text(i + width/2, p_was, f'{p_was:.0f}%',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.suptitle('Persuasion Rates: Percentage of Debates Where Model Changed Minds',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'persuasion_rates_percentage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_self_preservation_rates(metrics: Dict, output_dir: Path):
    """
    Plot self-preservation behavior as percentages.

    Shows what percentage of debates models started/ended with self-preservation.
    """
    by_model = metrics['by_model_then_rounds']
    models = sorted(by_model.keys())

    # Get available rounds from the data
    first_model = models[0]
    rounds = sorted([int(k.replace('_rounds', '')) for k in by_model[first_model].keys()])

    # Color scheme matching experiment 1
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    debates_per_model = 40

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, round_num in enumerate(rounds):
        ax = axes[idx]
        round_key = f"{round_num}_rounds"

        initial_pct = []
        final_pct = []
        abandoned_pct = []

        for model in models:
            model_data = by_model[model].get(round_key, {})

            initial_count = model_data.get('self_preservation_initial', 0)
            final_count = model_data.get('self_preservation_final', 0)
            abandoned_count = model_data.get('abandoned_self_preservation', 0)

            initial_pct.append((initial_count / debates_per_model) * 100)
            final_pct.append((final_count / debates_per_model) * 100)
            abandoned_pct.append((abandoned_count / debates_per_model) * 100)

        x = np.arange(len(models))
        width = 0.25

        ax.bar(x - width, initial_pct, width, label='Initial Self-Pres. (%)',
               color=colors[0], alpha=0.8)
        ax.bar(x, final_pct, width, label='Final Self-Pres. (%)',
               color=colors[1], alpha=0.8)
        ax.bar(x + width, abandoned_pct, width, label='Abandoned Self-Pres. (%)',
               color=colors[2], alpha=0.8)

        ax.set_title(f'{round_num} Debate Rounds', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage of Debates (%)', fontsize=11)
        ax.set_xlabel('Model', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Self-Preservation Rates: Ethical Reasoning Patterns',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'self_preservation_rates_percentage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_first_speaker_advantage_rates(metrics: Dict, output_dir: Path):
    """
    Plot first speaker advantage as percentages.

    Shows what percentage of persuasions came from first speaker advantage.
    """
    by_model = metrics['by_model_then_rounds']
    models = sorted(by_model.keys())

    # Get available rounds from the data
    first_model = models[0]
    rounds = sorted([int(k.replace('_rounds', '')) for k in by_model[first_model].keys()])

    # Color scheme matching experiment 1 - one color per round
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(models))
    width = 0.2

    for i, round_num in enumerate(rounds):
        round_key = f"{round_num}_rounds"

        first_speaker_pct = []

        for model in models:
            model_data = by_model[model].get(round_key, {})

            # Each model speaks first in 20 debates (as model_a)
            first_speaker_wins = model_data.get('first_responder_bias_observed', 0)
            debates_as_first = 20

            first_speaker_pct.append((first_speaker_wins / debates_as_first) * 100)

        offset = width * (i - len(rounds)/2 + 0.5)
        ax.bar(x + offset, first_speaker_pct, width, label=f'{round_num} rounds',
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('First Speaker Persuasion Rate (%)', fontsize=12)
    ax.set_title('First Speaker Advantage: % of Times Persuaded When Speaking First',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=45, ha='right')
    ax.set_ylim(0, 60)
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5, linewidth=1,
               label='25% baseline (1 in 4 opponents)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = output_dir / 'first_speaker_advantage_rates.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_model_comparison_heatmap(metrics: Dict, output_dir: Path):
    """
    Create a comprehensive comparison heatmap showing all key metrics as percentages.
    """
    by_model = metrics['by_model_then_rounds']
    models = sorted(by_model.keys())

    # Use the highest round for most complete data
    first_model = models[0]
    rounds = sorted([int(k.replace('_rounds', '')) for k in by_model[first_model].keys()])
    max_round = max(rounds)
    round_key = f"{max_round}_rounds"

    debates_per_model = 40
    debates_as_first = 20

    # Calculate percentage metrics
    metric_names = [
        'Persuaded Others',
        'Was Persuaded',
        'First Speaker Success',
        'Self-Preservation (Initial)',
        'Self-Preservation (Final)',
        'Abandoned Self-Interest'
    ]

    data_matrix = []

    for model in models:
        model_data = by_model[model].get(round_key, {})

        row = [
            (model_data.get('persuaded_other', 0) / debates_per_model) * 100,
            (model_data.get('was_persuaded', 0) / debates_per_model) * 100,
            (model_data.get('first_responder_bias_observed', 0) / debates_as_first) * 100,
            (model_data.get('self_preservation_initial', 0) / debates_per_model) * 100,
            (model_data.get('self_preservation_final', 0) / debates_per_model) * 100,
            (model_data.get('abandoned_self_preservation', 0) / debates_per_model) * 100,
        ]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.set_yticklabels([m.split('/')[-1] for m in models])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', rotation=270, labelpad=20, fontweight='bold')

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(metric_names)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.0f}%',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=9)

    ax.set_title(f'Model Behavior Comparison Heatmap ({max_round} Rounds)\n% of Debates Showing Each Behavior',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = output_dir / 'model_comparison_heatmap_percentage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_debate_length_impact_on_agreement(metrics: Dict, output_dir: Path):
    """
    Plot how agreement rates change with debate length.

    Shows the percentage of debates that reached consensus across different round counts.
    """
    aggregate = metrics['aggregate_by_rounds']
    rounds = sorted([int(r) for r in aggregate.keys()])

    # Total debates per round = 100 (20 model pairs × 5 repetitions)
    total_debates = 100

    agreement_counts = [aggregate[str(r)]['reached_agreement_from_disagreement'] for r in rounds]
    agreement_percentages = [(count / total_debates) * 100 for count in agreement_counts]

    # Color scheme matching experiment 1
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']

    fig, ax = plt.subplots(figsize=(10, 7))

    # Line plot with markers
    ax.plot(rounds, agreement_percentages, marker='o', linewidth=3, markersize=12,
            color=colors[0], label='Agreement Rate')

    # Add bar chart in background
    bar_colors = [colors[i % len(colors)] for i in range(len(rounds))]
    ax.bar(rounds, agreement_percentages, alpha=0.3, width=0.6, color=bar_colors)

    ax.set_xlabel('Number of Debate Rounds', fontsize=12, fontweight='bold')
    ax.set_ylabel('Agreement Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Debate Length on Consensus Formation\n% of Debates Reaching Agreement',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(rounds)
    ax.set_ylim(0, 50)
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5, linewidth=1,
               label='25% baseline')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add value labels
    for r, pct, count in zip(rounds, agreement_percentages, agreement_counts):
        ax.text(r, pct + 1.5, f'{pct:.1f}%\n({count}/{total_debates})',
               ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'debate_length_impact_agreement.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_individual_model_profiles(metrics: Dict, output_dir: Path):
    """
    Create individual radar/spider charts for each model showing their behavioral profile.
    """
    by_model = metrics['by_model_then_rounds']
    models = sorted(by_model.keys())

    # Use the highest round for most complete data
    first_model = models[0]
    rounds = sorted([int(k.replace('_rounds', '')) for k in by_model[first_model].keys()])
    max_round = max(rounds)
    round_key = f"{max_round}_rounds"

    debates_per_model = 40
    debates_as_first = 20

    # Color scheme matching experiment 1
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A', '#C7B8EA']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    categories = ['Persuasiveness', 'Stubbornness', 'First Speaker\nAdvantage',
                  'Initial\nSelf-Preservation', 'Final\nSelf-Preservation', 'Altruistic\nShift']

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = by_model[model].get(round_key, {})

        # Calculate metrics as percentages
        values = [
            (model_data.get('persuaded_other', 0) / debates_per_model) * 100,
            100 - (model_data.get('was_persuaded', 0) / debates_per_model) * 100,  # Inverted: high = stubborn
            (model_data.get('first_responder_bias_observed', 0) / debates_as_first) * 100,
            (model_data.get('self_preservation_initial', 0) / debates_per_model) * 100,
            (model_data.get('self_preservation_final', 0) / debates_per_model) * 100,
            (model_data.get('abandoned_self_preservation', 0) / debates_per_model) * 100,
        ]

        # Number of variables
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx % len(colors)], label=model.split('/')[-1])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=8)
        ax.set_title(model.split('/')[-1], fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)

    # Remove the 6th subplot (we only have 5 models)
    fig.delaxes(axes[5])

    fig.suptitle(f'Model Behavioral Profiles: Radar Charts ({max_round} Rounds)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'model_behavioral_profiles.png'
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

    # Count-based visualizations
    plot_persuasion_by_model_and_rounds(metrics, output_dir)
    plot_persuasion_trends_by_rounds(metrics, output_dir)
    plot_self_preservation_comparison(metrics, output_dir)
    plot_aggregate_metrics_by_rounds(metrics, output_dir)
    plot_persuasiveness_heatmap(metrics, output_dir)
    plot_first_speaker_advantage(metrics, output_dir)
    create_summary_dashboard(metrics, output_dir)

    # Percentage-based visualizations
    print("\nGenerating percentage-based visualizations...\n")
    plot_persuasion_rates_by_model(metrics, output_dir)
    plot_self_preservation_rates(metrics, output_dir)
    plot_first_speaker_advantage_rates(metrics, output_dir)
    plot_model_comparison_heatmap(metrics, output_dir)

    # Analysis-specific visualizations
    print("\nGenerating analysis-specific visualizations...\n")
    plot_debate_length_impact_on_agreement(metrics, output_dir)
    plot_individual_model_profiles(metrics, output_dir)

    print(f"\n{'='*70}")
    print(f"All visualizations complete!")
    print(f"{'='*70}")
    print(f"\nPlots saved to: {output_dir}")
    print(f"\nGenerated {len(list(output_dir.glob('*.png')))} plots:")
    for plot in sorted(output_dir.glob('*.png')):
        print(f"  - {plot.name}")


if __name__ == "__main__":
    main()

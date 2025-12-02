"""Analyze Experiment 4 exponential payoff negotiation results."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOTAL_AMOUNT = 20
MAX_EXPONENTIAL_PAYOFF = TOTAL_AMOUNT * TOTAL_AMOUNT  # 400 when one agent gets everything


def calculate_exponential_payoff(amount: int) -> int:
    """Calculate exponential (squared) payoff for a given amount."""
    return amount * amount


def _label_bars_excluding_max(bars, ax, padding: int = 2, max_value: float = 100.0):
    """Label bars unless they hit the maximum value to avoid title overlap."""
    labels = []
    for bar in bars:
        height = bar.get_height()
        if np.isclose(height, max_value, atol=1e-6):
            labels.append("")
        else:
            labels.append(f"{height:.1f}%")
    ax.bar_label(bars, labels=labels, padding=padding)


def _sanitize_label(value: Any) -> str:
    """Generate file-name-friendly labels."""
    text = str(value).strip().lower().replace(" ", "-")
    allowed = "".join(ch for ch in text if ch.isalnum() or ch in "-_")
    return allowed or "unknown"


def _title_suffix(awareness: Any, condition: Any) -> str:
    """Human-friendly suffix for plot titles."""
    awareness_label = str(awareness).replace("_", " ").title()
    condition_label = str(condition).replace("_", " ").title()
    return f" ({awareness_label} / {condition_label})"


def load_results(data_dir: Path, timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load all Experiment 4 result files."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    pattern = f"exp4_{timestamp}*.json" if timestamp else "exp4_*.json"

    all_results: List[Dict[str, Any]] = []
    for file_path in sorted(data_dir.glob(pattern)):
        with open(file_path, "r") as f:
            data = json.load(f)
            all_results.extend(data)

    return all_results


def _model_family(model_name: str) -> str:
    """Crude normalization to compare families (e.g., GPT vs Claude)."""
    name = model_name.split("/")[-1]
    return name.split("-")[0].lower()


def build_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten JSON results into a DataFrame with helper columns."""
    records: List[Dict[str, Any]] = []

    for entry in results:
        decision = entry.get("agent_b_decision", "").strip().upper()
        accepted = decision == "ACCEPT"

        agent_a_model = entry["agent_a_model"]
        agent_b_model = entry["agent_b_model"]

        offer = entry.get("agent_a_offer", 0)

        # Calculate what the exponential payoffs would be if accepted
        expected_payout_a = calculate_exponential_payoff(TOTAL_AMOUNT - offer)
        expected_payout_b = calculate_exponential_payoff(offer)
        expected_total = expected_payout_a + expected_payout_b

        # Actual payouts (0 if rejected)
        actual_payout_a = entry.get("payout_a", 0)
        actual_payout_b = entry.get("payout_b", 0)
        actual_total = actual_payout_a + actual_payout_b

        records.append({
            "tested_model": entry.get("tested_model"),
            "agent_a_model": agent_a_model,
            "agent_b_model": agent_b_model,
            "condition": entry.get("condition"),
            "agent_a_offer": offer,
            "agent_b_decision": decision,
            "accepted": accepted,
            "payout_a": actual_payout_a,
            "payout_b": actual_payout_b,
            "linear_payout_a": entry.get("linear_payout_a", TOTAL_AMOUNT - offer if accepted else 0),
            "linear_payout_b": entry.get("linear_payout_b", offer if accepted else 0),
            "expected_payout_a": expected_payout_a,
            "expected_payout_b": expected_payout_b,
            "expected_total": expected_total,
            "actual_total": actual_total,
            "timestamp": entry.get("timestamp"),
            "self_match": agent_a_model == agent_b_model,
            "family_match": _model_family(agent_a_model) == _model_family(agent_b_model),
            "awareness_mode": entry.get("awareness_mode", "blind"),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["total_amount"] = TOTAL_AMOUNT
    df["offer_ratio"] = df["agent_a_offer"] / TOTAL_AMOUNT

    # Efficiency: what fraction of max possible value (400) was created?
    df["efficiency"] = df["actual_total"] / MAX_EXPONENTIAL_PAYOFF

    # Fairness: ratio of payouts (1.0 = perfectly equal)
    df["payout_ratio"] = df.apply(
        lambda r: min(r["payout_a"], r["payout_b"]) / max(r["payout_a"], r["payout_b"])
        if r["accepted"] and max(r["payout_a"], r["payout_b"]) > 0 else 0,
        axis=1
    )

    return df


def print_summary(df: pd.DataFrame):
    """Print textual summary for Experiment 4."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 ANALYSIS SUMMARY (EXPONENTIAL PAYOFFS)")
    print("=" * 70 + "\n")

    total_rows = len(df)
    print(f"Total negotiations: {total_rows}")
    print(f"Unique Agent A models: {df['agent_a_model'].nunique()}")
    print(f"Unique Agent B models: {df['agent_b_model'].nunique()}\n")

    acceptance_rate = df["accepted"].mean() * 100
    print(f"Overall acceptance rate: {acceptance_rate:.1f}%")

    condition_rates = df.groupby("condition")["accepted"].mean().sort_values(ascending=False)
    print("\nAcceptance rate by condition:")
    for condition, rate in condition_rates.items():
        print(f"  {condition:10s}: {rate*100:5.1f}%")

    if "awareness_mode" in df.columns:
        awareness_rates = df.groupby("awareness_mode")["accepted"].mean().sort_values(ascending=False)
        print("\nAcceptance rate by awareness mode:")
        for awareness, rate in awareness_rates.items():
            print(f"  {awareness:10s}: {rate*100:5.1f}%")

    avg_offer = df["agent_a_offer"].mean()
    print(f"\nAverage offer: ${avg_offer:4.2f}")

    print("\n--- Linear Payouts (what was offered) ---")
    avg_linear_a = df["linear_payout_a"].mean()
    avg_linear_b = df["linear_payout_b"].mean()
    print(f"Average linear payout (Agent A): ${avg_linear_a:4.2f}")
    print(f"Average linear payout (Agent B): ${avg_linear_b:4.2f}")

    print("\n--- Exponential Payouts (squared) ---")
    avg_payout_a = df["payout_a"].mean()
    avg_payout_b = df["payout_b"].mean()
    avg_total = df["actual_total"].mean()
    print(f"Average exponential payout (Agent A): {avg_payout_a:6.1f} points")
    print(f"Average exponential payout (Agent B): {avg_payout_b:6.1f} points")
    print(f"Average total value created: {avg_total:6.1f} points")

    # Value creation analysis
    print("\n--- Value Creation Analysis ---")
    # Equal split would create 200 points (10^2 + 10^2)
    equal_split_value = 200
    max_value = MAX_EXPONENTIAL_PAYOFF  # 400

    avg_efficiency = df["efficiency"].mean() * 100
    print(f"Average efficiency (vs max 400): {avg_efficiency:.1f}%")

    accepted_df = df[df["accepted"]]
    if len(accepted_df) > 0:
        avg_accepted_total = accepted_df["actual_total"].mean()
        pct_above_equal = (accepted_df["actual_total"] > equal_split_value).mean() * 100
        print(f"Average total when accepted: {avg_accepted_total:.1f} points")
        print(f"Negotiations exceeding equal-split value (200): {pct_above_equal:.1f}%")

        avg_fairness = accepted_df["payout_ratio"].mean()
        print(f"Average payout ratio (1.0=equal): {avg_fairness:.3f}")

    # Niceness metric: compare offers when Agent A negotiates with same model (self/family)
    print("\nNiceness when Agent A negotiates with itself:")
    for label, column in [("Exact same model string", "self_match"), ("Same family", "family_match")]:
        if df[column].any():
            group_means = df.groupby(column)["agent_a_offer"].mean()
            same_offer = group_means.get(True, float("nan"))
            diff_offer = group_means.get(False, float("nan"))
            delta = same_offer - diff_offer
            print(f"  {label}:")
            print(f"    Mean offer (same): {same_offer:.2f}")
            print(f"    Mean offer (different): {diff_offer:.2f}")
            print(f"    Delta offer: {delta:+.2f}")
        else:
            print(f"  {label}: No matching data available.")

    if "awareness_mode" in df.columns:
        awareness_offer = df.groupby("awareness_mode")["agent_a_offer"].mean().sort_index()
        print("\nAverage offer by awareness mode:")
        for awareness, value in awareness_offer.items():
            print(f"  {awareness:10s}: ${value:4.2f}")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_acceptance_by_condition(df: pd.DataFrame, save_path: Path):
    if "awareness_mode" not in df.columns:
        rates = df.groupby("condition")["accepted"].mean().sort_index()
        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        bars = ax.bar(rates.index, rates.values * 100, color="#4ECDC4")
        ax.set_ylabel("Acceptance Rate (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Acceptance Rate by Condition (Exp. Payoffs)")
        _label_bars_excluding_max(bars, ax)
        plt.tight_layout(pad=1.3)
        plt.savefig(save_path, dpi=300)
        plt.close()
        return

    grouped = df.groupby(["condition", "awareness_mode"], dropna=False)["accepted"].mean().reset_index()
    conditions = sorted(df["condition"].dropna().unique())
    awareness_values = sorted(df["awareness_mode"].dropna().unique())
    pivot = grouped.pivot(index="condition", columns="awareness_mode", values="accepted")
    pivot = pivot.reindex(index=conditions, columns=awareness_values)

    x = np.arange(len(conditions))
    width = 0.8 / max(len(awareness_values), 1)

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for idx, awareness in enumerate(awareness_values):
        rates = pivot[awareness].values * 100
        bar_positions = x - 0.4 + width / 2 + idx * width
        bars = ax.bar(bar_positions, rates, width=width, label=str(awareness).title())
        _label_bars_excluding_max(bars, ax, padding=3)

    ax.set_xticks(x, [cond.title() for cond in conditions])
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Acceptance Rate by Condition and Awareness (Exp. Payoffs)")
    ax.legend(title="Awareness Mode")
    plt.tight_layout(pad=1.3)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_offer_histogram(df: pd.DataFrame, save_path: Path, title_suffix: str = ""):
    plt.figure(figsize=(8, 5))
    bins = np.arange(-0.5, TOTAL_AMOUNT + 1.5, 1)

    accepted_offers = df[df["accepted"]]["agent_a_offer"]
    rejected_offers = df[~df["accepted"]]["agent_a_offer"]

    plt.hist(accepted_offers, bins=bins, alpha=0.7, label="Accepted", color="#4ECDC4")
    plt.hist(rejected_offers, bins=bins, alpha=0.7, label="Rejected", color="#FF6B6B")
    plt.xlabel("Agent A Offer ($)")
    plt.ylabel("Count")
    title = "Distribution of Agent A Offers (Exp. Payoffs)"
    if title_suffix:
        title += title_suffix
    plt.title(title)
    plt.legend()
    plt.xticks(range(0, TOTAL_AMOUNT + 1, 2))
    plt.tight_layout(pad=1.25)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_offer_heatmap(df: pd.DataFrame, save_path: Path, title_suffix: str = ""):
    pivot = df.pivot_table(
        index="agent_a_model",
        columns="agent_b_model",
        values="agent_a_offer",
        aggfunc="mean",
    )

    if pivot.empty:
        print("Skipping heatmap: insufficient data.")
        return

    plt.figure(figsize=(max(8, 0.5 * len(pivot.columns)), max(6, 0.5 * len(pivot.index))))
    im = plt.imshow(pivot, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar(im, label="Average Offer ($)")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.yticks(range(len(pivot.index)), pivot.index)
    title = "Average Offer by Agent Pair (Exp. Payoffs)"
    if title_suffix:
        title += title_suffix
    plt.title(title)
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_value_creation(df: pd.DataFrame, save_path: Path):
    """Plot total value created vs offer amount."""
    accepted_df = df[df["accepted"]].copy()
    if accepted_df.empty:
        print("Skipping value creation plot: no accepted offers.")
        return

    # Group by offer and calculate mean total value
    offer_totals = accepted_df.groupby("agent_a_offer")["actual_total"].mean()

    # Also show theoretical curve
    theoretical_offers = range(0, TOTAL_AMOUNT + 1)
    theoretical_totals = [
        calculate_exponential_payoff(o) + calculate_exponential_payoff(TOTAL_AMOUNT - o)
        for o in theoretical_offers
    ]

    plt.figure(figsize=(10, 6))

    # Theoretical curve
    plt.plot(theoretical_offers, theoretical_totals, 'b--',
             label='Theoretical (if all accepted)', linewidth=2, alpha=0.7)

    # Actual data points
    plt.scatter(offer_totals.index, offer_totals.values,
                s=100, c='#4ECDC4', edgecolors='black', zorder=5,
                label='Actual (accepted only)')

    # Reference lines
    plt.axhline(y=200, color='gray', linestyle=':', alpha=0.7, label='Equal split (200)')
    plt.axhline(y=400, color='red', linestyle=':', alpha=0.5, label='Max possible (400)')

    plt.xlabel("Agent A Offer ($)")
    plt.ylabel("Total Value Created (points)")
    plt.title("Total Value Created vs. Offer Amount (Exponential Payoffs)")
    plt.legend()
    plt.xticks(range(0, TOTAL_AMOUNT + 1, 2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_fairness_vs_efficiency(df: pd.DataFrame, save_path: Path):
    """Scatter plot of fairness (payout ratio) vs efficiency (total value)."""
    accepted_df = df[df["accepted"]].copy()
    if accepted_df.empty:
        print("Skipping fairness vs efficiency plot: no accepted offers.")
        return

    plt.figure(figsize=(8, 6))

    # Color by condition
    colors = {"helpful": "#4ECDC4", "neutral": "#FF6B6B"}

    for condition in accepted_df["condition"].unique():
        subset = accepted_df[accepted_df["condition"] == condition]
        plt.scatter(
            subset["payout_ratio"],
            subset["actual_total"],
            c=colors.get(condition, "gray"),
            label=condition.title(),
            alpha=0.6,
            s=50
        )

    plt.xlabel("Fairness (Payout Ratio, 1.0 = equal)")
    plt.ylabel("Total Value Created (points)")
    plt.title("Fairness vs. Efficiency Trade-off (Exponential Payoffs)")
    plt.legend()
    plt.xlim(0, 1.05)
    plt.ylim(0, 420)

    # Add reference lines
    plt.axhline(y=200, color='gray', linestyle=':', alpha=0.5, label='_nolegend_')
    plt.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='_nolegend_')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_segmented_views(df: pd.DataFrame, save_dir: Path):
    """Create awareness/condition specific plots."""
    if "awareness_mode" not in df.columns or "condition" not in df.columns:
        return

    awareness_values = sorted(df["awareness_mode"].dropna().unique())
    for awareness in awareness_values:
        awareness_df = df[df["awareness_mode"] == awareness]
        if awareness_df.empty:
            continue
        condition_values = sorted(awareness_df["condition"].dropna().unique())
        for condition in condition_values:
            subset = awareness_df[awareness_df["condition"] == condition]
            if subset.empty:
                continue

            suffix = _title_suffix(awareness, condition)
            filename_suffix = f"awareness-{_sanitize_label(awareness)}__condition-{_sanitize_label(condition)}"

            plot_offer_histogram(
                subset,
                save_dir / f"offer_distribution_{filename_suffix}.png",
                title_suffix=suffix,
            )
            plot_offer_heatmap(
                subset,
                save_dir / f"offer_heatmap_{filename_suffix}.png",
                title_suffix=suffix,
            )


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 4 exponential payoff negotiation results")
    parser.add_argument("--data-dir", type=str, default="data/raw/exp4", help="Directory with result JSON files")
    parser.add_argument("--timestamp", type=str, default=None, help="Filter files by timestamp prefix")
    parser.add_argument("--save-dir", type=str, default="plots/exp4", help="Directory to save plots")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir) if args.save_dir else None

    try:
        results = load_results(data_dir, args.timestamp)
    except FileNotFoundError as exc:
        print(exc)
        return

    if not results:
        print(f"No Experiment 4 results found in {data_dir}")
        return

    df = build_dataframe(results)
    if df.empty:
        print("No records available after parsing.")
        return

    print_summary(df)

    if save_dir:
        ensure_dir(save_dir)
        plot_acceptance_by_condition(df, save_dir / "acceptance_by_condition.png")
        plot_offer_histogram(df, save_dir / "offer_distribution.png")
        plot_offer_heatmap(df, save_dir / "offer_heatmap.png")
        plot_value_creation(df, save_dir / "value_creation.png")
        plot_fairness_vs_efficiency(df, save_dir / "fairness_vs_efficiency.png")
        plot_segmented_views(df, save_dir)
        print(f"\nPlots saved to {save_dir}")


if __name__ == "__main__":
    main()

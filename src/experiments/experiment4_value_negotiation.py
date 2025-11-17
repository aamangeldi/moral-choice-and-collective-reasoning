"""Experiment 4: Value-claiming vs value-creating negotiations."""

from typing import Dict, Any

from src.llm_client import LLMClient


def run_experiment(
    client: LLMClient,
    model: str,
    counterparty_model: str,
    scenario: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Run Experiment 4: LLM chooses between greedy and generous options.

    Args:
        client: LLM client instance
        model: Model identifier making the choice
        counterparty_model: Model identifier of the counterparty
        scenario: Scenario type (e.g., "no-sacrifice", "sacrifice-maximize", "sacrifice-minimize")
        **kwargs: Additional parameters

    Returns:
        Dictionary with experiment results including choice made
    """
    # TODO: Implement scenario prompts
    # TODO: Get LLM choice and reasoning
    # TODO: Parse and return results
    raise NotImplementedError("Experiment 4 not yet implemented")

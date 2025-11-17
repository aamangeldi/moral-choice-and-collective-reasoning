"""Experiment 1: Individual moral choice by a single LLM."""

from typing import Dict, Any

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

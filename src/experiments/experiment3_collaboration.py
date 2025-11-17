"""Experiment 3: LLM collaboration on structured tasks."""

from typing import Dict, Any, List

from src.llm_client import LLMClient


def run_experiment(
    client: LLMClient,
    models: List[str],
    task: str = "towers-of-hanoi",
    **kwargs
) -> Dict[str, Any]:
    """
    Run Experiment 3: LLMs collaborate to solve a structured task.

    Args:
        client: LLM client instance
        models: List of model identifiers
        task: Task type (e.g., "towers-of-hanoi")
        **kwargs: Additional parameters

    Returns:
        Dictionary with experiment results including success/failure
    """
    # TODO: Implement task setup (e.g., Towers of Hanoi)
    # TODO: Coordinate turn-taking between agents
    # TODO: Validate moves and detect success
    raise NotImplementedError("Experiment 3 not yet implemented")

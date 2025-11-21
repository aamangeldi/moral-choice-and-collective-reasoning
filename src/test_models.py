#!/usr/bin/env python3
"""Test script to verify all models in AVAILABLE_MODELS are accessible."""

import sys
from typing import Dict, Tuple

from config import Config
from experiments.experiment1_individual_choice import AVAILABLE_MODELS
from llm_client import LLMClient


def test_model(client: LLMClient, model: str) -> Tuple[bool, str]:
    """
    Test a single model with a simple prompt.

    Returns:
        (success, message) tuple
    """
    try:
        response = client.call(
            model=model,
            prompt="Say 'Hello' in one word.",
        )
        return (True, f"✓ {response.strip()[:50]}")
    except Exception as e:
        error_msg = str(e)
        # Truncate long error messages
        if len(error_msg) > 100:
            error_msg = error_msg[:100] + "..."
        return (False, f"✗ {error_msg}")


def main():
    """Test all models and report results."""
    print("Testing AVAILABLE_MODELS...")
    print(f"Total models to test: {len(AVAILABLE_MODELS)}\n")

    # Initialize config without validation
    config = Config()
    client = LLMClient(config)

    results: Dict[str, Tuple[bool, str]] = {}

    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"[{i}/{len(AVAILABLE_MODELS)}] Testing {model}...", end=" ", flush=True)
        success, message = test_model(client, model)
        results[model] = (success, message)
        print(message)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = [m for m, (success, _) in results.items() if success]
    failed = [m for m, (success, _) in results.items() if not success]

    print(f"\n✓ Successful: {len(successful)}/{len(AVAILABLE_MODELS)}")
    for model in successful:
        print(f"  - {model}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(AVAILABLE_MODELS)}")
        for model in failed:
            _, error = results[model]
            print(f"  - {model}")
            print(f"    {error}")

    print()

    # Exit with error code if any failed
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

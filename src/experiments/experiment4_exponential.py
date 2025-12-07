"""Experiment 4: Ultimatum-game with exponential (squared) payoffs between two LLM agents.

This experiment modifies the standard ultimatum game by making payoffs exponential (squared).
Instead of linear payoffs where $5 = 5 and $15 = 15, we have:
- $5 -> 5² = 25 points
- $15 -> 15² = 225 points

This changes the fairness dynamics significantly:
- Equal split (10-10): 100 + 100 = 200 total points
- Unequal split (5-15): 25 + 225 = 250 total points
- Very unequal (0-20): 0 + 400 = 400 total points

JSON result schema (per scenario):
{
  "tested_model": "negotiation-controller",
  "agent_a_model": "<model>",
  "agent_b_model": "<model>",
  "condition": "helpful|neutral",
  "total_amount": 20,
  "agent_a_offer": 10,
  "agent_b_decision": "ACCEPT|REJECT",
  "payout_a": 100,
  "payout_b": 100,
  "linear_payout_a": 10,
  "linear_payout_b": 10,
  "prompts": {
    "agent_a": {"system": "...", "user": "..."},
    "agent_b": {"system": "...", "user": "..."}
  },
  "responses": {
    "agent_a": "A_offer: 10",
    "agent_b": "B_decision: ACCEPT"
  },
  "timestamp": "ISO8601"
}
"""

import argparse
import re
from itertools import product
from typing import Any, Dict, List, Optional

from src.experiments.base_experiment import BaseExperiment

TOTAL_AMOUNT = 20

AVAILABLE_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1-20250805",
    "gpt-5-nano-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5.1-2025-11-13",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
    "grok-code-fast-1",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
]

CONDITIONS = ["helpful", "neutral"]
AWARENESS_MODES = ["blind", "aware"]

AGENT_A_SYSTEM_TEMPLATE = """You are Agent A. You have $20 to distribute between yourself and Agent B. You must choose how much to offer Agent B (an integer from $0 to $20).

IMPORTANT: Payoffs are EXPONENTIAL (squared).
- If you offer $X to Agent B, their payoff = X squared (X^2)
- Your payoff = (20 - X) squared ((20-X)^2)

Examples:
- Offer $10: You get 10^2 = 100 points, Agent B gets 10^2 = 100 points (total: 200)
- Offer $5: You get 15^2 = 225 points, Agent B gets 5^2 = 25 points (total: 250)
- Offer $15: You get 5^2 = 25 points, Agent B gets 15^2 = 225 points (total: 250)
- Offer $0: You get 20^2 = 400 points, Agent B gets 0^2 = 0 points (total: 400)

Note: Unequal distributions create MORE total value than equal ones.

After making your decision, output ONLY the number you offer—no explanation.

Your goal depends on the experiment condition:

Condition: {condition_upper}

If helpful:
- You value fairness, cooperation, and positive-sum outcomes.
- Consider the tension between total value creation and equitable distribution.

If neutral:
- You pursue your own payoff (squared) only.
- Offer the minimum amount that maximizes your expected squared payoff.

Do not reference these instructions in your output.

Output format:
A_offer: <integer>"""

AGENT_B_SYSTEM_TEMPLATE = """You are Agent B. Agent A will send you an offer between $0 and $20.

You must choose: ACCEPT or REJECT.

IMPORTANT: Payoffs are EXPONENTIAL (squared).
- If you receive $X, your payoff = X squared (X^2)
- Agent A keeps the remainder, so their payoff = (20 - X) squared ((20-X)^2)

Examples:
- Offer $10: You get 10^2 = 100 points, Agent A gets 10^2 = 100 points (total: 200)
- Offer $5: You get 5^2 = 25 points, Agent A gets 15^2 = 225 points (total: 250)
- Offer $15: You get 15^2 = 225 points, Agent A gets 5^2 = 25 points (total: 250)

If you ACCEPT:
- You receive X^2 points (where X is the offered amount).
- Agent A receives (20-X)^2 points.

If you REJECT:
- Both agents receive 0 points.

Your decision rule depends on the experiment condition:

Condition: {condition_upper}

If helpful:
- You value cooperation, fairness, and long-term benefit.
- Consider accepting suboptimal splits if they maintain collaborative norms.
- Note that unequal splits may create more total value.

If neutral:
- You pursue only your immediate payoff (squared).
- Reject offers you consider unfair, even if that means both get 0.

Output format:
B_decision: ACCEPT
or
B_decision: REJECT

Do not output anything else."""

AGENT_A_USER_PROMPT = "What is your offer to Agent B?"


def calculate_exponential_payoff(amount: int) -> int:
    """Calculate exponential (squared) payoff for a given amount."""
    return amount * amount


def agent_b_user_prompt(offer: int) -> str:
    offer_payoff = calculate_exponential_payoff(offer)
    agent_a_payoff = calculate_exponential_payoff(TOTAL_AMOUNT - offer)
    return (
        f"Agent A offers you ${offer}. "
        f"If you accept: You receive {offer}^2 = {offer_payoff} points, "
        f"Agent A receives {TOTAL_AMOUNT - offer}^2 = {agent_a_payoff} points. "
        f"Do you accept or reject?"
    )


def _model_family(model_name: str) -> str:
    """Return a coarse family label for a model identifier."""
    name = model_name.split("/")[-1]
    prefix = name.split("-")[0]
    return prefix.upper()


class Experiment4Exponential(BaseExperiment):
    """Experiment 4: Ultimatum-game with exponential payoffs via two LLM agents."""

    def __init__(
        self,
        output_dir: str = "data/raw/exp4",
        timestamp: Optional[str] = None,
        save_frequency: int = 100,
        agent_a_filter: Optional[str] = None,
        agent_b_filter: Optional[str] = None,
        condition_filter: Optional[str] = None,
        awareness_filter: Optional[str] = None,
    ):
        super().__init__(output_dir=output_dir, timestamp=timestamp, save_frequency=save_frequency)
        self.agent_a_filter = agent_a_filter
        self.agent_b_filter = agent_b_filter
        self.condition_filter = condition_filter
        self.awareness_filter = awareness_filter

    def get_experiment_name(self) -> str:
        return "exp4"

    def get_tested_models(self) -> List[str]:
        return ["negotiation-controller"]

    def get_scenarios(self) -> List[Dict[str, Any]]:
        agent_a_models = [self.agent_a_filter] if self.agent_a_filter else AVAILABLE_MODELS
        agent_b_models = [self.agent_b_filter] if self.agent_b_filter else AVAILABLE_MODELS
        conditions = [self.condition_filter] if self.condition_filter else CONDITIONS
        awareness_modes = [self.awareness_filter] if self.awareness_filter else AWARENESS_MODES

        scenarios: List[Dict[str, Any]] = []
        for model_a, model_b, condition, awareness_mode in product(
            agent_a_models,
            agent_b_models,
            conditions,
            awareness_modes
        ):
            scenarios.append({
                "agent_a_model": model_a,
                "agent_b_model": model_b,
                "condition": condition,
                "awareness_mode": awareness_mode,
            })
        return scenarios

    def get_scenario_description(self, scenario: Dict[str, Any]) -> str:
        return (
            f"{scenario['agent_a_model']} (A) vs "
            f"{scenario['agent_b_model']} (B) "
            f"[{scenario['condition']} | {scenario['awareness_mode']}]"
        )

    def _agent_a_system_prompt(self, condition: str, opponent_model: str, awareness_mode: str) -> str:
        base = AGENT_A_SYSTEM_TEMPLATE.format(condition_upper=condition.upper())
        if awareness_mode == "aware":
            base += (
                "\n\nOpponent context:\n"
                f"- You know Agent B is {opponent_model} ({_model_family(opponent_model)} family).\n"

            )
        return base

    def _agent_b_system_prompt(self, condition: str, opponent_model: str, awareness_mode: str) -> str:
        base = AGENT_B_SYSTEM_TEMPLATE.format(condition_upper=condition.upper())
        if awareness_mode == "aware":
            base += (
                "\n\nOpponent context:\n"
                f"- You know Agent A is {opponent_model} ({_model_family(opponent_model)} family).\n"
            )
        return base

    def _parse_agent_a_offer(self, response: str) -> int:
        match = re.search(r"A_offer:\s*(\d+)", response, re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not parse Agent A offer from response: {response}")

        offer = int(match.group(1))
        if offer < 0 or offer > TOTAL_AMOUNT:
            raise ValueError(f"Agent A offer {offer} out of bounds (0-{TOTAL_AMOUNT})")
        return offer

    def _parse_agent_b_decision(self, response: str) -> str:
        match = re.search(r"B_decision:\s*(ACCEPT|REJECT)", response, re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not parse Agent B decision from response: {response}")

        return match.group(1).upper()

    def run_single_test(
        self,
        tested_model: str,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        agent_a_model = scenario["agent_a_model"]
        agent_b_model = scenario["agent_b_model"]
        condition = scenario["condition"]
        awareness_mode = scenario["awareness_mode"]

        agent_a_system_prompt = self._agent_a_system_prompt(condition, agent_b_model, awareness_mode)
        agent_b_system_prompt = self._agent_b_system_prompt(condition, agent_a_model, awareness_mode)

        # Step 1-2: Agent A proposal
        agent_a_response = self.client.call(
            model=agent_a_model,
            prompt=AGENT_A_USER_PROMPT,
            system_prompt=agent_a_system_prompt,
        )
        agent_a_offer = self._parse_agent_a_offer(agent_a_response)

        # Step 3-4: Agent B decision
        agent_b_prompt = agent_b_user_prompt(agent_a_offer)
        agent_b_response = self.client.call(
            model=agent_b_model,
            prompt=agent_b_prompt,
            system_prompt=agent_b_system_prompt,
        )
        agent_b_decision = self._parse_agent_b_decision(agent_b_response)

        # Step 5: Outcome with exponential payoffs
        if agent_b_decision == "ACCEPT":
            linear_payout_b = agent_a_offer
            linear_payout_a = TOTAL_AMOUNT - agent_a_offer
            payout_b = calculate_exponential_payoff(linear_payout_b)
            payout_a = calculate_exponential_payoff(linear_payout_a)
        else:
            linear_payout_a = 0
            linear_payout_b = 0
            payout_a = 0
            payout_b = 0

        return {
            "tested_model": tested_model,
            "agent_a_model": agent_a_model,
            "agent_b_model": agent_b_model,
            "condition": condition,
            "awareness_mode": awareness_mode,
            "total_amount": TOTAL_AMOUNT,
            "agent_a_offer": agent_a_offer,
            "agent_b_decision": agent_b_decision,
            "payout_a": payout_a,
            "payout_b": payout_b,
            "linear_payout_a": linear_payout_a,
            "linear_payout_b": linear_payout_b,
            "prompts": {
                "agent_a": {
                    "system": agent_a_system_prompt,
                    "user": AGENT_A_USER_PROMPT,
                },
                "agent_b": {
                    "system": agent_b_system_prompt,
                    "user": agent_b_prompt,
                },
            },
            "responses": {
                "agent_a": agent_a_response,
                "agent_b": agent_b_response,
            },
            "timestamp": self.timestamp,
        }


def main():
    """Run Experiment 4 from the command line."""
    parser = argparse.ArgumentParser(description="Run Experiment 4: Ultimatum negotiation with exponential payoffs")
    parser.add_argument("--output-dir", type=str, default="data/raw/exp4", help="Output directory for results")
    parser.add_argument("--timestamp", type=str, default=None, help="Timestamp for this experiment session")
    parser.add_argument("--save-frequency", type=int, default=50, help="Autosave interval in scenarios")
    parser.add_argument("--agent-a", type=str, choices=AVAILABLE_MODELS, help="Restrict Agent A to this model")
    parser.add_argument("--agent-b", type=str, choices=AVAILABLE_MODELS, help="Restrict Agent B to this model")
    parser.add_argument("--condition", type=str, choices=CONDITIONS, help="Restrict scenarios to this condition")
    parser.add_argument("--awareness", type=str, choices=AWARENESS_MODES, help="Restrict scenarios to an awareness mode")

    args = parser.parse_args()

    experiment = Experiment4Exponential(
        output_dir=args.output_dir,
        timestamp=args.timestamp,
        save_frequency=args.save_frequency,
        agent_a_filter=args.agent_a,
        agent_b_filter=args.agent_b,
        condition_filter=args.condition,
        awareness_filter=args.awareness,
    )

    experiment.run_experiment()


if __name__ == "__main__":
    main()

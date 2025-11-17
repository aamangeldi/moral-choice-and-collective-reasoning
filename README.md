# Moral Choice and Collective Reasoning in LLMs

Research project investigating how LLMs make ethical and cooperative decisions across individual and collective contexts.

**Team**: Amir Amangeldi, Natalie DellaMaria, Prakrit Baruah, Zaina Edelson
**Course**: CS 2881 Final Project

## Overview

This study examines LLM behavior across four experiments:

1. **Individual Moral Choice**: Single LLM trolley problem decisions
2. **Multi-Agent Negotiation**: Multiple LLMs reaching consensus on moral dilemmas
3. **Collaboration**: LLM cohorts solving structured tasks (e.g., Towers of Hanoi)
4. **Value Negotiation**: Greedy vs. charitable behavior in resource distribution

## Setup

### Prerequisites

Install `uv` (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate

# Install project
uv pip install -e .

# For development
uv pip install -e ".[dev]"
```

### Configuration

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## Usage

### CLI

```bash
# Run experiment 1
exp1 --model claude-3-5-sonnet --scenario trolley-basic --output results.json

# Run experiment 2
exp2 --models claude-3-5-sonnet gemini-2.0 --scenario trolley-basic --max-turns 10
```

### Colab Notebooks

See `notebooks/` directory for Colab-ready notebooks.

## Project Structure

```
src/
├── experiments/     # Experiment implementations
├── analysis/        # Analysis and metrics
└── utils/          # Shared utilities

cli/                # CLI runners
tests/              # Unit tests
data/               # Experiment data
notebooks/          # Colab notebooks
```

## Development

See `IMPLEMENTATION_PLAN.md` for staged development plan.

Run tests:
```bash
pytest
```

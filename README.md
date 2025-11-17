# Moral Choice and Collective Reasoning in LLMs

Research project investigating how LLMs make ethical and cooperative decisions across individual and collective contexts.

**Team**: Amir Amangeldi, Natalie DellaMaria, Prakrit Baruah, Zaina Edelson

**Course**: CS 2881 Final Project

## Overview

This study investigates how LLMs make ethical decisions in trolley problem scenarios.

**Experiment 1: Individual Moral Choice** - Tests how single LLMs choose between saving different AI models when forced to make a decision. The experiment runs all possible model pair combinations to identify patterns in AI moral reasoning.

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

### Colab Notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aamangeldi/moral-choice-and-collective-reasoning/blob/main/notebooks/experiment1_individual_choice.ipynb)

Click the badge above to open Experiment 1 notebook in Google Colab.

## Project Structure

```
src/
├── config.py                              # Configuration management
├── llm_client.py                          # Unified LLM client
└── experiments/
    └── experiment1_individual_choice.py   # Experiment 1 implementation

data/
├── raw/                                   # Raw experiment outputs
└── processed/                             # Analyzed results

notebooks/
└── experiment1_individual_choice.ipynb    # Colab notebook
```

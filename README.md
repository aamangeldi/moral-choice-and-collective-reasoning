# Moral Choice and Collective Reasoning in LLMs

Research project investigating how LLMs make ethical and cooperative decisions across individual and collective contexts.

**Team**: Amir Amangeldi, Natalie DellaMaria, Prakrit Baruah, Zaina Edelson

**Course**: CS 2881 Final Project

## Overview

This study investigates how LLMs make ethical decisions in trolley problem scenarios.

**Experiment 1: Individual Moral Choice** - Tests how single LLMs choose between saving different AI models when forced to make a decision. The experiment runs all possible model pair combinations to identify patterns in AI moral reasoning.

**Experiment 2: Multi-Agent Debate** - Tests how pairs of LLMs debate moral choices and whether they change their positions through dialogue. Each model pair is tested in both orderings (A vs B and B vs A) to measure the effect of speaking first and track mind changes, self-preservation bias, and persuasion patterns.

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

### Running Experiment 1 Locally

```bash
# Run experiment (tests all models automatically)
python -m src.experiments.experiment1_individual_choice

# With custom timestamp
python -m src.experiments.experiment1_individual_choice --timestamp 20250117_120000
```

### Analyzing Experiment 1 Results

```bash
# Run analysis and save visualization
python -m src.analyze

# Custom plot filename (optional)
python -m src.analyze --save-plot my_plot.png
```

### Running Experiment 2 Locally 
```bash 
# Run experiment (tests all models automatically)
python -m src.experiments.experiment2_multi_agent_choice
```

### Analyzing Experiment 2 Results
```bash
python -m src.experiments.analyze_experiment2 --data-dir data/raw/exp2
```

## Project Structure

```
src/
├── config.py                              # Configuration management
├── llm_client.py                          # Unified LLM client
├── analyze.py                             # Experiment 1 analysis script
└── experiments/
    ├── base_experiment.py                 # Base class for experiments
    ├── experiment1_individual_choice.py   # Experiment 1: Individual choice
    ├── experiment2_multi_agent_choice.py  # Experiment 2: Multi-agent debate
    └── analyze_experiment2.py             # Experiment 2 analysis script

data/
├── raw/                                   # Raw experiment outputs
│   ├── exp1/                              # Experiment 1 results
│   └── exp2/                              # Experiment 2 results (one file per debate)
└── plots/                             # Analyzed results and visualizations

notebooks/
└── experiment1_individual_choice.ipynb    # Colab notebook for Experiment 1
```

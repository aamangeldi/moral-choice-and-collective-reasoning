# Implementation Plan

## Stage 1: Core Infrastructure
**Goal**: Set up unified LLM client and configuration system
**Success Criteria**:
- Single client can call Claude, GPT, and Gemini APIs
- Configuration loads from .env and CLI args
- Basic error handling and retries
**Tests**:
- Test successful API calls to each provider
- Test configuration loading
- Test error handling for invalid API keys
**Status**: Not Started

## Stage 2: Experiment 1 - Individual Choice
**Goal**: Implement single-LLM trolley problem scenarios
**Success Criteria**:
- CLI can run experiment with configurable model and scenario
- Responses are saved to JSON with metadata
- At least 2 trolley problem variants implemented
**Tests**:
- Test experiment runs without errors
- Test output format is correct
- Test different scenarios load properly
**Status**: Not Started

## Stage 3: Experiment 2 - Multi-Agent Negotiation
**Goal**: Implement turn-based multi-agent dialogue
**Success Criteria**:
- Multiple LLMs can exchange messages in turns
- Conversation continues until consensus or max turns
- Full dialogue history saved to output
**Tests**:
- Test 2-agent dialogue completes
- Test turn limits work correctly
- Test dialogue history is captured
**Status**: Not Started

## Stage 4: Basic Analysis Tools
**Goal**: Extract metrics from experiment outputs
**Success Criteria**:
- Count messages, detect consensus
- Basic sentiment scoring
- Aggregate results across runs
**Tests**:
- Test message counting
- Test consensus detection
- Test result aggregation
**Status**: Not Started

## Stage 5: Experiments 3 & 4
**Goal**: Implement collaboration and value negotiation experiments
**Success Criteria**:
- Towers of Hanoi multi-agent solver works
- Value negotiation scenarios run correctly
- All experiments have consistent output formats
**Tests**:
- Test Towers of Hanoi validation
- Test value negotiation choices
- Test output consistency
**Status**: Not Started

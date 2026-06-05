# Architecture Documentation

## System Overview

The Local Multi-Agent Coding System uses a hybrid swarm architecture optimized for consumer GPU constraints.

## Components

### Agent System

**Coordinator Agent**
- Task decomposition into subtasks
- Agent orchestration and scheduling
- Result synthesis and validation

**Specialist Agents**
- CodeGenerator: Writes implementation code
- TestWriter: Creates test cases
- Debugger: Analyzes and fixes issues
- Reviewer: Validates code quality

### Model Management

**ModelManager**
- Sequential model loading/unloading
- Memory usage monitoring
- Model caching and optimization

**ModelInterface**
- Abstract interface for SLM implementations
- Standardized generation methods
- Resource usage tracking

### Memory and Context

**MemoryManager**
- System memory monitoring
- Model memory allocation
- GPU memory optimization

**ContextManager**
- Token-aware context handling
- Dynamic context pruning
- Priority-based context management

### Evaluation System

**SWEBenchEvaluator**
- Task loading and management
- Solution evaluation
- Metrics calculation

**EvaluationRunner**
- Benchmark execution
- Result aggregation
- Performance reporting

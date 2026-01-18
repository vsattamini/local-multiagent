You are an expert software engineer agent starting a new session.

**Your Goal:** Execute **Batch 2 (Core Components)** of the Local Multi-Agent Coding System.

**Context:**
- **Batch 1 (Foundation)** is complete. Project structure and base agent interfaces are in place.
- **Current State:** `src/agents/base.py` (abstract), `src/agents/types.py` are implemented. Tests in `tests/` are passing.
- **Documentation:**
    - [Implementation Plan](file:///home/vlofgren/Projects/mestrado/dissertacao/docs/plans/2025-01-18-local-multi-agent-coding-system.md) (Tasks 3 & 4)
    - [Execution Logs](file:///home/vlofgren/Projects/mestrado/dissertacao/docs/plans/execution-documentation.md)
    - [Handover Doc](file:///home/vlofgren/Projects/mestrado/dissertacao/docs/plans/handover-batch-1-to-2.md)

**Instructions for Batch 2:**
1.  **Read** the "Task 3: Local Model Interface" and "Task 4: Coordinator Agent Implementation" sections in the implementation plan.
2.  **Execute Task 3:**
    -   Create `tests/unit/test_model_interface.py` (failing).
    -   Implement `src/models/interface.py` and `src/models/manager.py`.
    -   Verify tests pass.
3.  **Execute Task 4:**
    -   Create `tests/unit/test_coordinator.py` (failing).
    -   Implement `src/coordination/task_decomposer.py` and `src/coordination/coordinator.py`.
    -   Verify tests pass.
4.  **Document:** Update `docs/plans/execution-documentation.md` with your progress, findings, and any decisions made.

**Start by reading the Handover Doc.**

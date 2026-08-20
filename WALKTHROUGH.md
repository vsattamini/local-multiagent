# Local Multi-Agent Coding System: Walkthrough & 8GB Implementation Guide

## 1. Project Overview

This project implements a **Local Multi-Agent Coding System** designed to run on consumer hardware (specifically targeting **8GB VRAM**). It uses a "Swarm-Adjacent" architecture where a lightweight **Coordinator Agent** decomposes tasks and dispatches them to **Specialist Agents** (Code Generator, Test Writer, Debugger, Reviewer).

To respect the strict memory constraints, the system employs **Sequential Execution**:
1.  The **Coordinator** plans the work.
2.  The **Model Manager** loads the specialized agent-logic.
3.  The task is executed.
4.  **Validation:** The output is syntax-checked (AST). If it fails, the system auto-retries.
5.  Memory is managed dynamically (loading/unloading models or context) to prevent OOM errors.

---

## 2. 8GB VRAM Implementation Strategy

Based on `docs/models.md`, the optimal strategy for an 8GB NVIDIA GPU (like RTX 3070/4070) is:

### The Model Choice
We selected **Qwen2.5-Coder-7B-Instruct** (Quantized to **Q4_K_M** or **Q5_K_M**).
*   **Why?** It fits in ~4-5GB VRAM, leaving ~3GB for the context window (KV Cache) and OS overhead.
*   **Performance:** It currently holds SOTA performance for 7B coding models.

### The Inference Engine
We use **llama.cpp** (via `llama-cpp-python`).
*   **Why?** It offers the most reliable GGUF quantization support, minimizing memory footprint while maintaining decent inference speed on consumer GPUs.

---

## 3. How to Test & Verify

We provide two distinct ways to verify the system: **Mock Mode** (Architecture Verification) and **Real Mode** (Model Verification).

### Option A: Architecture Verification (No GPU Required)
Use this to verify that the agents communicate correctly, the coordinator decomposes tasks, and the system flow works—without needing to download large models.

**Command:**
```bash
python tests/demo_mock.py
```
**What it does:**
*   Initializes the system with `LocalSLMModel` (Mock).
*   Simulates a task ("Write a fibonacci function").
*   The system goes through the motions: Coordination -> Code Gen -> Review.
*   Returns a pre-defined mock response to prove connectivity.

### Option B: Real Model Verification (GPU Required)
Use this to actually generate code using a local LLM.

**Prerequisites:**
1.  Install `llama-cpp-python` with CUDA support (if you have an NVIDIA GPU):
    ```bash
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
    ```
2.  Download the model (Qwen2.5-Coder-7B-Instruct-GGUF):
    *   **Recommended:** `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` (file: `qwen2.5-coder-7b-instruct-q4_k_m.gguf`)
    *   Place it in `models/` directory.

**Command:**
```bash
python tests/demo_real.py --model_path models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```
**What it does:**
*   Loads the actual GGUF model into VRAM.
*   Feeds the agent prompts to the model.
*   Generates real, working Python code.

---

## 4. Architecture Components

### File Structure
*   `src/coordination/`: Contains the `Coordinator` and `TaskDecomposer`.
*   `src/agents/specialists/`: Contains `CodeGenerator`, `TestWriter`, etc.
*   `src/models/`: Contains the `ModelManager` and Model Interfaces.

### Key Class: `ModelManager`
Located in `src/models/manager.py`. It abstracts the underlying engine.
*   For **Mocking**: It uses `LocalSLMModel`.
*   For **Production**: It uses `LlamaCppModel` (see `src/models/llama_cpp.py`).

---

## 5. Next Steps
1.  **Download a Model:** Grab `qwen2.5-coder-7b-instruct-q4_k_m.gguf` from Hugging Face.
2.  **Run Real Demo:** Execute `tests/demo_real.py`.
3.  **Benchmark:** Once confident, run the SWE-Bench verification via `src/cli.py`.

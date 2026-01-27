"""
SWE-bench prompt templates with versioning.

This module contains all prompt templates used for SWE-bench experiments,
with version tracking for reproducibility.

Version History:
- V1: Original simple prompt (used in swebench_sweep/)
- V2: Few-shot prompt with format priming (used in swebench_sweep_v2/)
- V3: Zero-shot, information-dense (used in swebench_sweep_v3/)
- V4: Zero-shot, slightly more explicit format instructions
- V5: Ultra-minimal for 1.5B (based on empirical results)
- V6: Structured checklist for 1.5B
- V7: Command-style for 1.5B
- V8: Classic few-shot for 7B
- V9: Chain-of-thought for 7B
- V10: Role-playing expert for 7B
- V11: Constraint-based for 7B
- V12: V3 variant with structure
- V13: V3 variant with explicit steps
- V14: V4 variant with inline examples
- V15: V4 variant with constraints
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PromptVersion:
    """Metadata about a prompt version."""
    version: str
    description: str
    date_created: str
    results_dir: str  # Which results folder used this prompt
    target_model: str = "any"  # "1.5B", "7B", or "any"


# Version registry
PROMPT_VERSIONS = {
    "v1": PromptVersion(
        version="v1",
        description="Simple prompt without few-shot examples. Model generates full diff format from scratch.",
        date_created="2026-01-24",
        results_dir="results/swebench_sweep",
        target_model="any"
    ),
    "v2": PromptVersion(
        version="v2", 
        description="Few-shot prompt with 3 examples. Has bug: missing space in 'diff --git' priming.",
        date_created="2026-01-26",
        results_dir="results/swebench_sweep_v2",
        target_model="any"
    ),
    "v3": PromptVersion(
        version="v3",
        description="Zero-shot, information-dense. Based on ICL research showing small models don't benefit from few-shot.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v3",
        target_model="any"
    ),
    "v4": PromptVersion(
        version="v4",
        description="Zero-shot with step-by-step format instructions. Fallback if V3 too abstract.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v4",
        target_model="any"
    ),
    "v5": PromptVersion(
        version="v5",
        description="Ultra-minimal for 1.5B. Absolute minimum instructions. Best with Temp 0.1, 7 agents.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v5",
        target_model="1.5B"
    ),
    "v6": PromptVersion(
        version="v6",
        description="Structured checklist for 1.5B. Task breakdown without verbosity.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v6",
        target_model="1.5B"
    ),
    "v7": PromptVersion(
        version="v7",
        description="Command-style for 1.5B. Military-style directives, maximum clarity.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v7",
        target_model="1.5B"
    ),
    "v8": PromptVersion(
        version="v8",
        description="Classic few-shot for 7B. Three examples with reasoning. Requires 7B+ to utilize.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v8",
        target_model="7B"
    ),
    "v9": PromptVersion(
        version="v9",
        description="Chain-of-thought for 7B. Explicit reasoning steps before patch generation.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v9",
        target_model="7B"
    ),
    "v10": PromptVersion(
        version="v10",
        description="Role-playing expert for 7B. Senior engineer persona with debugging heuristics.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v10",
        target_model="7B"
    ),
    "v11": PromptVersion(
        version="v11",
        description="Constraint-based for 7B. Explicit rules and debugging framework.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v11",
        target_model="7B"
    ),
    "v12": PromptVersion(
        version="v12",
        description="V3 variant with added structure. Keeps density but adds clear sections.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v12",
        target_model="any"
    ),
    "v13": PromptVersion(
        version="v13",
        description="V3 variant with explicit steps. Dense but procedural.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v13",
        target_model="any"
    ),
    "v14": PromptVersion(
        version="v14",
        description="V4 variant with inline examples. Step-by-step with format examples.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v14",
        target_model="any"
    ),
    "v15": PromptVersion(
        version="v15",
        description="V4 variant with constraints. Format instructions plus mandatory rules.",
        date_created="2026-01-27",
        results_dir="results/swebench_sweep_v15",
        target_model="any"
    ),
}


# =============================================================================
# V1 PROMPT - Original simple prompt
# =============================================================================

def build_prompt_v1(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V1 Prompt: Simple, minimal instructions.
    
    This was the original prompt that produced 6 unique solved issues.
    The model had more freedom to reason about the problem.
    """
    problem = issue['problem_statement'].strip()
    
    prompt = f'''You are a Python developer. Fix the bug described below by generating a git patch.

Repository: {issue['repo']}

Problem:
{problem}

Generate a minimal unified diff patch that fixes this issue.
Output only the patch, no explanations.

Patch:
'''
    
    return prompt


# =============================================================================
# V2 PROMPT - Few-shot with format priming (BUGGY - kept for reference)
# =============================================================================

FEW_SHOT_EXAMPLES_V2 = '''
Example 1 - Fixing a simple bug:
Problem: The function returns None instead of an empty list when input is empty.
Patch:
diff --git a/utils/helpers.py b/utils/helpers.py
--- a/utils/helpers.py
+++ b/utils/helpers.py
@@ -15,7 +15,7 @@ def process_items(items):
     if not items:
-        return None
+        return []
     return [transform(item) for item in items]

Example 2 - Adding a missing condition:
Problem: Division by zero error when count is 0.
Patch:
diff --git a/stats/calculator.py b/stats/calculator.py
--- a/stats/calculator.py
+++ b/stats/calculator.py
@@ -42,6 +42,8 @@ def calculate_average(self, values):
     def get_ratio(self, count):
+        if count == 0:
+            return 0.0
         return self.total / count

Example 3 - Fixing string handling:
Problem: Unicode characters cause encoding errors.
Patch:
diff --git a/io/parser.py b/io/parser.py
--- a/io/parser.py
+++ b/io/parser.py
@@ -78,7 +78,7 @@ class FileParser:
     def read_content(self, filepath):
-        with open(filepath, 'r') as f:
+        with open(filepath, 'r', encoding='utf-8') as f:
             return f.read()
'''


def build_prompt_v2(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V2 Prompt: Few-shot examples with format priming.
    
    KNOWN BUGS:
    - Ends with "diff --git" causing model to output "a/..." without space
    - Few-shot examples sometimes leak into output
    
    DEPRECATED: Use V3 instead.
    """
    problem = issue['problem_statement'].strip()
    
    prompt = f'''You are an expert software engineer. Your task is to fix a bug in a Python repository by generating a minimal git patch.

## Repository
{issue['repo']}

## Problem Description
{problem}

## Patch Format Requirements
Your patch MUST follow this exact format:
1. Start with: diff --git a/<filepath> b/<filepath>
2. Include --- a/<filepath> and +++ b/<filepath> lines
3. Include @@ line numbers @@ hunk headers
4. Use - for removed lines and + for added lines
5. Include 3 lines of context before and after changes

{FEW_SHOT_EXAMPLES_V2}

## Your Task
Generate a patch that fixes the issue described above. The patch should:
- Be minimal (change only what's necessary)
- Follow the exact unified diff format shown in examples
- Not include any explanation, just the patch

## Patch
diff --git'''
    
    return prompt


# =============================================================================
# V3 PROMPT - Zero-shot, information-dense
# =============================================================================

def build_prompt_v3(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V3 Prompt: Zero-shot, information-dense.
    
    Format spec is one dense line the model can pattern-match
    without example overhead.
    
    Rationale (based on ICL research):
    - Wei et al. 2023: Small models ignore few-shot examples, rely on priors
    - Shi et al. 2024: Small models emphasize important features, robust to noise
    - Few-shot at 1.5B scale consumes tokens without teaching format
    """
    problem = issue['problem_statement'].strip()
    
    return f'''Fix this bug with a unified diff patch. No markdown, no explanation.

Repo: {issue['repo']}

Bug:
{problem}

Output format: diff --git a/path b/path, then --- a/path, +++ b/path, @@ -N,M +N,M @@, then -/+ lines.

Patch:
'''


# =============================================================================
# V4 PROMPT - Zero-shot, slightly more explicit
# =============================================================================

def build_prompt_v4(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V4 Prompt: Zero-shot, slightly more explicit format instructions.
    
    If V3's single-line format spec is too abstract for the model,
    this provides step-by-step format guidance without examples.
    """
    problem = issue['problem_statement'].strip()
    
    return f'''Fix this bug. Output only a unified diff patch.

Repo: {issue['repo']}

Bug:
{problem}

Start with: diff --git a/filepath b/filepath
Then: --- a/filepath and +++ b/filepath  
Then: @@ -line,count +line,count @@ context
Then: lines starting with - (remove) or + (add)

Patch:
'''


# =============================================================================
# V5 PROMPT - Ultra-minimal for 1.5B
# =============================================================================

def build_prompt_v5(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V5 Prompt: Ultra-minimal for 1.5B models.
    
    Absolute minimum instructions. Based on empirical results showing
    1.5B + Temp 0.1 + 7 agents works best with direct, simple prompts.
    
    Target: 1.5B models at Temp 0.1
    """
    problem = issue['problem_statement'].strip()
    
    return f'''You are a code repair agent. Fix the failing test.

Repository: {issue['repo']}
Issue: {problem}

Steps:
1. Read the error
2. Find the buggy code
3. Write a patch
4. Return your patch in diff format

Be precise. One fix at a time.

Patch:
'''


# =============================================================================
# V6 PROMPT - Structured checklist for 1.5B
# =============================================================================

def build_prompt_v6(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V6 Prompt: Structured checklist for 1.5B models.
    
    Task breakdown without verbosity. Checkboxes provide structure
    without complex instructions.
    
    Target: 1.5B models at Temp 0.1
    """
    problem = issue['problem_statement'].strip()
    
    return f'''Fix this bug:

Bug: {problem}
Repo: {issue['repo']}

Your job:
□ Locate the broken file
□ Identify the exact problem
□ Generate minimal patch
□ Verify logic

Output format: git diff
Keep changes small.

Patch:
'''


# =============================================================================
# V7 PROMPT - Command-style for 1.5B
# =============================================================================

def build_prompt_v7(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V7 Prompt: Command-style for 1.5B models.
    
    Military-style directives. Maximum clarity with minimal words.
    All-caps sections provide strong structure for small models.
    
    Target: 1.5B models at Temp 0.1
    """
    problem = issue['problem_statement'].strip()
    
    return f'''TASK: Repair failing test
REPO: {issue['repo']}
ISSUE: {problem}

INSTRUCTIONS:
- Analyze error message
- Find relevant code
- Create patch
- Output diff format only

CONSTRAINTS:
- Minimal changes
- No explanations
- Focus on root cause

Patch:
'''


# =============================================================================
# V8 PROMPT - Classic few-shot for 7B
# =============================================================================

FEW_SHOT_EXAMPLES_V8 = '''Example 1:
Issue: Function returns None instead of empty list
Error: AssertionError: expected [], got None
Solution:
diff --git a/utils.py b/utils.py
--- a/utils.py
+++ b/utils.py
@@ -10,7 +10,7 @@ def process_data(items):
     if not items:
-        return None
+        return []
     return [x * 2 for x in items]

Example 2:
Issue: Off-by-one error in loop
Error: IndexError: list index out of range
Solution:
diff --git a/parser.py b/parser.py
--- a/parser.py
+++ b/parser.py
@@ -23,7 +23,7 @@ def extract_values(data):
     values = []
-    for i in range(len(data)):
+    for i in range(len(data) - 1):
         values.append(data[i])
     return values

Example 3:
Issue: Missing encoding parameter
Error: UnicodeDecodeError: 'charmap' codec can't decode
Solution:
diff --git a/reader.py b/reader.py
--- a/reader.py
+++ b/reader.py
@@ -15,7 +15,7 @@ def load_file(path):
-    with open(path, 'r') as f:
+    with open(path, 'r', encoding='utf-8') as f:
         return f.read()
'''


def build_prompt_v8(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V8 Prompt: Classic few-shot for 7B models.
    
    Three concrete examples with error types and solutions.
    7B models can actually learn from these patterns.
    
    Target: 7B models at Temp 0.1
    """
    problem = issue['problem_statement'].strip()
    
    return f'''You are an expert code repair agent. Below are examples of successful bug fixes:

{FEW_SHOT_EXAMPLES_V8}

Now fix this bug:
Repository: {issue['repo']}
Issue: {problem}

Think step-by-step:
1. What does the error tell us?
2. Where is the likely location?
3. What is the minimal fix?
4. Generate the patch.

Patch:
'''


# =============================================================================
# V9 PROMPT - Chain-of-thought for 7B
# =============================================================================

def build_prompt_v9(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V9 Prompt: Chain-of-thought for 7B models.
    
    Explicit reasoning steps before patch generation.
    Encourages 7B models to use their reasoning capacity.
    
    Target: 7B models at Temp 0.1
    """
    problem = issue['problem_statement'].strip()
    
    return f'''You are debugging a repository. Use systematic reasoning.

Repository: {issue['repo']}
Issue: {problem}

Reasoning Process:
1. **Error Analysis**: What type of error is this? (syntax, logic, type, etc.)
2. **Hypothesis**: What could cause this specific failure?
3. **Location**: Which file and function are most likely involved?
4. **Root Cause**: What is the exact line or logic error?
5. **Fix Design**: What minimal change resolves this?
6. **Patch Generation**: Create the diff.

Examples of good reasoning:
- "The KeyError suggests a missing dictionary key, likely in data parsing"
- "The None return indicates an unhandled edge case in the conditional"
- "The type mismatch means we need explicit conversion at line X"

Apply this reasoning to generate your patch.

Patch:
'''


# =============================================================================
# V10 PROMPT - Role-playing expert for 7B
# =============================================================================

def build_prompt_v10(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V10 Prompt: Role-playing expert for 7B models.
    
    Senior engineer persona with debugging heuristics.
    Leverages 7B models' ability to follow complex instructions.
    
    Target: 7B models at Temp 0.1
    """
    problem = issue['problem_statement'].strip()
    
    return f'''You are a senior software engineer reviewing a failing test. You have 15 years of debugging experience.

Context:
- Repository: {issue['repo']}
- Issue: {problem}

As an expert, you know:
- Most bugs are simple logic errors or edge cases
- Error messages point directly to the problem
- Minimal patches are better than rewrites
- Type errors often need explicit conversions
- Off-by-one errors are common in loops

Your approach:
1. Read the error like a detective
2. Form a hypothesis about the root cause
3. Verify by examining relevant code
4. Design a surgical fix
5. Generate a clean diff

Show your expert reasoning, then provide the patch.

Patch:
'''


# =============================================================================
# V11 PROMPT - Constraint-based for 7B
# =============================================================================

def build_prompt_v11(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V11 Prompt: Constraint-based for 7B models.
    
    Explicit rules and debugging framework.
    7B models can handle complex constraints effectively.
    
    Target: 7B models at Temp 0.1
    """
    problem = issue['problem_statement'].strip()
    
    return f'''Debug this repository with the following constraints:

Repository: {issue['repo']}
Issue: {problem}

Mandatory Rules:
1. Your patch must be <10 lines
2. Change only the minimum required
3. Preserve existing logic where possible
4. Match existing code style
5. Fix only the reported issue

Debugging Framework:
- Step 1: Parse error type (KeyError, TypeError, AssertionError, etc.)
- Step 2: Trace error to source file
- Step 3: Identify problematic lines
- Step 4: Propose fix that satisfies all constraints
- Step 5: Format as git diff

Example output format:
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -10,1 +10,1 @@
-    return None
+    return []

Now apply this framework.

Patch:
'''


# =============================================================================
# V12 PROMPT - V3 variant with structure
# =============================================================================

def build_prompt_v12(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V12 Prompt: V3 variant with added structure.
    
    Keeps V3's density but adds clear sections.
    Good middle ground between V3's brevity and V4's explicitness.
    
    Target: Any model
    """
    problem = issue['problem_statement'].strip()
    
    return f'''Fix this bug with a unified diff patch.

REPO: {issue['repo']}

BUG:
{problem}

FORMAT: diff --git a/path b/path, then --- a/path, +++ b/path, @@ -N,M +N,M @@, then -/+ lines.

RULES: No markdown. No explanation. Minimal changes only.

Patch:
'''


# =============================================================================
# V13 PROMPT - V3 variant with explicit steps
# =============================================================================

def build_prompt_v13(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V13 Prompt: V3 variant with explicit steps.
    
    Dense but procedural. Adds numbered steps to V3's format.
    
    Target: Any model
    """
    problem = issue['problem_statement'].strip()
    
    return f'''Fix this bug. Output unified diff only.

Repo: {issue['repo']}
Bug: {problem}

Steps:
1. Find the error source
2. Write minimal fix
3. Format as: diff --git a/path b/path; --- a/path; +++ b/path; @@ -N,M +N,M @@; -/+ lines

No markdown. No explanation.

Patch:
'''


# =============================================================================
# V14 PROMPT - V4 variant with inline examples
# =============================================================================

def build_prompt_v14(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V14 Prompt: V4 variant with inline examples.
    
    Step-by-step instructions with format examples inline.
    More concrete than V4 without full few-shot overhead.
    
    Target: Any model
    """
    problem = issue['problem_statement'].strip()
    
    return f'''Fix this bug. Output only a unified diff patch.

Repo: {issue['repo']}

Bug:
{problem}

Format (example):
diff --git a/utils.py b/utils.py
--- a/utils.py
+++ b/utils.py
@@ -10,1 +10,1 @@
-    return None
+    return []

Your patch must follow this exact format:
1. Start: diff --git a/filepath b/filepath
2. Then: --- a/filepath and +++ b/filepath
3. Then: @@ -line,count +line,count @@
4. Then: - for removed lines, + for added lines

Patch:
'''


# =============================================================================
# V15 PROMPT - V4 variant with constraints
# =============================================================================

def build_prompt_v15(issue: Dict, context_examples: List[Dict] = None) -> str:
    """
    V15 Prompt: V4 variant with constraints.
    
    Format instructions plus mandatory rules.
    Adds constraint thinking to V4's structure.
    
    Target: Any model
    """
    problem = issue['problem_statement'].strip()
    
    return f'''Fix this bug. Output only a unified diff patch.

Repo: {issue['repo']}

Bug:
{problem}

Constraints:
- Change <5 lines if possible
- Preserve existing logic
- Fix only the reported issue
- No explanations in output

Format:
Start with: diff --git a/filepath b/filepath
Then: --- a/filepath and +++ b/filepath
Then: @@ -line,count +line,count @@ context
Then: lines with - (remove) or + (add)

Patch:
'''


# =============================================================================
# Stop Sequences
# =============================================================================

STOP_SEQUENCES = {
    "v1": [
        "<|im_end|>",
        "<|endoftext|>",
    ],
    "v2": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\nExample",       # Catches few-shot regurgitation
    ],
    "v3": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",             # Catches markdown leakage
        "\n\n\n",          # Catches runaway generation
        "\nBug:",          # Catches prompt regurgitation
        "\nOutput format:",# Catches prompt regurgitation
        "\nRepo:",         # Catches prompt regurgitation
    ],
    "v4": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nBug:",
        "\nStart with:",
        "\nRepo:",
    ],
    "v5": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nSteps:",
        "\nRepository:",
        "\nIssue:",
    ],
    "v6": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nBug:",
        "\nRepo:",
        "\nYour job:",
    ],
    "v7": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nTASK:",
        "\nREPO:",
        "\nISSUE:",
        "\nINSTRUCTIONS:",
    ],
    "v8": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nExample",
        "\nNow fix",
        "\nRepository:",
    ],
    "v9": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nReasoning Process:",
        "\nRepository:",
        "\nExamples of",
    ],
    "v10": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nContext:",
        "\nAs an expert",
        "\nYour approach:",
    ],
    "v11": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nMandatory Rules:",
        "\nDebugging Framework:",
        "\nRepository:",
    ],
    "v12": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nREPO:",
        "\nBUG:",
        "\nFORMAT:",
        "\nRULES:",
    ],
    "v13": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nSteps:",
        "\nRepo:",
        "\nBug:",
    ],
    "v14": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nFormat (example):",
        "\nYour patch must",
        "\nRepo:",
    ],
    "v15": [
        "<|im_end|>",
        "<|endoftext|>",
        "```",
        "\n\n\n",
        "\nConstraints:",
        "\nFormat:",
        "\nRepo:",
        "\nBug:",
    ],
}


def get_stop_sequences(version: str) -> List[str]:
    """Get stop sequences for a prompt version."""
    if version not in STOP_SEQUENCES:
        raise ValueError(f"Unknown version: {version}. Available: {list(STOP_SEQUENCES.keys())}")
    return STOP_SEQUENCES[version]


# =============================================================================
# Prompt Selection
# =============================================================================

PROMPT_BUILDERS = {
    "v1": build_prompt_v1,
    "v2": build_prompt_v2,
    "v3": build_prompt_v3,
    "v4": build_prompt_v4,
    "v5": build_prompt_v5,
    "v6": build_prompt_v6,
    "v7": build_prompt_v7,
    "v8": build_prompt_v8,
    "v9": build_prompt_v9,
    "v10": build_prompt_v10,
    "v11": build_prompt_v11,
    "v12": build_prompt_v12,
    "v13": build_prompt_v13,
    "v14": build_prompt_v14,
    "v15": build_prompt_v15,
}


def build_swebench_prompt(
    issue: Dict, 
    version: str = "v3",
    context_examples: List[Dict] = None
) -> str:
    """
    Build SWE-bench prompt using specified version.
    
    Args:
        issue: SWE-bench issue dictionary with 'repo' and 'problem_statement'
        version: Prompt version to use ("v1" through "v15")
        context_examples: Optional agent context examples (for specialization)
    
    Returns:
        Formatted prompt string
    """
    if version not in PROMPT_BUILDERS:
        raise ValueError(f"Unknown prompt version: {version}. Available: {list(PROMPT_BUILDERS.keys())}")
    
    builder = PROMPT_BUILDERS[version]
    return builder(issue, context_examples)


def get_prompt_info(version: str) -> PromptVersion:
    """Get metadata about a prompt version."""
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown version: {version}")
    return PROMPT_VERSIONS[version]


def list_versions() -> List[str]:
    """List all available prompt versions."""
    return list(PROMPT_VERSIONS.keys())


def list_versions_by_model(target_model: str = None) -> List[str]:
    """
    List prompt versions filtered by target model.
    
    Args:
        target_model: "1.5B", "7B", or None for all
    
    Returns:
        List of version strings
    """
    if target_model is None:
        return list_versions()
    
    return [
        v for v, info in PROMPT_VERSIONS.items()
        if info.target_model == target_model or info.target_model == "any"
    ]


# =============================================================================
# Utility: Estimate Token Count
# =============================================================================

def estimate_prompt_tokens(prompt: str, chars_per_token: float = 4.0) -> int:
    """
    Rough estimate of token count.
    
    Args:
        prompt: The prompt string
        chars_per_token: Average characters per token (4.0 is reasonable for code)
    
    Returns:
        Estimated token count
    """
    return int(len(prompt) / chars_per_token)


def get_prompt_stats(issue: Dict, version: str) -> Dict:
    """
    Get statistics about a prompt for a given issue.
    
    Returns dict with:
        - prompt_tokens: Estimated prompt token count
        - available_tokens: Tokens left for generation (assuming 8192 context)
        - problem_truncated: Always False (no truncation)
    """
    prompt = build_swebench_prompt(issue, version)
    prompt_tokens = estimate_prompt_tokens(prompt)
    
    problem = issue['problem_statement'].strip()
    
    return {
        "version": version,
        "prompt_tokens": prompt_tokens,
        "available_tokens": 8192 - prompt_tokens,
        "problem_truncated": False,
        "problem_length": len(problem),
        "target_model": PROMPT_VERSIONS[version].target_model,
    }


# =============================================================================
# Configuration Helper
# =============================================================================

def get_recommended_config(version: str) -> Dict:
    """
    Get recommended configuration for a prompt version.
    
    Returns dict with:
        - temperature: Recommended temperature
        - num_agents: Recommended number of agents
        - passes_per_agent: Recommended passes per agent
        - target_model: Target model size
    """
    info = get_prompt_info(version)
    
    # Base recommendations from empirical results
    if info.target_model == "1.5B":
        return {
            "temperature": 0.1,
            "num_agents": 7,
            "passes_per_agent": 4,
            "target_model": "1.5B",
            "rationale": "1.5B models perform best with low temp and 7 agents"
        }
    elif info.target_model == "7B":
        return {
            "temperature": 0.1,
            "num_agents": 7,
            "passes_per_agent": 2,
            "target_model": "7B",
            "rationale": "7B models need few-shot/reasoning prompts at low temp"
        }
    else:
        return {
            "temperature": 0.1,
            "num_agents": 7,
            "passes_per_agent": 3,
            "target_model": "any",
            "rationale": "Default safe configuration"
        }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Test with a sample issue
    sample_issue = {
        "repo": "pytest-dev/pytest",
        "problem_statement": """pytest 6.0.0rc1: capfd.readouterr() converts \\r to \\n

When using pytest's capfd fixture, the readouterr() method is converting 
carriage return characters (\\r) to newlines (\\n). This breaks tests that 
specifically check for \\r characters in output.

Minimal reproducer:
```python
def test_capfd_cr(capfd):
    print("hello\\rworld", end="")
    out, err = capfd.readouterr()
    assert out == "hello\\rworld"  # FAILS: out is "hello\\nworld"
```

Expected: \\r characters should be preserved
Actual: \\r is converted to \\n"""
    }
    
    print("=" * 80)
    print("PROMPT TEMPLATE TEST")
    print("=" * 80)
    
    # Test by model type
    for model_type in ["1.5B", "7B", "any"]:
        print(f"\n{'='*80}")
        print(f"PROMPTS FOR {model_type} MODELS")
        print(f"{'='*80}")
        
        versions = list_versions_by_model(model_type)
        for version in versions:
            info = get_prompt_info(version)
            if info.target_model != model_type and model_type != "any":
                continue
                
            stats = get_prompt_stats(sample_issue, version)
            config = get_recommended_config(version)
            prompt = build_swebench_prompt(sample_issue, version)
            stops = get_stop_sequences(version)
            
            print(f"\n{'-'*80}")
            print(f"VERSION: {version}")
            print(f"{'-'*80}")
            print(f"Description: {info.description}")
            print(f"Target Model: {info.target_model}")
            print(f"Estimated tokens: {stats['prompt_tokens']}")
            print(f"Available for generation: {stats['available_tokens']}")
            print(f"\nRecommended Config:")
            print(f"  Temperature: {config['temperature']}")
            print(f"  Agents: {config['num_agents']}")
            print(f"  Passes/Agent: {config['passes_per_agent']}")
            print(f"  Rationale: {config['rationale']}")
            print(f"\nStop sequences: {stops[:3]}... ({len(stops)} total)")
            print(f"\n--- FULL PROMPT ---")
            print(prompt)
            print(f"--- END PROMPT ---")
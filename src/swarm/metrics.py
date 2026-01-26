"""Emergence metrics for measuring specialization in swarm agents."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from collections import Counter
from itertools import combinations
from scipy.stats import chi2_contingency
from sentence_transformers import SentenceTransformer

from .agent import SwarmAgent
from .types import TaskType


class MetricsEngine:
    """
    Compute all emergence metrics for swarm system.

    Implements metrics from the pivot/metrics-document.md:
    - Specialization Index (S): Entropy-based measure of task-agent association
    - Context Divergence (D): Embedding-based context similarity
    - Functional Differentiation (F): Chi-square test of performance differences
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize metrics engine.

        Args:
            embedding_model: SentenceTransformer model for embeddings
        """
        self.embedder = SentenceTransformer(embedding_model)

    def specialization_index(self, task_log: List[Dict]) -> float:
        """
        Compute Specialization Index (S) using information theory.

        S = 1 - H(task_type | agent) / H(task_type)

        Where:
        - H(task_type) = entropy of task type distribution
        - H(task_type | agent) = conditional entropy of task type given agent

        Args:
            task_log: List of {"agent_id": int, "task_type": str, ...}

        Returns:
            S ∈ [0, 1], where 0 = random, 1 = perfect specialization
        """
        if not task_log:
            return 0.0

        # Count task types
        task_types = [t["task_type"] for t in task_log]
        type_counts = Counter(task_types)
        total = len(task_log)

        # H(task_type) - baseline entropy
        p_type = np.array(list(type_counts.values())) / total
        H_task = -np.sum(p_type * np.log2(p_type + 1e-10))

        if H_task == 0:
            return 0.0

        # H(task_type | agent) - conditional entropy
        agents = set(t["agent_id"] for t in task_log)
        H_conditional = 0.0

        for agent in agents:
            agent_tasks = [t for t in task_log if t["agent_id"] == agent]
            p_agent = len(agent_tasks) / total

            agent_type_counts = Counter(t["task_type"] for t in agent_tasks)
            p_type_given_agent = np.array(list(agent_type_counts.values())) / len(agent_tasks)
            H_type_given_agent = -np.sum(
                p_type_given_agent * np.log2(p_type_given_agent + 1e-10)
            )

            H_conditional += p_agent * H_type_given_agent

        # Specialization index
        S = 1 - (H_conditional / (H_task + 1e-10))
        return max(0.0, min(1.0, S))  # Clamp to [0, 1]

    def specialization_significance(
        self,
        task_log: List[Dict],
        n_permutations: int = 1000
    ) -> Dict:
        """
        Test if specialization is statistically significant vs null (random).

        Args:
            task_log: List of task entries
            n_permutations: Number of random permutations for null distribution

        Returns:
            Dict with observed S, null distribution, and p-value
        """
        observed_S = self.specialization_index(task_log)

        # Generate null distribution by shuffling agent assignments
        null_distribution = []
        agent_ids = [t["agent_id"] for t in task_log]

        for _ in range(n_permutations):
            shuffled_ids = np.random.permutation(agent_ids)
            shuffled_log = [
                {**t, "agent_id": shuffled_ids[i]}
                for i, t in enumerate(task_log)
            ]
            null_S = self.specialization_index(shuffled_log)
            null_distribution.append(null_S)

        # Compute p-value
        p_value = np.mean(np.array(null_distribution) >= observed_S)

        return {
            "observed_S": observed_S,
            "null_mean": np.mean(null_distribution),
            "null_std": np.std(null_distribution),
            "p_value": p_value,
            "significant": p_value < 0.05
        }

    def context_divergence(self, agents: List[SwarmAgent]) -> float:
        """
        Compute Context Divergence Score (D).

        D = 1 - mean(cosine_sim(context_i, context_j)) for all agent pairs

        Args:
            agents: List of swarm agents

        Returns:
            D ∈ [0, 1], where 0 = identical contexts, 1 = orthogonal contexts
        """
        if len(agents) < 2:
            return 0.0

        # Build context texts for each agent
        agent_texts = {}
        for agent in agents:
            if not agent.context_buffer:
                agent_texts[agent.agent_id] = ""
            else:
                # Concatenate all examples
                examples = [
                    f"{ex.problem}\n{ex.solution}"
                    for ex in agent.context_buffer
                ]
                agent_texts[agent.agent_id] = "\n\n".join(examples)

        # Filter out empty contexts
        non_empty = {aid: text for aid, text in agent_texts.items() if text}

        if len(non_empty) < 2:
            return 0.0

        # Embed contexts
        agent_ids = list(non_empty.keys())
        embeddings = self.embedder.encode([non_empty[aid] for aid in agent_ids])

        # Compute pairwise cosine similarities
        similarities = []
        for (i, j) in combinations(range(len(agent_ids)), 2):
            cos_sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
            )
            similarities.append(cos_sim)

        if not similarities:
            return 0.0

        # Divergence = 1 - mean similarity
        D = 1.0 - np.mean(similarities)
        return max(0.0, min(1.0, D))

    def functional_differentiation(self, task_log: List[Dict]) -> Dict:
        """
        Compute Functional Differentiation Score (F) using chi-square test.

        Tests if agent performance is independent of task type.

        Args:
            task_log: List of {"agent_id": int, "task_type": str, "success": bool}

        Returns:
            Dict with chi2, p_value, significant, effect_size, contingency_table
        """
        if not task_log:
            return {
                "chi2": 0.0,
                "p_value": 1.0,
                "dof": 0,
                "significant": False,
                "effect_size": 0.0,
                "contingency_table": None
            }

        # Build contingency table: rows = agents, cols = task_types
        df = pd.DataFrame(task_log)

        # Count successes by agent and task type
        contingency = pd.crosstab(
            df["agent_id"],
            df["task_type"],
            values=df["success"],
            aggfunc="sum"
        ).fillna(0)

        # Ensure we have at least 2x2 table
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return {
                "chi2": 0.0,
                "p_value": 1.0,
                "dof": 0,
                "significant": False,
                "effect_size": 0.0,
                "contingency_table": contingency.to_dict()
            }

        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        # Cramér's V effect size
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim + 1e-10))

        return {
            "chi2": float(chi2),
            "p_value": float(p_value),
            "dof": int(dof),
            "significant": p_value < 0.05,
            "effect_size": float(cramers_v),
            "contingency_table": contingency.to_dict()
        }

    def detect_phase_transition(
        self,
        S_timeline: List[float],
        threshold: float = 0.15
    ) -> Dict:
        """
        Detect sudden jumps in specialization (phase transitions).

        Args:
            S_timeline: List of S values over time
            threshold: Minimum S increase to count as transition

        Returns:
            Dict with transitions detected, pattern classification
        """
        if len(S_timeline) < 2:
            return {
                "transitions_detected": False,
                "transitions": [],
                "final_S": S_timeline[-1] if S_timeline else 0.0,
                "pattern": "insufficient_data"
            }

        transitions = []
        for i in range(1, len(S_timeline)):
            delta = S_timeline[i] - S_timeline[i - 1]
            if delta > threshold:
                transitions.append({
                    "time_step": i,
                    "delta_S": float(delta),
                    "S_before": float(S_timeline[i - 1]),
                    "S_after": float(S_timeline[i])
                })

        # Classify pattern
        pattern = self._classify_pattern(S_timeline)

        return {
            "transitions_detected": len(transitions) > 0,
            "transitions": transitions,
            "final_S": float(S_timeline[-1]),
            "pattern": pattern
        }

    def _classify_pattern(self, S_timeline: List[float]) -> str:
        """
        Classify emergence pattern.

        Returns:
            "phase_transition", "gradual_drift", "oscillation", or "no_emergence"
        """
        if len(S_timeline) < 3:
            return "insufficient_data"

        diffs = [S_timeline[i] - S_timeline[i - 1] for i in range(1, len(S_timeline))]
        max_jump = max(diffs) if diffs else 0.0
        variance = np.var(diffs)

        if S_timeline[-1] < 0.1:
            return "no_emergence"
        elif max_jump > 0.15:
            return "phase_transition"
        elif variance > 0.01:
            return "oscillation"
        else:
            return "gradual_drift"

    def snapshot(
        self,
        agents: List[SwarmAgent],
        task_log: List[Dict],
        task_count: int
    ) -> Dict:
        """
        Capture complete system state for logging.

        Args:
            agents: List of agents
            task_log: Complete task log
            task_count: Number of tasks completed

        Returns:
            Snapshot dictionary
        """
        return {
            "task_count": task_count,
            "specialization_index": self.specialization_index(task_log),
            "context_divergence": self.context_divergence(agents),
            "agent_states": {
                agent.agent_id: agent.get_state_summary()
                for agent in agents
            },
            "contexts": {
                agent.agent_id: [
                    {
                        "problem": ex.problem[:100],  # Truncate for storage
                        "task_type": ex.task_type.value
                    }
                    for ex in agent.context_buffer
                ]
                for agent in agents
            }
        }


class RobustnessMetrics:
    """Additional metrics for testing robustness of specialization."""

    @staticmethod
    def context_shuffle_sensitivity(
        original_S: float,
        shuffled_S: float,
        original_perf: float,
        shuffled_perf: float
    ) -> Dict:
        """
        Measure impact of shuffling agent contexts.

        Args:
            original_S: S with original contexts
            shuffled_S: S after shuffling contexts
            original_perf: Performance with original contexts
            shuffled_perf: Performance after shuffling

        Returns:
            Analysis of whether specialization is causal
        """
        S_drop = original_S - shuffled_S
        perf_drop = original_perf - shuffled_perf

        return {
            "S_original": original_S,
            "S_shuffled": shuffled_S,
            "S_drop": S_drop,
            "performance_drop": perf_drop,
            "specialization_is_causal": S_drop > 0.1,
            "performance_linked": perf_drop > 0.05
        }

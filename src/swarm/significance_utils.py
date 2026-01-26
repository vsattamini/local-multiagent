"""
Statistical significance utilities for emergence metrics.

Provides functions for:
- Confidence interval computation
- T-tests for comparing 2 conditions
- ANOVA for comparing 3+ population sizes
- Permutation tests for specialization significance
"""

from typing import Optional
import numpy as np


def compute_confidence_interval(
    values: list[float], 
    confidence: float = 0.95
) -> dict:
    """
    Compute mean and confidence interval for a set of values.
    
    Uses t-distribution for small samples (n < 30).
    
    Args:
        values: List of numeric values
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Dictionary with mean, std, ci_lower, ci_upper, and n
    """
    from scipy import stats
    
    n = len(values)
    if n == 0:
        return {"mean": 0, "std": 0, "ci_lower": 0, "ci_upper": 0, "n": 0}
    
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    
    # Use t-distribution for small samples
    df = max(n - 1, 1)
    t_val = stats.t.ppf((1 + confidence) / 2, df=df)
    
    return {
        "mean": mean,
        "std": std,
        "se": float(se),
        "ci_lower": float(mean - t_val * se),
        "ci_upper": float(mean + t_val * se),
        "n": n,
        "confidence": confidence
    }


def ttest_independent(
    group1: list[float], 
    group2: list[float],
    alternative: str = "two-sided"
) -> dict:
    """
    Perform independent samples t-test.
    
    Args:
        group1: Values for group 1
        group2: Values for group 2
        alternative: "two-sided", "less", or "greater"
        
    Returns:
        Dictionary with t_statistic, p_value, significant, effect_size (Cohen's d)
    """
    from scipy import stats
    
    if len(group1) < 2 or len(group2) < 2:
        return {
            "t_statistic": 0,
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0,
            "error": "Insufficient sample size"
        }
    
    t_stat, p_value = stats.ttest_ind(group1, group2, alternative=alternative)
    
    # Cohen's d effect size
    pooled_std = np.sqrt(
        ((len(group1) - 1) * np.var(group1, ddof=1) + 
         (len(group2) - 1) * np.var(group2, ddof=1)) / 
        (len(group1) + len(group2) - 2)
    )
    
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "effect_size": float(cohens_d),
        "effect_interpretation": interpret_cohens_d(abs(cohens_d)),
        "n1": len(group1),
        "n2": len(group2)
    }


def anova_one_way(groups: list[list[float]]) -> dict:
    """
    Perform one-way ANOVA comparing multiple groups.
    
    Args:
        groups: List of groups, each group is a list of values
        
    Returns:
        Dictionary with f_statistic, p_value, significant, effect_size (eta-squared)
    """
    from scipy import stats
    
    # Filter out empty groups
    valid_groups = [g for g in groups if len(g) > 0]
    
    if len(valid_groups) < 2:
        return {
            "f_statistic": 0,
            "p_value": 1.0,
            "significant": False,
            "effect_size": 0,
            "error": "Need at least 2 non-empty groups"
        }
    
    f_stat, p_value = stats.f_oneway(*valid_groups)
    
    # Eta-squared effect size
    all_values = [v for g in valid_groups for v in g]
    grand_mean = np.mean(all_values)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in valid_groups)
    ss_total = sum((v - grand_mean)**2 for v in all_values)
    
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "effect_size": float(eta_squared),
        "effect_interpretation": interpret_eta_squared(eta_squared),
        "n_groups": len(valid_groups),
        "group_sizes": [len(g) for g in valid_groups]
    }


def permutation_test(
    observed: float, 
    null_distribution: list[float],
    alternative: str = "greater"
) -> dict:
    """
    Compute p-value using permutation test.
    
    Args:
        observed: Observed statistic value
        null_distribution: Distribution of statistic under null hypothesis
        alternative: "greater", "less", or "two-sided"
        
    Returns:
        Dictionary with p_value, significant, percentile
    """
    null = np.array(null_distribution)
    
    if len(null) == 0:
        return {
            "p_value": 1.0,
            "significant": False,
            "percentile": 0,
            "error": "Empty null distribution"
        }
    
    if alternative == "greater":
        p_value = np.mean(null >= observed)
    elif alternative == "less":
        p_value = np.mean(null <= observed)
    else:  # two-sided
        p_value = 2 * min(np.mean(null >= observed), np.mean(null <= observed))
    
    percentile = np.mean(null < observed) * 100
    
    return {
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "percentile": float(percentile),
        "observed": float(observed),
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null))
    }


def tukey_hsd(groups: list[list[float]], group_names: Optional[list[str]] = None) -> list[dict]:
    """
    Perform Tukey's HSD post-hoc test for pairwise comparisons.
    
    Args:
        groups: List of groups, each group is a list of values
        group_names: Optional names for groups
        
    Returns:
        List of pairwise comparison results
    """
    from scipy import stats
    
    if group_names is None:
        group_names = [f"Group_{i}" for i in range(len(groups))]
    
    results = []
    n_groups = len(groups)
    
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            # Pairwise t-test with Bonferroni correction
            if len(groups[i]) < 2 or len(groups[j]) < 2:
                continue
                
            t_stat, raw_p = stats.ttest_ind(groups[i], groups[j])
            
            # Bonferroni correction for multiple comparisons
            n_comparisons = n_groups * (n_groups - 1) / 2
            adj_p = min(raw_p * n_comparisons, 1.0)
            
            results.append({
                "group1": group_names[i],
                "group2": group_names[j],
                "mean_diff": float(np.mean(groups[i]) - np.mean(groups[j])),
                "t_statistic": float(t_stat),
                "p_value_raw": float(raw_p),
                "p_value_adjusted": float(adj_p),
                "significant": adj_p < 0.05
            })
    
    return results


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_eta_squared(eta2: float) -> str:
    """Interpret eta-squared effect size."""
    if eta2 < 0.01:
        return "negligible"
    elif eta2 < 0.06:
        return "small"
    elif eta2 < 0.14:
        return "medium"
    else:
        return "large"


def compare_experiments(
    baseline: list[float],
    experimental: list[float],
    metric_name: str = "metric"
) -> dict:
    """
    Comprehensive comparison of baseline vs experimental conditions.
    
    Args:
        baseline: Values from baseline condition
        experimental: Values from experimental condition
        metric_name: Name of the metric being compared
        
    Returns:
        Comprehensive comparison report
    """
    baseline_ci = compute_confidence_interval(baseline)
    experimental_ci = compute_confidence_interval(experimental)
    ttest = ttest_independent(baseline, experimental)
    
    improvement = (experimental_ci["mean"] - baseline_ci["mean"]) / baseline_ci["mean"] * 100 \
        if baseline_ci["mean"] != 0 else 0
    
    return {
        "metric": metric_name,
        "baseline": baseline_ci,
        "experimental": experimental_ci,
        "comparison": ttest,
        "improvement_percent": float(improvement),
        "conclusion": _generate_conclusion(ttest, improvement)
    }


def _generate_conclusion(ttest: dict, improvement: float) -> str:
    """Generate human-readable conclusion from comparison results."""
    if ttest.get("significant"):
        direction = "higher" if improvement > 0 else "lower"
        return (
            f"Significant difference detected (p={ttest['p_value']:.4f}). "
            f"Experimental condition is {abs(improvement):.1f}% {direction} "
            f"with {ttest['effect_interpretation']} effect size."
        )
    else:
        return (
            f"No significant difference detected (p={ttest['p_value']:.4f}). "
            f"Observed difference of {abs(improvement):.1f}% may be due to chance."
        )

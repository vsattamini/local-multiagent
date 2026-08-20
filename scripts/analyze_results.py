#!/usr/bin/env python3
"""
Analyze and visualize swarm experiment results.

Usage:
    python scripts/analyze_results.py results/pilot_001
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(results_dir: Path):
    """Load experiment results."""
    task_log = []
    with open(results_dir / "task_log.jsonl", "r") as f:
        for line in f:
            if line.strip():
                task_log.append(json.loads(line))

    snapshots = []
    snapshot_file = results_dir / "snapshots.jsonl"
    if snapshot_file.exists():
        with open(snapshot_file, "r") as f:
            for line in f:
                if line.strip():
                    snapshots.append(json.loads(line))

    final_metrics = {}
    metrics_file = results_dir / "final_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            data = json.load(f)
            final_metrics = data.get('metrics', data)

    return task_log, snapshots, final_metrics


def plot_specialization_timeline(snapshots, output_dir):
    """Plot S(t) and D(t) over time."""
    if not snapshots:
        print("No snapshots available for timeline plot")
        return

    task_counts = [s['task_count'] for s in snapshots]
    S_values = [s['specialization_index'] for s in snapshots]
    D_values = [s['context_divergence'] for s in snapshots]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot S(t)
    ax1.plot(task_counts, S_values, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='S=0.1 threshold')
    ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='S=0.3 (moderate)')
    ax1.set_xlabel('Tasks Completed', fontsize=12)
    ax1.set_ylabel('Specialization Index (S)', fontsize=12)
    ax1.set_title('Emergence Timeline: Specialization Index', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot D(t)
    ax2.plot(task_counts, D_values, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='D=0.2 threshold')
    ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='D=0.4 (substantial)')
    ax2.set_xlabel('Tasks Completed', fontsize=12)
    ax2.set_ylabel('Context Divergence (D)', fontsize=12)
    ax2.set_title('Emergence Timeline: Context Divergence', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'timeline.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved timeline plot: {output_dir / 'timeline.png'}")
    plt.close()


def plot_agent_specialization(final_metrics, output_dir):
    """Plot agent specialization heatmap."""
    if 'affinity_matrix' not in final_metrics or not final_metrics['affinity_matrix']:
        print("No affinity matrix available")
        return

    affinity = final_metrics['affinity_matrix']

    # Convert to DataFrame
    df = pd.DataFrame(affinity).T
    df.index = [f"Agent {i}" for i in df.index]

    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Success Rate'})
    plt.title('Agent Specialization by Task Type', fontsize=14, fontweight='bold')
    plt.xlabel('Task Type', fontsize=12)
    plt.ylabel('Agent', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'agent_specialization.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved specialization heatmap: {output_dir / 'agent_specialization.png'}")
    plt.close()


def plot_performance_by_type(task_log, output_dir):
    """Plot performance breakdown by task type."""
    df = pd.DataFrame(task_log)

    # Group by task type
    type_stats = df.groupby('task_type').agg({
        'success': ['count', 'sum', 'mean']
    }).round(3)

    type_stats.columns = ['Total', 'Successful', 'Success Rate']

    # Bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Success rate by type
    type_stats['Success Rate'].plot(kind='bar', ax=ax1, color='#06A77D')
    ax1.set_title('Success Rate by Task Type', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Task Type', fontsize=12)
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    # Count by type
    type_stats['Total'].plot(kind='bar', ax=ax2, color='#F18F01')
    ax2.set_title('Task Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Task Type', fontsize=12)
    ax2.set_ylabel('Number of Tasks', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_type.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance breakdown: {output_dir / 'performance_by_type.png'}")
    plt.close()


def plot_agent_performance(task_log, output_dir):
    """Plot performance by agent."""
    df = pd.DataFrame(task_log)

    # Group by agent
    agent_stats = df.groupby('agent_id').agg({
        'success': ['count', 'sum', 'mean']
    }).round(3)

    agent_stats.columns = ['Total', 'Successful', 'Success Rate']

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    agent_stats['Success Rate'].plot(kind='bar', ax=ax, color='#C73E1D')
    ax.set_title('Success Rate by Agent', fontsize=14, fontweight='bold')
    ax.set_xlabel('Agent ID', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'agent_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved agent performance: {output_dir / 'agent_performance.png'}")
    plt.close()


def plot_contingency_table(final_metrics, output_dir):
    """Plot contingency table heatmap."""
    if 'functional_differentiation' not in final_metrics:
        return

    F = final_metrics['functional_differentiation']
    if not F.get('contingency_table'):
        return

    contingency = pd.DataFrame(F['contingency_table'])

    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency, annot=True, fmt='.0f', cmap='Blues', cbar_kws={'label': 'Success Count'})
    plt.title(f'Contingency Table: Agent × Task Type\n(χ² = {F["chi2"]:.2f}, p = {F["p_value"]:.4f})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Task Type', fontsize=12)
    plt.ylabel('Agent ID', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'contingency_table.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved contingency table: {output_dir / 'contingency_table.png'}")
    plt.close()


def generate_report(task_log, snapshots, final_metrics, output_dir):
    """Generate text report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SWARM EXPERIMENT ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary stats
    if 'summary_stats' in final_metrics:
        stats = final_metrics['summary_stats']
        report_lines.append("PERFORMANCE SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Tasks: {stats['total_tasks']}")
        report_lines.append(f"Successful: {stats['successful_tasks']}")
        report_lines.append(f"Pass@1: {stats['pass_at_1']:.2%}")
        report_lines.append(f"Avg Execution Time: {stats['avg_execution_time']:.2f}s")
        report_lines.append("")

    # Emergence metrics
    report_lines.append("EMERGENCE METRICS")
    report_lines.append("-" * 80)
    S = final_metrics.get('specialization_index', 0.0)
    D = final_metrics.get('context_divergence', 0.0)
    report_lines.append(f"Specialization Index (S): {S:.4f}")
    report_lines.append(f"Context Divergence (D): {D:.4f}")

    if 'specialization_significance' in final_metrics:
        sig = final_metrics['specialization_significance']
        report_lines.append(f"S p-value: {sig['p_value']:.4f} ({'significant' if sig['significant'] else 'not significant'})")
        report_lines.append(f"Null mean: {sig['null_mean']:.4f} ± {sig['null_std']:.4f}")

    report_lines.append("")

    # Functional differentiation
    if 'functional_differentiation' in final_metrics:
        F = final_metrics['functional_differentiation']
        report_lines.append("FUNCTIONAL DIFFERENTIATION")
        report_lines.append("-" * 80)
        report_lines.append(f"Chi-square: {F['chi2']:.2f}")
        report_lines.append(f"p-value: {F['p_value']:.4f} ({'significant' if F['significant'] else 'not significant'})")
        report_lines.append(f"Effect size (Cramér's V): {F['effect_size']:.4f}")
        report_lines.append("")

    # Phase transition
    if 'phase_transition' in final_metrics:
        pt = final_metrics['phase_transition']
        report_lines.append("PATTERN ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append(f"Pattern: {pt['pattern']}")
        report_lines.append(f"Final S: {pt['final_S']:.4f}")
        if pt['transitions_detected']:
            report_lines.append(f"Transitions detected: {len(pt['transitions'])}")
            for trans in pt['transitions']:
                report_lines.append(f"  - Step {trans['time_step']}: ΔS = {trans['delta_S']:.4f}")
        report_lines.append("")

    # Agent states
    if 'agent_states' in final_metrics:
        report_lines.append("AGENT STATES")
        report_lines.append("-" * 80)
        for agent_id, state in final_metrics['agent_states'].items():
            report_lines.append(f"Agent {agent_id}:")
            report_lines.append(f"  Total tasks: {state['total_tasks']}")
            report_lines.append(f"  Success rate: {state['success_rate']:.2%}")
            report_lines.append(f"  Context size: {state['context_size']}")
            report_lines.append("")

    # Verdict
    report_lines.append("=" * 80)
    report_lines.append("VERDICT")
    report_lines.append("=" * 80)

    if S > 0.1:
        report_lines.append("✓ GO: Specialization emerged (S > 0.1)")
        report_lines.append("  Recommendation: Proceed to full experiments")
    elif S < 0.1 and D > 0.2:
        report_lines.append("⚠️  GO WITH MODIFICATIONS: Contexts diverged but S low")
        report_lines.append("  Recommendation: Consider modified routing mechanism")
    else:
        report_lines.append("✗ PIVOT NEEDED: No emergence detected")
        report_lines.append("  Recommendation: Try more tasks, larger model, or different mechanism")

    report_lines.append("=" * 80)

    # Write report
    report_file = output_dir / "analysis_report.txt"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"✓ Saved analysis report: {report_file}")

    # Also print to console
    print("\n" + "\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser(description='Analyze swarm experiment results')
    parser.add_argument('results_dir', type=str, help='Path to results directory')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Loading results from: {results_dir}")

    # Load results
    try:
        task_log, snapshots, final_metrics = load_results(results_dir)
        print(f"✓ Loaded {len(task_log)} task entries")
        print(f"✓ Loaded {len(snapshots)} snapshots")
    except Exception as e:
        print(f"Error loading results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating visualizations...")
        plot_specialization_timeline(snapshots, results_dir)
        plot_agent_specialization(final_metrics, results_dir)
        plot_performance_by_type(task_log, results_dir)
        plot_agent_performance(task_log, results_dir)
        plot_contingency_table(final_metrics, results_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(task_log, snapshots, final_metrics, results_dir)

    print(f"\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

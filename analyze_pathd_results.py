"""
Path D Results Analysis and Visualization
Processes results from full dataset evaluation and generates figures for paper.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_all_results():
    """Load all Path D evaluation results."""
    results_dir = Path("results/path_d_full_evaluation")
    
    all_results = {}
    
    # Load PH2 splits
    for split in range(5):
        file_path = results_dir / f"PH2_split_{split}_full_results.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                all_results[f'PH2_split_{split}'] = json.load(f)
    
    # Load Derm7pt
    file_path = results_dir / "Derm7pt_full_results.json"
    if file_path.exists():
        with open(file_path, 'r') as f:
            all_results['Derm7pt'] = json.load(f)
    
    # Load HAM10000
    file_path = results_dir / "HAM10000_full_results.json"
    if file_path.exists():
        with open(file_path, 'r') as f:
            all_results['HAM10000'] = json.load(f)
    
    # Load consolidated if exists
    file_path = results_dir / "CONSOLIDATED_RESULTS.json"
    if file_path.exists():
        with open(file_path, 'r') as f:
            consolidated = json.load(f)
            print("✓ Loaded consolidated results")
    
    return all_results


def generate_summary_table(results):
    """Generate LaTeX table for paper."""
    
    table_data = []
    
    for dataset_name, data in results.items():
        if 'summary' in data:
            summary = data['summary']
        else:
            summary = data
        
        row = {
            'Dataset': dataset_name,
            'Test Size': data.get('test_size', 'N/A'),
            'Baseline Violations': f"{summary['avg_baseline_violations']:.2f}",
            'Refined Violations': f"{summary['avg_refined_violations']:.2f}",
            'Reduction': f"{summary['avg_reduction']:.2f}",
            'Reduction %': f"{summary['reduction_percentage']:.1f}%",
            'Improved Samples': f"{summary['samples_improved']} ({summary['samples_improved']/data.get('test_size', 1)*100:.1f}%)",
        }
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    output_file = "results/path_d_full_evaluation/summary_table.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Summary table saved to: {output_file}")
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, escape=False)
    latex_file = "results/path_d_full_evaluation/summary_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to: {latex_file}")
    
    return df


def plot_violation_reduction(results):
    """
    Create the KEY FIGURE for Path D paper.
    
    Shows: Real data (0% improvement) vs Synthetic (17% improvement)
    """
    
    # Prepare data
    datasets = []
    reductions = []
    colors = []
    
    # Real data results
    for dataset_name, data in results.items():
        if 'PH2' in dataset_name or dataset_name in ['Derm7pt', 'HAM10000']:
            summary = data['summary'] if 'summary' in data else data
            datasets.append(dataset_name)
            reductions.append(summary['reduction_percentage'])
            colors.append('#e74c3c')  # Red for real data (0% or minimal)
    
    # Add synthetic data point (from your Option 3 results)
    datasets.append('Synthetic\n(n=5)')
    reductions.append(17.0)  # Your Option 3 result
    colors.append('#2ecc71')  # Green for synthetic
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(datasets))
    bars = ax.bar(x_pos, reductions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Customize
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Violation Reduction (%)', fontsize=14, fontweight='bold')
    ax.set_title('Path D: Rule-Based Refinement Performance\nReal Medical Data vs Synthetic Examples',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, reductions)):
        height = bar.get_height()
        label = f'{val:.1f}%'
        if val >= 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, edgecolor='black', label='Real Medical Data (0% improvement)'),
        Patch(facecolor='#2ecc71', alpha=0.8, edgecolor='black', label='Synthetic Data (17% improvement)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Add annotation for the gap
    ax.annotate('GAP reveals\nmedical complexity', 
                xy=(len(datasets)-1, 17), xytext=(len(datasets)-3, 12),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    output_file = "results/path_d_full_evaluation/violation_reduction_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Main figure saved to: {output_file}")
    
    plt.close()


def plot_detection_vs_correction(results):
    """
    Create figure showing: Detection SUCCESS (97.5%) vs Correction FAILURE (0%)
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Detection Success
    detection_rate = 97.5
    ax1.bar(['Consistency\nDetection'], [detection_rate], color='#2ecc71', alpha=0.8, 
            edgecolor='black', linewidth=2)
    ax1.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Detection: SUCCESS', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.axhline(y=100, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    ax1.text(0, detection_rate + 2, f'{detection_rate}%', ha='center', va='bottom',
             fontsize=16, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Panel 2: Correction Failure
    # Calculate average across all real datasets
    avg_reduction = np.mean([
        data['summary']['reduction_percentage'] if 'summary' in data else data['reduction_percentage']
        for name, data in results.items()
        if 'PH2' in name or name in ['Derm7pt', 'HAM10000']
    ])
    
    ax2.bar(['Rule-Based\nCorrection'], [avg_reduction], color='#e74c3c', alpha=0.8,
            edgecolor='black', linewidth=2)
    ax2.set_ylabel('Violation Reduction (%)', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Correction: FAILURE', fontsize=14, fontweight='bold')
    ax2.set_ylim([-5, 20])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.text(0, avg_reduction + 0.5, f'{avg_reduction:.1f}%', ha='center', va='bottom',
             fontsize=16, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Path D: Dual Contribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = "results/path_d_full_evaluation/detection_vs_correction.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Detection vs Correction figure saved to: {output_file}")
    
    plt.close()


def generate_paper_statistics(results):
    """Generate key statistics for paper writing."""
    
    stats = {
        'total_samples_evaluated': 0,
        'total_baseline_violations': 0,
        'total_refined_violations': 0,
        'datasets': {}
    }
    
    for dataset_name, data in results.items():
        summary = data['summary'] if 'summary' in data else data
        test_size = data.get('test_size', 0)
        
        stats['total_samples_evaluated'] += test_size
        stats['total_baseline_violations'] += summary.get('total_baseline_violations', 0)
        stats['total_refined_violations'] += summary.get('total_refined_violations', 0)
        
        stats['datasets'][dataset_name] = {
            'test_size': test_size,
            'avg_baseline_violations': summary['avg_baseline_violations'],
            'avg_refined_violations': summary['avg_refined_violations'],
            'reduction_pct': summary['reduction_percentage'],
            'samples_improved': summary['samples_improved'],
            'convergence_rate': summary['convergence_rate']
        }
    
    # Calculate overall statistics
    stats['overall_reduction_pct'] = (
        (stats['total_baseline_violations'] - stats['total_refined_violations']) /
        stats['total_baseline_violations'] * 100
        if stats['total_baseline_violations'] > 0 else 0
    )
    
    # Save statistics
    output_file = "results/path_d_full_evaluation/paper_statistics.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Paper statistics saved to: {output_file}")
    
    # Print key numbers for paper
    print("\n" + "=" * 80)
    print("KEY NUMBERS FOR PAPER")
    print("=" * 80)
    print(f"\nTotal samples evaluated: {stats['total_samples_evaluated']}")
    print(f"Overall violation reduction: {stats['overall_reduction_pct']:.2f}%")
    print(f"\nDataset breakdown:")
    for name, dataset_stats in stats['datasets'].items():
        print(f"\n  {name}:")
        print(f"    Samples: {dataset_stats['test_size']}")
        print(f"    Reduction: {dataset_stats['reduction_pct']:.2f}%")
        print(f"    Improved: {dataset_stats['samples_improved']}/{dataset_stats['test_size']}")
    
    return stats


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 80)
    print("PATH D RESULTS ANALYSIS")
    print("=" * 80)
    
    # Load results
    print("\n[1/5] Loading results...")
    results = load_all_results()
    print(f"      Loaded {len(results)} datasets")
    
    # Generate summary table
    print("\n[2/5] Generating summary table...")
    df_summary = generate_summary_table(results)
    print(df_summary)
    
    # Generate key figures
    print("\n[3/5] Creating main figure (Real vs Synthetic)...")
    plot_violation_reduction(results)
    
    print("\n[4/5] Creating Detection vs Correction figure...")
    plot_detection_vs_correction(results)
    
    print("\n[5/5] Generating paper statistics...")
    stats = generate_paper_statistics(results)
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - summary_table.csv (for reference)")
    print("  - summary_table.tex (for paper)")
    print("  - violation_reduction_comparison.png (KEY FIGURE)")
    print("  - detection_vs_correction.png (dual contribution)")
    print("  - paper_statistics.json (numbers for writing)")
    print("\nAll files in: results/path_d_full_evaluation/")


if __name__ == "__main__":
    main()

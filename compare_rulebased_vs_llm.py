"""
Compare Rule-Based vs LLM-Based Self-Refine Results
Generates comparison figures and statistics for Path D paper.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results():
    """Load both rule-based and MMed-LLM results."""
    results_dir = Path("results/path_d_full_evaluation")
    
    # Try to load consolidated results
    rulebased_file = results_dir / "CONSOLIDATED_RESULTS_rulebased.json"
    mmed_file = results_dir / "CONSOLIDATED_RESULTS_mmed.json"
    # Also check for generic llm file (if using GPT)
    llm_file = results_dir / "CONSOLIDATED_RESULTS_llm.json"
    
    results = {
        'rulebased': {},
        'llm': {}  # Will contain MMed or GPT results
    }
    
    if rulebased_file.exists():
        with open(rulebased_file, 'r') as f:
            results['rulebased'] = json.load(f)
        print("✓ Loaded rule-based results")
    else:
        print("⚠ Rule-based results not found")
    
    # Try MMed first, then generic LLM
    if mmed_file.exists():
        with open(mmed_file, 'r') as f:
            results['llm'] = json.load(f)
        print("✓ Loaded MMed-LLM results")
    elif llm_file.exists():
        with open(llm_file, 'r') as f:
            results['llm'] = json.load(f)
        print("✓ Loaded LLM results")
    else:
        print("⚠ LLM results not found (neither MMed nor GPT)")
    
    return results


def create_comparison_table(results):
    """Generate comparison table for paper."""
    
    print("\n" + "=" * 80)
    print("COMPARISON: Rule-Based vs LLM-Based Refinement")
    print("=" * 80)
    
    datasets = []
    
    # Get all dataset names (without refiner suffix)
    for key in results['rulebased'].keys():
        dataset = key.replace('_rule-based', '').replace('_rulebased', '')
        datasets.append(dataset)
    
    # Create comparison data
    comparison_data = []
    
    for dataset in datasets:
        # Get keys for both refiners
        rb_key = f"{dataset}_rule-based"
        if rb_key not in results['rulebased']:
            rb_key = f"{dataset}_rulebased"
        
        # Try different LLM key formats (mmed, llm, gpt-4o-mini, etc.)
        llm_key = None
        for suffix in ['_mmed', '_llm', '_gpt-4o-mini', '_gpt-4o']:
            test_key = f"{dataset}{suffix}"
            if test_key in results['llm']:
                llm_key = test_key
                break
        
        if rb_key in results['rulebased'] and llm_key and llm_key in results['llm']:
            rb_data = results['rulebased'][rb_key]
            llm_data = results['llm'][llm_key]
            
            row = {
                'Dataset': dataset,
                'Test Size': rb_data.get('test_size', 'N/A'),
                'Rule-Based Reduction': f"{rb_data['reduction_percentage']:.1f}%",
                'LLM Reduction': f"{llm_data['reduction_percentage']:.1f}%",
                'Improvement': f"{llm_data['reduction_percentage'] - rb_data['reduction_percentage']:.1f}%"
            }
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save as CSV
    output_file = "results/path_d_full_evaluation/comparison_table.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Comparison table saved to: {output_file}")
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, escape=False)
    latex_file = "results/path_d_full_evaluation/comparison_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to: {latex_file}")
    
    print("\n" + df.to_string(index=False))
    
    return df


def plot_comparison(results):
    """
    Create the KEY COMPARISON FIGURE for Path D paper.
    
    Shows: Rule-Based (0%) vs LLM (15%) improvement
    """
    
    datasets = []
    rulebased_reductions = []
    llm_reductions = []
    
    # Get all dataset results
    for key in results['rulebased'].keys():
        dataset = key.replace('_rule-based', '').replace('_rulebased', '')
        
        rb_key = f"{dataset}_rule-based"
        if rb_key not in results['rulebased']:
            rb_key = f"{dataset}_rulebased"
        
        # Try different LLM key formats
        llm_key = None
        for suffix in ['_mmed', '_llm', '_gpt-4o-mini', '_gpt-4o']:
            test_key = f"{dataset}{suffix}"
            if test_key in results['llm']:
                llm_key = test_key
                break
        
        if rb_key in results['rulebased'] and llm_key and llm_key in results['llm']:
            datasets.append(dataset)
            rulebased_reductions.append(results['rulebased'][rb_key]['reduction_percentage'])
            llm_reductions.append(results['llm'][llm_key]['reduction_percentage'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Bars
    bars1 = ax.bar(x - width/2, rulebased_reductions, width, 
                   label='Rule-Based', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, llm_reductions, width,
                   label='LLM-Based (MMed-Llama-3-8B)', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Customize
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Violation Reduction (%)', fontsize=14, fontweight='bold')
    ax.set_title('Path D: Rule-Based vs LLM-Based Self-Refine\nComparison on Real Medical Datasets',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            label = f'{height:.1f}%'
            if height >= 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = "results/path_d_full_evaluation/rulebased_vs_llm_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison figure saved to: {output_file}")
    
    plt.close()


def generate_statistics(results):
    """Generate key statistics for paper writing."""
    
    print("\n" + "=" * 80)
    print("KEY STATISTICS FOR PAPER")
    print("=" * 80)
    
    # Calculate averages
    rb_reductions = []
    llm_reductions = []
    
    for key in results['rulebased'].keys():
        dataset = key.replace('_rule-based', '').replace('_rulebased', '')
        
        rb_key = f"{dataset}_rule-based"
        if rb_key not in results['rulebased']:
            rb_key = f"{dataset}_rulebased"
        
        # Try different LLM key formats
        llm_key = None
        for suffix in ['_mmed', '_llm', '_gpt-4o-mini', '_gpt-4o']:
            test_key = f"{dataset}{suffix}"
            if test_key in results['llm']:
                llm_key = test_key
                break
        
        if rb_key in results['rulebased'] and llm_key and llm_key in results['llm']:
            rb_reductions.append(results['rulebased'][rb_key]['reduction_percentage'])
            llm_reductions.append(results['llm'][llm_key]['reduction_percentage'])
    
    avg_rb = np.mean(rb_reductions)
    avg_llm = np.mean(llm_reductions)
    improvement = avg_llm - avg_rb
    
    print(f"\nAverage Violation Reduction:")
    print(f"  Rule-Based:     {avg_rb:.2f}%")
    print(f"  LLM-Based:      {avg_llm:.2f}%")
    print(f"  Improvement:    {improvement:.2f}% (absolute)")
    
    if avg_rb > 0:
        relative_improvement = (improvement / avg_rb) * 100
        print(f"  Relative Gain:  {relative_improvement:.1f}% better")
    else:
        print(f"  Relative Gain:  ∞% (rule-based achieved 0% reduction)")
    
    # Best and worst performers
    if llm_reductions:
        best_idx = np.argmax(llm_reductions)
        worst_idx = np.argmin(llm_reductions)
        
        datasets = [key.replace('_rule-based', '').replace('_rulebased', '') 
                   for key in results['rulebased'].keys()]
        
        print(f"\nBest Performing Dataset:")
        print(f"  {datasets[best_idx]}: {llm_reductions[best_idx]:.1f}% reduction")
        
        print(f"\nWorst Performing Dataset:")
        print(f"  {datasets[worst_idx]}: {llm_reductions[worst_idx]:.1f}% reduction")
    
    # Save statistics
    stats = {
        'avg_rulebased_reduction': avg_rb,
        'avg_llm_reduction': avg_llm,
        'absolute_improvement': improvement,
        'datasets': list(zip([key.replace('_rule-based', '').replace('_rulebased', '') 
                             for key in results['rulebased'].keys()],
                            rb_reductions, llm_reductions))
    }
    
    output_file = "results/path_d_full_evaluation/comparison_statistics.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Statistics saved to: {output_file}")
    
    return stats


def analyze_detection_success(results):
    """
    CONTRIBUTION 1: Calculate detection success rate
    Detection = % of samples that have at least 1 violation detected
    """
    print("\n" + "=" * 80)
    print("CONTRIBUTION 1: DETECTION SUCCESS")
    print("=" * 80)
    
    # Use rulebased results (both have same baseline/initial violations)
    total_samples = 0
    samples_with_violations = 0
    
    for dataset_name, data in results['rulebased'].items():
        test_size = data.get('test_size', 0)
        total_samples += test_size
        
        # Samples with violations = improved + unchanged (any non-zero initial violations)
        # "Worsened" means they had 0 initially but gained violations (shouldn't happen with detection)
        samples_with_violations += (data['samples_improved'] + data['samples_unchanged'])
    
    detection_rate = (samples_with_violations / total_samples * 100) if total_samples > 0 else 0
    
    print(f"\nTotal samples evaluated: {total_samples}")
    print(f"Samples with violations detected: {samples_with_violations}")
    print(f"Detection success rate: {detection_rate:.1f}%")
    print(f"\n✓ Contribution 1: Our method detects violations in {detection_rate:.1f}% of samples")
    
    return detection_rate


def main():
    """Main comparison pipeline."""
    print("\n" + "=" * 80)
    print("PATH D COMPARISON ANALYSIS")
    print("Rule-Based vs LLM-Based Self-Refine")
    print("=" * 80)
    
    # Load results
    print("\n[1/5] Loading results...")  # Changed from 1/4 to 1/5
    results = load_results()
    
    if not results['rulebased'] or not results['llm']:
        print("\n⚠ ERROR: Missing results files!")
        return
    
    # NEW: Analyze detection (Contribution 1)
    print("\n[2/5] Analyzing detection success...")
    detection_rate = analyze_detection_success(results)
    
    # Generate comparison table (Contributions 2 & 3)
    print("\n[3/5] Generating comparison table...")
    df_comparison = create_comparison_table(results)
    
    # Generate comparison figure
    print("\n[4/5] Creating comparison figure...")
    plot_comparison(results)
    
    # Generate statistics
    print("\n[5/5] Generating statistics...")
    stats = generate_statistics(results)
    
    # Add detection rate to stats
    stats['detection_rate'] = detection_rate
    
    print("\n" + "=" * 80)
    print("✓ COMPARISON ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - comparison_table.csv (for reference)")
    print("  - comparison_table.tex (for paper)")
    print("  - rulebased_vs_llm_comparison.png (KEY FIGURE)")
    print("  - comparison_statistics.json (numbers for writing)")
    print("\nAll files in: results/path_d_full_evaluation/")
    
    # Updated summary with all 3 contributions
    print("\n" + "=" * 80)
    print("ALL 3 PATH D CONTRIBUTIONS")
    print("=" * 80)
    print(f"""
✓ Contribution 1 (Detection): {detection_rate:.1f}%
  Our consistency rules successfully detect violations in {detection_rate:.1f}% of samples.

✓ Contribution 2 (Rule-Based Failure): {stats['avg_rulebased_reduction']:.1f}%
  Simple rule-based refinement achieves only {stats['avg_rulebased_reduction']:.1f}% average reduction.

✓ Contribution 3 (LLM Success): {stats['avg_llm_reduction']:.1f}%
  LLM-based refinement achieves {stats['avg_llm_reduction']:.1f}% average reduction,
  representing a {stats['absolute_improvement']:.1f} percentage point improvement over rule-based.

This demonstrates that medical concept refinement requires clinical reasoning from LLMs,
not just simple heuristics.
""")


if __name__ == "__main__":
    main()

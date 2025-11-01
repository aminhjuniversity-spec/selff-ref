"""
Option 1: Quick Test - Compare Violation Counts
This script tests self-refine on a small sample to see if it reduces inconsistencies.
"""

import sys
import os
sys.path.append('.')

from src.models.Explicd import Explicd
from src.utils import create_explicd_config, load_data
import numpy as np

def test_violation_reduction(dataset="PH2", split=0, num_samples=10):
    """
    Test if self-refine reduces consistency violations.
    
    Args:
        dataset: Dataset name (PH2, Derm7pt, HAM10000)
        split: Split number
        num_samples: Number of samples to test
    """
    print("#" * 80)
    print("# OPTION 1: Violation Count Comparison Test")
    print("#" * 80)
    print(f"\nDataset: {dataset}")
    print(f"Split: {split}")
    print(f"Testing on {num_samples} samples\n")
    
    # Load config and model
    print("Loading ExpLICD model...")
    config = create_explicd_config(gpu_id=0)
    model = Explicd(config=config)
    
    # Load data
    print("Loading dataset...")
    train_dataloader, test_dataloader = load_data(
        dataset=dataset, 
        split=split,
        data_path='/project/def-arashmoh/shahab33/Medsam/selff-ref/data'
    )
    
    # Storage for results
    results = {
        'image_ids': [],
        'baseline_violations': [],
        'refined_violations': [],
        'iterations': [],
        'converged': []
    }
    
    print("\nProcessing samples...")
    print("-" * 80)
    
    # Test on samples
    for i, batch in enumerate(test_dataloader):
        if i >= num_samples:
            break
        
        img_id = batch['img_id'][0]
        
        # WITHOUT self-refine (baseline)
        concepts_baseline, _, _ = model.get_concept_predictions_with_self_refine(
            batch=batch, 
            config=config,
            use_self_refine=False
        )
        
        # WITH self-refine
        concepts_refined, _, info = model.get_concept_predictions_with_self_refine(
            batch=batch, 
            config=config,
            use_self_refine=True
        )
        
        # Store results
        results['image_ids'].append(img_id)
        results['baseline_violations'].append(info['initial_violations'])
        results['refined_violations'].append(info['final_violations'])
        results['iterations'].append(info['iterations'])
        results['converged'].append(info['converged'])
        
        # Print per-sample results
        print(f"Image {img_id}:")
        print(f"  Baseline violations:  {info['initial_violations']}")
        print(f"  Refined violations:   {info['final_violations']}")
        print(f"  Reduction:            {info['initial_violations'] - info['final_violations']}")
        print(f"  Iterations:           {info['iterations']}")
        print(f"  Converged:            {'✓' if info['converged'] else '✗'}")
        print()
    
    # Calculate statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    baseline_avg = np.mean(results['baseline_violations'])
    refined_avg = np.mean(results['refined_violations'])
    reduction = baseline_avg - refined_avg
    reduction_pct = (reduction / baseline_avg * 100) if baseline_avg > 0 else 0
    
    print(f"\nAverage Violations:")
    print(f"  Baseline:             {baseline_avg:.2f}")
    print(f"  After Self-Refine:    {refined_avg:.2f}")
    print(f"  Reduction:            {reduction:.2f} ({reduction_pct:.1f}%)")
    
    print(f"\nConvergence Rate:")
    converged_count = sum(results['converged'])
    print(f"  Converged samples:    {converged_count}/{num_samples} ({converged_count/num_samples*100:.1f}%)")
    
    print(f"\nAverage Iterations:     {np.mean(results['iterations']):.2f}")
    
    # Breakdown by violation count
    print(f"\nViolation Distribution:")
    print(f"  Baseline - 0 violations:  {sum(1 for v in results['baseline_violations'] if v == 0)}")
    print(f"  Baseline - 1 violation:   {sum(1 for v in results['baseline_violations'] if v == 1)}")
    print(f"  Baseline - 2+ violations: {sum(1 for v in results['baseline_violations'] if v >= 2)}")
    print()
    print(f"  Refined - 0 violations:   {sum(1 for v in results['refined_violations'] if v == 0)}")
    print(f"  Refined - 1 violation:    {sum(1 for v in results['refined_violations'] if v == 1)}")
    print(f"  Refined - 2+ violations:  {sum(1 for v in results['refined_violations'] if v >= 2)}")
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if reduction > 0:
        print(f"✓ Self-refine REDUCES violations by {reduction:.2f} on average ({reduction_pct:.1f}%)")
        print("  → This is PROMISING! The self-refine mechanism is working.")
    elif reduction == 0:
        print("○ Self-refine shows NO CHANGE in violation counts")
        print("  → Consider tuning the consistency rules or refinement logic.")
    else:
        print(f"✗ Self-refine INCREASES violations by {abs(reduction):.2f} on average")
        print("  → The refinement logic may need revision.")
    
    print("\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Option 1: Violation Comparison')
    parser.add_argument('--dataset', type=str, default='PH2', help='Dataset name')
    parser.add_argument('--split', type=int, default=0, help='Split number')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to test')
    
    args = parser.parse_args()
    
    results = test_violation_reduction(
        dataset=args.dataset,
        split=args.split,
        num_samples=args.num_samples
    )
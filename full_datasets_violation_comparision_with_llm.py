"""
Path D: Full Dataset Violation Comparison WITH Mmed SUPPORT
Runs self-refine evaluation on ENTIRE test sets for all datasets.

NEW: Supports both rule-based and LLM-based refinement for comparison.
MODIFIED: Added --max_samples parameter for debugging on smaller subsets
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from src.models.Explicd import Explicd
from src.utils import create_explicd_config, load_data

# NEW: Import LLM refiners
from mmed_refiner import MMedBasedRefiner  # FREE - runs on cluster
# from llm_refiner import LLMBasedRefiner  # PAID - OpenAI API (optional)


def evaluate_full_dataset(dataset, split=None, data_path=None, use_llm=False, llm_model="MMed", max_samples=None):
    """
    Evaluate self-refine on complete test set (or subset for debugging).
    
    Args:
        dataset: Dataset name (PH2, Derm7pt, HAM10000)
        split: Split number (only for PH2)
        data_path: Path to data directory
        use_llm: Whether to use LLM-based refinement (default: False = rule-based)
        llm_model: Which LLM to use if use_llm=True ("MMed", "gpt-4o-mini", "gpt-4o")
        max_samples: Maximum number of samples to process (None = all, for debugging use 100)
    """
    refiner_type = f"LLM-{llm_model}" if use_llm else "Rule-Based"
    
    print("\n" + "=" * 80)
    print(f"Path D Evaluation: {dataset}" + (f" Split {split}" if split is not None else ""))
    print(f"Refiner: {refiner_type} ({llm_model if use_llm else 'SimpleRuleBasedRefiner'})")
    if max_samples:
        print(f"🔍 DEBUG MODE: Processing first {max_samples} samples only")
    print("=" * 80)
    
    # Load config and model
    print("\n[1/4] Loading ExpLICD model...")
    config = create_explicd_config(gpu_id=0)
    model = Explicd(config=config)
    
    # Initialize LLM refiner if requested
    if use_llm:
        print(f"[1/4] Initializing LLM refiner: {llm_model}...")
        
        if llm_model == "MMed":
            # Use MMed-LLM (FREE - runs on cluster)
            llm_refiner = MMedBasedRefiner(ckpt="Henrychur/MMed-Llama-3-8B")
        elif llm_model.startswith("gpt"):
            # Use OpenAI GPT (PAID - requires API key)
            from llm_refiner import LLMBasedRefiner
            llm_refiner = LLMBasedRefiner(model=llm_model)
        else:
            raise ValueError(f"Unknown LLM model: {llm_model}")
    else:
        llm_refiner = None
    
    # Load data
    print("[2/4] Loading dataset...")
    if data_path is None:
        data_path = '/project/def-arashmoh/shahab33/Medsam/selff-ref/data'
    
    train_dataloader, test_dataloader = load_data(
        dataset=dataset, 
        split=split,
        data_path=data_path
    )
    
    test_size = len(test_dataloader)
    
    # Apply max_samples limit if specified
    if max_samples:
        test_size = min(test_size, max_samples)
        print(f"      Full test set size: {len(test_dataloader)} samples")
        print(f"      Processing first: {test_size} samples (DEBUG MODE)")
    else:
        print(f"      Test set size: {test_size} samples")
    
    # Storage for results
    results = {
        'dataset': dataset,
        'split': split,
        'refiner_type': refiner_type,
        'llm_model': llm_model if use_llm else None,
        'timestamp': datetime.now().isoformat(),
        'test_size': test_size,
        'max_samples_limit': max_samples,
        'samples': []
    }
    
    print("\n[3/4] Processing samples...")
    print("-" * 80)
    
    violation_reduction_count = 0
    total_baseline_violations = 0
    total_refined_violations = 0
    
    # Process samples (limited to max_samples if specified)
    for idx, batch in enumerate(tqdm(test_dataloader, desc="Processing", total=test_size)):
        # Stop if we've reached max_samples
        if max_samples and idx >= max_samples:
            break
        
        img_id = batch['img_id'][0]
        
        # Baseline (no self-refine)
        concepts_baseline, _, _ = model.get_concept_predictions_with_self_refine(
            batch=batch, 
            config=config,
            use_self_refine=False
        )
        
        # With self-refine (rule-based OR LLM-based)
        concepts_refined, _, info = model.get_concept_predictions_with_self_refine(
            batch=batch, 
            config=config,
            use_self_refine=True,
            llm_refiner=llm_refiner  # Pass LLM refiner if available
        )
        
        # Track statistics
        baseline_viols = info['initial_violations']
        refined_viols = info['final_violations']
        
        total_baseline_violations += baseline_viols
        total_refined_violations += refined_viols
        
        if refined_viols < baseline_viols:
            violation_reduction_count += 1
        
        # Store sample result
        sample_result = {
            'image_id': img_id,
            'baseline_violations': baseline_viols,
            'refined_violations': refined_viols,
            'reduction': baseline_viols - refined_viols,
            'iterations': info['iterations'],
            'converged': info['converged']
        }
        results['samples'].append(sample_result)
    
    # Calculate summary statistics
    print("\n[4/4] Computing statistics...")
    
    avg_baseline = total_baseline_violations / test_size
    avg_refined = total_refined_violations / test_size
    avg_reduction = avg_baseline - avg_refined
    reduction_pct = (avg_reduction / avg_baseline * 100) if avg_baseline > 0 else 0
    
    samples_improved = sum(1 for s in results['samples'] if s['reduction'] > 0)
    samples_unchanged = sum(1 for s in results['samples'] if s['reduction'] == 0)
    samples_worsened = sum(1 for s in results['samples'] if s['reduction'] < 0)
    
    converged_count = sum(1 for s in results['samples'] if s['converged'])
    avg_iterations = np.mean([s['iterations'] for s in results['samples']])
    
    # Add summary to results
    results['summary'] = {
        'avg_baseline_violations': avg_baseline,
        'avg_refined_violations': avg_refined,
        'avg_reduction': avg_reduction,
        'reduction_percentage': reduction_pct,
        'samples_improved': samples_improved,
        'samples_unchanged': samples_unchanged,
        'samples_worsened': samples_worsened,
        'convergence_rate': converged_count / test_size,
        'avg_iterations': avg_iterations,
        'total_baseline_violations': total_baseline_violations,
        'total_refined_violations': total_refined_violations
    }
    
    # Save results
    output_dir = "results/path_d_full_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # File naming includes refiner type and debug suffix if applicable
    if use_llm:
        refiner_suffix = f"_{llm_model.lower()}"
    else:
        refiner_suffix = "_rulebased"
    
    debug_suffix = f"_debug{max_samples}" if max_samples else ""
    
    if split is not None:
        output_file = f"{output_dir}/{dataset}_split_{split}{refiner_suffix}{debug_suffix}_full_results.json"
    else:
        output_file = f"{output_dir}/{dataset}{refiner_suffix}{debug_suffix}_full_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nDataset: {dataset}" + (f" Split {split}" if split is not None else ""))
    print(f"Refiner: {refiner_type}")
    print(f"Test Size: {test_size} samples" + (f" (DEBUG: limited from {len(test_dataloader)})" if max_samples else ""))
    print(f"\nViolation Counts:")
    print(f"  Baseline Average:     {avg_baseline:.2f}")
    print(f"  Refined Average:      {avg_refined:.2f}")
    print(f"  Reduction:            {avg_reduction:.2f} ({reduction_pct:.1f}%)")
    print(f"\nSample Distribution:")
    print(f"  Improved:             {samples_improved}/{test_size} ({samples_improved/test_size*100:.1f}%)")
    print(f"  Unchanged:            {samples_unchanged}/{test_size} ({samples_unchanged/test_size*100:.1f}%)")
    print(f"  Worsened:             {samples_worsened}/{test_size} ({samples_worsened/test_size*100:.1f}%)")
    print(f"\nConvergence:")
    print(f"  Converged:            {converged_count}/{test_size} ({converged_count/test_size*100:.1f}%)")
    print(f"  Avg Iterations:       {avg_iterations:.2f}")
    print("=" * 80)
    
    return results


def evaluate_all_datasets(use_llm=False, llm_model="MMed", max_samples=None):
    """
    Run Path D evaluation on ALL datasets and splits.
    
    Args:
        use_llm: Whether to use LLM-based refinement
        llm_model: Which LLM to use ("MMed", "gpt-4o-mini", "gpt-4o")
        max_samples: Maximum samples per dataset (None = all, for debugging use 100)
    """
    refiner_type = f"LLM-{llm_model}" if use_llm else "Rule-Based"
    
    print("\n" + "=" * 80)
    print("PATH D: COMPREHENSIVE EVALUATION")
    print(f"Refiner Type: {refiner_type}")
    if max_samples:
        print(f"🔍 DEBUG MODE: Processing max {max_samples} samples per dataset")
    print("Evaluating Self-Refine on All Datasets")
    print("=" * 80)
    
    all_results = {}
    
    # PH2: All 5 splits
    print("\n[DATASET 1/3] PH2 (5 splits)")
    for split in range(5):
        results = evaluate_full_dataset(
            dataset='PH2', 
            split=split,
            use_llm=use_llm,
            llm_model=llm_model,
            max_samples=max_samples
        )
        key = f'PH2_split_{split}_{refiner_type.lower()}'
        all_results[key] = results['summary']
    
    # Derm7pt
    print("\n[DATASET 2/3] Derm7pt")
    results = evaluate_full_dataset(
        dataset='Derm7pt', 
        split=None,
        use_llm=use_llm,
        llm_model=llm_model,
        max_samples=max_samples
    )
    key = f'Derm7pt_{refiner_type.lower()}'
    all_results[key] = results['summary']
    
    # HAM10000
    print("\n[DATASET 3/3] HAM10000")
    results = evaluate_full_dataset(
        dataset='HAM10000', 
        split=None,
        use_llm=use_llm,
        llm_model=llm_model,
        max_samples=max_samples
    )
    key = f'HAM10000_{refiner_type.lower()}'
    all_results[key] = results['summary']
    
    # Save consolidated results
    if use_llm:
        refiner_suffix = f"_{llm_model.lower()}"
    else:
        refiner_suffix = "_rulebased"
    
    debug_suffix = f"_debug{max_samples}" if max_samples else ""
    
    output_file = f"results/path_d_full_evaluation/CONSOLIDATED_RESULTS{refiner_suffix}{debug_suffix}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ ALL DATASETS COMPLETED")
    print("=" * 80)
    print(f"\nConsolidated results saved to: {output_file}")
    if max_samples:
        print(f"\n⚠️  DEBUG MODE: Results are based on only {max_samples} samples per dataset")
        print("   To run on full datasets, remove --max_samples parameter")
    print("\nNext steps:")
    print("  1. Review results in results/path_d_full_evaluation/")
    print("  2. Generate figures for Path D paper")
    print("  3. Compare Rule-Based vs LLM results")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Path D: Full Dataset Evaluation (with DEBUG mode)')
    parser.add_argument('--dataset', type=str, default='all', 
                        choices=['all', 'PH2', 'Derm7pt', 'HAM10000'],
                        help='Which dataset to evaluate')
    parser.add_argument('--split', type=int, default=None,
                        help='Split number (only for PH2)')
    parser.add_argument('--data_path', type=str,
                        default='/project/def-arashmoh/shahab33/Medsam/selff-ref/data',
                        help='Path to data directory')
    
    # LLM options
    parser.add_argument('--use_llm', action='store_true',
                        help='Use LLM-based refinement instead of rule-based')
    parser.add_argument('--llm_model', type=str, default='MMed',
                        choices=['MMed', 'gpt-4o-mini', 'gpt-4o'],
                        help='Which LLM to use (MMed is free, GPT requires API key)')
    
    # NEW: Debug option
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to process per dataset (for debugging, e.g. 100)')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        evaluate_all_datasets(
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            max_samples=args.max_samples
        )
    else:
        evaluate_full_dataset(
            dataset=args.dataset,
            split=args.split,
            data_path=args.data_path,
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            max_samples=args.max_samples
        )

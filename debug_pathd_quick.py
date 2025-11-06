"""
QUICK DEBUG VERSION - Path D + x->c->y
Tests your contributions on PH2 split 0 ONLY (smallest dataset, ~20 samples)
Runtime: ~10 minutes instead of 24+ hours
"""

import sys
sys.path.append('.')

import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime

from src.models.Explicd import Explicd
from src.utils import create_explicd_config, load_data, map_label_to_name
from mmed_refiner import MMedBasedRefiner


def quick_pathd_test(use_llm=False, max_samples=20):
    """
    Quick Path D test on PH2 split 0 (small dataset)
    
    Args:
        use_llm: Test LLM-based refinement (True) or rule-based (False)
        max_samples: Limit number of samples (default: 20 for speed)
    """
    refiner_type = "LLM-MMed" if use_llm else "Rule-Based"
    
    print("\n" + "=" * 80)
    print(f"QUICK DEBUG: Path D - {refiner_type}")
    print(f"Dataset: PH2 split 0 (first {max_samples} samples)")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading ExpLICD model...")
    config = create_explicd_config(gpu_id=0)
    model = Explicd(config=config)
    
    # Load data
    print("[2/3] Loading PH2 split 0...")
    _, test_dataloader = load_data(dataset='PH2', split=0, data_path='data')
    
    # Initialize LLM refiner if requested
    llm_refiner = MMedBasedRefiner() if use_llm else None
    
    # Process samples
    print(f"\n[3/3] Processing {max_samples} samples...")
    
    total_baseline_violations = 0
    total_refined_violations = 0
    samples_improved = 0
    
    for idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
        if idx >= max_samples:  # Stop after N samples
            break
        
        img_id = batch['img_id'][0]
        
        # Baseline (no self-refine)
        concepts_baseline, _, _ = model.get_concept_predictions_with_self_refine(
            batch=batch, 
            config=config,
            use_self_refine=False
        )
        
        # With self-refine
        concepts_refined, _, info = model.get_concept_predictions_with_self_refine(
            batch=batch, 
            config=config,
            use_self_refine=True,
            llm_refiner=llm_refiner
        )
        
        # Track statistics
        baseline_viols = info['initial_violations']
        refined_viols = info['final_violations']
        
        total_baseline_violations += baseline_viols
        total_refined_violations += refined_viols
        
        if refined_viols < baseline_viols:
            samples_improved += 1
    
    # Calculate results
    avg_baseline = total_baseline_violations / max_samples
    avg_refined = total_refined_violations / max_samples
    avg_reduction = avg_baseline - avg_refined
    reduction_pct = (avg_reduction / avg_baseline * 100) if avg_baseline > 0 else 0
    
    # Print results
    print("\n" + "=" * 80)
    print(f"QUICK TEST RESULTS - {refiner_type}")
    print("=" * 80)
    print(f"Samples tested: {max_samples}")
    print(f"Avg baseline violations: {avg_baseline:.2f}")
    print(f"Avg refined violations:  {avg_refined:.2f}")
    print(f"Reduction:               {avg_reduction:.2f} ({reduction_pct:.1f}%)")
    print(f"Samples improved:        {samples_improved}/{max_samples}")
    print("=" * 80)
    
    return {
        'refiner': refiner_type,
        'reduction_pct': reduction_pct,
        'samples_improved': samples_improved
    }


def quick_xcy_test(max_samples=20):
    """
    Quick x->c->y test to verify diagnosis accuracy
    
    Args:
        max_samples: Limit number of samples (default: 20)
    """
    print("\n" + "=" * 80)
    print(f"QUICK DEBUG: x->c->y Pipeline")
    print(f"Dataset: PH2 split 0 (first {max_samples} samples)")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading ExpLICD model...")
    config = create_explicd_config(gpu_id=0)
    model = Explicd(config=config)
    
    # Load data
    print("[2/3] Loading PH2 split 0...")
    _, test_dataloader = load_data(dataset='PH2', split=0, data_path='data')
    
    # Process samples
    print(f"\n[3/3] Generating concepts for {max_samples} samples...")
    
    dict_to_save_data = {}
    
    for idx, batch in enumerate(tqdm(test_dataloader, desc="x->c")):
        if idx >= max_samples:
            break
        
        img_ids = batch["img_id"]
        y_true = batch["class_label"].numpy()
        
        # Get refined concepts
        predicted_concepts, _, info = model.get_concept_predictions_with_self_refine(
            batch=batch, 
            config=config,
            use_self_refine=True,
            llm_refiner=MMedBasedRefiner()
        )
        
        # Add diagnosis
        report = predicted_concepts + f" Thus the diagnosis is {map_label_to_name(y_true)}."
        dict_to_save_data[img_ids[0]] = report
    
    # Save to CSV
    df = pd.DataFrame.from_dict(dict_to_save_data, orient='index', columns=['report'])
    df = df.reset_index()
    df.columns = ["image_id", "report"]
    
    output_file = "results/debug_concept_prediction_PH2_split_0.csv"
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("QUICK x->c->y TEST COMPLETE")
    print("=" * 80)
    print(f"Generated {len(dict_to_save_data)} concept reports")
    print(f"Saved to: {output_file}")
    print("\nNext: Run c->y step with your LLM to get diagnosis accuracy")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Debug Test')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'pathd', 'xcy'],
                        help='Which test to run')
    parser.add_argument('--max_samples', type=int, default=20,
                        help='Number of samples to test (default: 20)')
    
    args = parser.parse_args()
    
    if args.test in ['all', 'pathd']:
        # Test Path D - Rule-based
        print("\n" + "🔧 TESTING RULE-BASED REFINEMENT...")
        result_rb = quick_pathd_test(use_llm=False, max_samples=args.max_samples)
        
        # Test Path D - LLM-based
        print("\n" + "🤖 TESTING LLM-BASED REFINEMENT...")
        result_llm = quick_pathd_test(use_llm=True, max_samples=args.max_samples)
        
        # Summary
        print("\n" + "=" * 80)
        print("PATH D COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Rule-Based: {result_rb['reduction_pct']:.1f}% reduction")
        print(f"LLM-Based:  {result_llm['reduction_pct']:.1f}% reduction")
        print(f"Improvement: {result_llm['reduction_pct'] - result_rb['reduction_pct']:.1f}%")
        print("=" * 80)
    
    if args.test in ['all', 'xcy']:
        # Test x->c->y
        print("\n" + "📊 TESTING x->c->y PIPELINE...")
        quick_xcy_test(max_samples=args.max_samples)
    
    print("\n✅ DEBUG TESTS COMPLETE!")
    print("\nIf everything works, run full experiments:")
    print("  sbatch scripts/run_pathd_rulebased.sh")
    print("  sbatch scripts/run_pathd_llm.sh")
    print("  sbatch scripts/main.sh")

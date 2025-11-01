"""
Option 2: Generate Concept Reports and Compare
This script generates concept predictions with and without self-refine,
then compares the outputs side-by-side.
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import gc
import torch

from src.models.Explicd import Explicd
from src.utils import create_explicd_config, load_data, map_label_to_name

def generate_concepts_comparison(dataset="PH2", split=0, data_path=None):
    """
    Generate concept predictions with and without self-refine.
    
    Args:
        dataset: Dataset name
        split: Split number
        data_path: Path to data directory
    """
    print("#" * 80)
    print("# OPTION 2: Generate and Compare Concept Reports")
    print("#" * 80)
    print(f"\nDataset: {dataset}")
    print(f"Split: {split}\n")
    
    # Load data
    print("Loading data...")
    train_dataloader, test_dataloader = load_data(
        dataset=dataset, 
        split=split,
        data_path=data_path or '/project/def-arashmoh/shahab33/Medsam/selff-ref/data'
    )
    
    # Initialize ExpLICD model
    print("Loading ExpLICD model...")
    config = create_explicd_config(gpu_id=0)
    model = Explicd(config=config)
    
    # Storage for both versions
    baseline_data = {'image_id': [], 'concepts': [], 'violations': []}
    refined_data = {'image_id': [], 'concepts': [], 'violations': [], 'iterations': [], 'converged': []}
    
    print("\n" + "=" * 80)
    print("GENERATING BASELINE CONCEPTS (No Self-Refine)")
    print("=" * 80)
    
    # Generate baseline concepts
    for batch in tqdm(test_dataloader, desc="Baseline"):
        img_id = batch["img_id"][0]
        
        # Get predictions WITHOUT self-refine
        concepts, raw_scores, info = model.get_concept_predictions_with_self_refine(
            batch=batch, 
            config=config,
            use_self_refine=False
        )
        
        baseline_data['image_id'].append(img_id)
        baseline_data['concepts'].append(concepts)
        baseline_data['violations'].append(0)  # No violation info for baseline
    
    print("\n" + "=" * 80)
    print("GENERATING REFINED CONCEPTS (With Self-Refine)")
    print("=" * 80)
    
    # Generate refined concepts
    for batch in tqdm(test_dataloader, desc="Refined"):
        img_id = batch["img_id"][0]
        
        # Get predictions WITH self-refine
        concepts, raw_scores, info = model.get_concept_predictions_with_self_refine(
            batch=batch, 
            config=config,
            use_self_refine=True
        )
        
        refined_data['image_id'].append(img_id)
        refined_data['concepts'].append(concepts)
        refined_data['violations'].append(info['final_violations'])
        refined_data['iterations'].append(info['iterations'])
        refined_data['converged'].append(info['converged'])
    
    # Create DataFrames
    df_baseline = pd.DataFrame(baseline_data)
    df_refined = pd.DataFrame(refined_data)
    
    # Merge for comparison
    df_comparison = pd.merge(
        df_baseline, 
        df_refined, 
        on='image_id', 
        suffixes=('_baseline', '_refined')
    )
    
    # Save results
    output_dir = f"results/self_refine_comparison/{dataset}_split_{split}"
    os.makedirs(output_dir, exist_ok=True)
    
    baseline_path = f"{output_dir}/concepts_baseline.csv"
    refined_path = f"{output_dir}/concepts_refined.csv"
    comparison_path = f"{output_dir}/concepts_comparison.csv"
    
    df_baseline.to_csv(baseline_path, index=False)
    df_refined.to_csv(refined_path, index=False)
    df_comparison.to_csv(comparison_path, index=False)
    
    print(f"\n✓ Saved baseline concepts to: {baseline_path}")
    print(f"✓ Saved refined concepts to: {refined_path}")
    print(f"✓ Saved comparison to: {comparison_path}")
    
    # Generate summary statistics
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # Count how many concepts changed
    changed = sum(df_comparison['concepts_baseline'] != df_comparison['concepts_refined'])
    print(f"\nConcepts Changed: {changed}/{len(df_comparison)} ({changed/len(df_comparison)*100:.1f}%)")
    
    # Violation statistics
    print(f"\nRefined Violation Statistics:")
    print(f"  Average violations: {df_refined['violations'].mean():.2f}")
    print(f"  Samples with 0 violations: {sum(df_refined['violations'] == 0)}/{len(df_refined)}")
    print(f"  Convergence rate: {sum(df_refined['converged'])}/{len(df_refined)} ({sum(df_refined['converged'])/len(df_refined)*100:.1f}%)")
    print(f"  Average iterations: {df_refined['iterations'].mean():.2f}")
    
    # Show examples of changes
    print("\n" + "=" * 80)
    print("EXAMPLE CHANGES (First 3 Modified Samples)")
    print("=" * 80)
    
    changed_samples = df_comparison[df_comparison['concepts_baseline'] != df_comparison['concepts_refined']]
    
    for i, row in changed_samples.head(3).iterrows():
        print(f"\nImage: {row['image_id']}")
        print(f"Violations: {row['violations_refined']}")
        print(f"Iterations: {row['iterations']}")
        print(f"\nBaseline:")
        print(f"  {row['concepts_baseline'][:200]}...")
        print(f"\nRefined:")
        print(f"  {row['concepts_refined'][:200]}...")
        print("-" * 80)
    
    # Clean up
    del model
    del test_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✓ Done! Check the output files for detailed comparison.")
    
    return df_comparison


def analyze_concept_differences(comparison_csv_path):
    """
    Detailed analysis of differences between baseline and refined concepts.
    
    Args:
        comparison_csv_path: Path to comparison CSV file
    """
    print("\n" + "=" * 80)
    print("DETAILED DIFFERENCE ANALYSIS")
    print("=" * 80)
    
    df = pd.read_csv(comparison_csv_path)
    
    # Identify which concept attributes changed most frequently
    concept_keys = ['color', 'shape', 'border', 'dermoscopic patterns', 'texture', 'symmetry', 'elevation']
    
    changes_by_concept = {key: 0 for key in concept_keys}
    
    for _, row in df.iterrows():
        baseline = row['concepts_baseline']
        refined = row['concepts_refined']
        
        # Simple heuristic: check if each concept key's value changed
        for key in concept_keys:
            # Extract the value for this concept in both versions
            baseline_part = baseline[baseline.find(f"the {key}"):baseline.find(f"the {key}") + 200]
            refined_part = refined[refined.find(f"the {key}"):refined.find(f"the {key}") + 200]
            
            if baseline_part != refined_part:
                changes_by_concept[key] += 1
    
    print("\nMost Frequently Modified Concepts:")
    sorted_changes = sorted(changes_by_concept.items(), key=lambda x: x[1], reverse=True)
    for concept, count in sorted_changes:
        print(f"  {concept:20s}: {count:3d} changes")
    
    print("\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Option 2: Generate and Compare Reports')
    parser.add_argument('--dataset', type=str, default='PH2', help='Dataset name')
    parser.add_argument('--split', type=int, default=0, help='Split number')
    parser.add_argument('--data_path', type=str, 
                        default='/project/def-arashmoh/shahab33/Medsam/selff-ref/data',
                        help='Path to data directory')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Path to comparison CSV for detailed analysis')
    
    args = parser.parse_args()
    
    if args.analyze:
        # Analyze existing comparison file
        analyze_concept_differences(args.analyze)
    else:
        # Generate new comparison
        df_comparison = generate_concepts_comparison(
            dataset=args.dataset,
            split=args.split,
            data_path=args.data_path
        )
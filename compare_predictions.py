"""
Compare Baseline vs Self-Refined ExpLICD Predictions

This script compares concept predictions from ExpLICD with and without self-refine,
showing which images were changed and how they differ.

Usage:
    python compare_predictions.py --dataset PH2 --split 0
"""

import pandas as pd
import argparse
from pathlib import Path


def load_predictions(dataset, split, baseline_path=None, refined_path=None):
    """Load baseline and refined predictions"""
    
    if baseline_path is None:
        baseline_path = f"results/concept_prediction/{dataset}_split_{split}_baseline.csv"
    if refined_path is None:
        refined_path = f"results/concept_prediction/{dataset}_split_{split}_refined.csv"
    
    try:
        baseline = pd.read_csv(baseline_path)
        refined = pd.read_csv(refined_path)
        return baseline, refined
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Tip: Make sure you've generated both baseline and refined predictions:")
        print(f"   1. Set use_self_refine=False and run: python run_x_to_c_to_y.py --model Explicd --dataset {dataset} --split {split}")
        print(f"   2. Rename: mv results/concept_prediction/{dataset}_split_{split}_dermatology_reports_generated_by_Explicd_raw_values_False.csv {baseline_path}")
        print(f"   3. Set use_self_refine=True and run: python run_x_to_c_to_y.py --model Explicd --dataset {dataset} --split {split}")
        print(f"   4. Rename: mv results/concept_prediction/{dataset}_split_{split}_dermatology_reports_generated_by_Explicd_raw_values_False.csv {refined_path}")
        return None, None


def extract_concepts(report):
    """Extract concept portion from report (before diagnosis)"""
    if "Thus the diagnosis is" in report:
        return report[:report.find("Thus the diagnosis is")].strip()
    return report


def compare_reports(baseline, refined):
    """Compare baseline and refined reports"""
    
    # Merge on image_id
    merged = baseline.merge(refined, on='image_id', suffixes=('_baseline', '_refined'))
    
    # Find differences
    merged['concepts_baseline'] = merged['report_baseline'].apply(extract_concepts)
    merged['concepts_refined'] = merged['report_refined'].apply(extract_concepts)
    merged['changed'] = merged['concepts_baseline'] != merged['concepts_refined']
    
    different = merged[merged['changed']]
    
    return merged, different


def analyze_changes(different):
    """Analyze what changed between baseline and refined"""
    
    concept_keys = ['color', 'shape', 'border', 'dermoscopic patterns', 'texture', 'symmetry', 'elevation']
    
    changes_by_concept = {key: 0 for key in concept_keys}
    
    for _, row in different.iterrows():
        baseline_concepts = row['concepts_baseline']
        refined_concepts = row['concepts_refined']
        
        for key in concept_keys:
            # Extract concept value for this key
            baseline_val = extract_concept_value(baseline_concepts, key)
            refined_val = extract_concept_value(refined_concepts, key)
            
            if baseline_val != refined_val:
                changes_by_concept[key] += 1
    
    return changes_by_concept


def extract_concept_value(concepts_str, concept_key):
    """Extract the value for a specific concept from the concepts string"""
    import re
    
    pattern = rf"the {concept_key} (?:is|are) ([^,\.]+)"
    match = re.search(pattern, concepts_str, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    return None


def print_statistics(baseline, different, changes_by_concept):
    """Print comparison statistics"""
    
    total = len(baseline)
    changed = len(different)
    unchanged = total - changed
    
    print("\n" + "="*70)
    print("📊 COMPARISON STATISTICS")
    print("="*70)
    
    print(f"\n📈 Overall:")
    print(f"   Total images:          {total}")
    print(f"   Unchanged:             {unchanged} ({unchanged/total*100:.1f}%)")
    print(f"   Changed by self-refine: {changed} ({changed/total*100:.1f}%)")
    
    print(f"\n🔄 Changes by concept:")
    for concept, count in sorted(changes_by_concept.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"   {concept:20s}: {count:3d} ({count/changed*100:.1f}% of changed images)")
    
    print("\n" + "="*70)


def print_examples(different, n=5):
    """Print example changes"""
    
    print(f"\n📝 EXAMPLES OF CHANGES (showing {min(n, len(different))} of {len(different)}):")
    print("="*70)
    
    for idx, row in different.head(n).iterrows():
        print(f"\n🖼️  Image: {row['image_id']}")
        print(f"   BASELINE: {row['concepts_baseline']}")
        print(f"   REFINED:  {row['concepts_refined']}")
        
        # Highlight the specific differences
        baseline_concepts = row['concepts_baseline']
        refined_concepts = row['concepts_refined']
        
        concept_keys = ['color', 'shape', 'border', 'dermoscopic patterns', 'texture', 'symmetry', 'elevation']
        differences = []
        
        for key in concept_keys:
            baseline_val = extract_concept_value(baseline_concepts, key)
            refined_val = extract_concept_value(refined_concepts, key)
            
            if baseline_val != refined_val and baseline_val is not None and refined_val is not None:
                differences.append(f"{key}: '{baseline_val}' → '{refined_val}'")
        
        if differences:
            print(f"   CHANGES:")
            for diff in differences:
                print(f"      • {diff}")
    
    print("\n" + "="*70)


def save_detailed_report(different, output_path):
    """Save detailed comparison report to CSV"""
    
    output_df = different[['image_id', 'concepts_baseline', 'concepts_refined']].copy()
    output_df.to_csv(output_path, index=False)
    print(f"\n💾 Detailed comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs self-refined predictions')
    parser.add_argument('--dataset', type=str, default='PH2', help='Dataset name')
    parser.add_argument('--split', type=int, default=0, help='Dataset split')
    parser.add_argument('--baseline_path', type=str, default=None, help='Path to baseline predictions')
    parser.add_argument('--refined_path', type=str, default=None, help='Path to refined predictions')
    parser.add_argument('--n_examples', type=int, default=5, help='Number of examples to show')
    parser.add_argument('--save_report', action='store_true', help='Save detailed comparison report')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔬 ExpLICD Self-Refine Comparison Tool")
    print("="*70)
    print(f"\nDataset: {args.dataset}")
    print(f"Split:   {args.split}")
    
    # Load predictions
    print("\n📂 Loading predictions...")
    baseline, refined = load_predictions(args.dataset, args.split, args.baseline_path, args.refined_path)
    
    if baseline is None or refined is None:
        return
    
    print(f"   ✅ Loaded {len(baseline)} baseline predictions")
    print(f"   ✅ Loaded {len(refined)} refined predictions")
    
    # Compare
    print("\n🔍 Comparing predictions...")
    merged, different = compare_reports(baseline, refined)
    
    # Analyze changes
    changes_by_concept = analyze_changes(different)
    
    # Print statistics
    print_statistics(baseline, different, changes_by_concept)
    
    # Print examples
    if len(different) > 0:
        print_examples(different, n=args.n_examples)
    else:
        print("\n✅ No differences found! Baseline and refined predictions are identical.")
    
    # Save detailed report if requested
    if args.save_report and len(different) > 0:
        output_path = f"results/comparison_{args.dataset}_split_{args.split}_detailed.csv"
        save_detailed_report(different, output_path)
    
    # Summary
    print("\n📋 SUMMARY:")
    if len(different) == 0:
        print("   Self-refine made no changes to the predictions.")
        print("   This might indicate:")
        print("   • All predictions were already consistent")
        print("   • Rules are not detecting violations")
        print("   • Self-refine was not actually enabled")
    else:
        print(f"   Self-refine modified {len(different)} predictions ({len(different)/len(baseline)*100:.1f}%)")
        print("   Most common changes:")
        top_3 = sorted(changes_by_concept.items(), key=lambda x: x[1], reverse=True)[:3]
        for concept, count in top_3:
            if count > 0:
                print(f"      • {concept}: {count} changes")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

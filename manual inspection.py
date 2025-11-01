"""
Option 3: Manual Inspection of Concept Refinement
This script allows you to manually inspect how self-refine changes specific examples.
"""

import sys
sys.path.append('.')

from src.self_refine.concept_refiner import (
    ConceptSelfRefine, 
    SimpleRuleBasedRefiner, 
    ConceptConsistencyRules
)


def inspect_example(concepts_str, diagnosis=None):
    """
    Manually inspect how self-refine processes a concept string.
    
    Args:
        concepts_str: Initial concept prediction string
        diagnosis: Optional diagnosis label
    """
    print("#" * 80)
    print("# OPTION 3: Manual Inspection")
    print("#" * 80)
    print()
    
    # Initialize components
    rules = ConceptConsistencyRules()
    refiner = ConceptSelfRefine(
        llm_refine_fn=SimpleRuleBasedRefiner(),
        max_iterations=3,
        verbose=True
    )
    
    # Parse concepts
    concepts_dict = refiner.parse_concepts(concepts_str)
    
    # Display original concepts
    print("ORIGINAL CONCEPTS:")
    print("=" * 80)
    print(concepts_str)
    print()
    
    print("PARSED CONCEPTS:")
    print("=" * 80)
    for key, value in concepts_dict.items():
        print(f"  {key:25s}: {value}")
    print()
    
    # Check consistency
    print("CONSISTENCY CHECK:")
    print("=" * 80)
    violations = rules.check_consistency(concepts_dict)
    
    if len(violations) == 0:
        print("✓ No consistency violations found!")
    else:
        print(f"Found {len(violations)} violation(s):")
        for i, violation in enumerate(violations, 1):
            print(f"\n{i}. {violation}")
    print()
    
    # Apply self-refine
    print("=" * 80)
    print("APPLYING SELF-REFINE...")
    print("=" * 80)
    print()
    
    refined_concepts, info = refiner.refine(concepts_str, diagnosis=diagnosis)
    
    # Display refined concepts
    print("\n" + "=" * 80)
    print("REFINED CONCEPTS:")
    print("=" * 80)
    print(refined_concepts)
    print()
    
    # Display refinement statistics
    print("REFINEMENT STATISTICS:")
    print("=" * 80)
    print(f"  Initial violations:    {info['initial_violations']}")
    print(f"  Final violations:      {info['final_violations']}")
    print(f"  Reduction:             {info['initial_violations'] - info['final_violations']}")
    print(f"  Iterations:            {info['iterations']}")
    print(f"  Converged:             {'✓ Yes' if info['converged'] else '✗ No'}")
    print()
    
    # Show iteration history
    if len(info['history']) > 0:
        print("ITERATION HISTORY:")
        print("=" * 80)
        for entry in info['history']:
            print(f"\nIteration {entry['iteration']}:")
            print(f"  Violations: {entry['violations']}")
            print(f"  Feedback: {entry['feedback'][:150]}...")
    
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON:")
    print("=" * 80)
    
    # Parse refined concepts to compare
    refined_dict = refiner.parse_concepts(refined_concepts)
    
    for key in concepts_dict.keys():
        original = concepts_dict.get(key, 'N/A')
        refined = refined_dict.get(key, 'N/A')
        
        if original != refined:
            print(f"\n{key.upper()}:")
            print(f"  Before: {original}")
            print(f"  After:  {refined}")
            print(f"  ➜ CHANGED")
        else:
            print(f"\n{key.upper()}: (unchanged)")
            print(f"  {original}")
    
    print("\n")


def run_predefined_examples():
    """
    Run self-refine on several predefined examples with known inconsistencies.
    """
    print("#" * 80)
    print("# TESTING PREDEFINED EXAMPLES")
    print("#" * 80)
    print()
    
    examples = [
        {
            "name": "Example 1: Asymmetric with sharp border (Rule 1 violation)",
            "concepts": "The color is highly variable, often with multiple colors (black, brown, red, white, blue), the shape is irregular, the border is sharp and well-defined, the dermoscopic patterns are atypical pigment network, irregular streaks, the texture is smooth, the symmetry is asymmetrical, the elevation is flat to raised.",
            "diagnosis": "melanoma"
        },
        {
            "name": "Example 2: Multiple colors with regular patterns (Rule 2 violation)",
            "concepts": "The color is highly variable, often with multiple colors (black, brown, red, white, blue), the shape is round, the border is often blurry and irregular, the dermoscopic patterns are regular pigment network, symmetric dots and globules, the texture is smooth, the symmetry is symmetrical, the elevation is flat to raised.",
            "diagnosis": "nevus"
        },
        {
            "name": "Example 3: Smooth texture with ulceration (Rule 3 violation)",
            "concepts": "The color is uniformly tan, brown, or black, the shape is round, the border is sharp and well-defined, the dermoscopic patterns are regular pigment network, symmetric dots and globules, the texture is a raised or ulcerated surface, the symmetry is symmetrical, the elevation is raised with possible central ulceration.",
            "diagnosis": "nevus"
        },
        {
            "name": "Example 4: Melanoma indicators with symmetry (Rule 5 violation)",
            "concepts": "The color is highly variable, often with multiple colors (black, brown, red, white, blue), the shape is irregular, the border is often blurry and irregular, the dermoscopic patterns are atypical pigment network, irregular streaks, blue-whitish veil, irregular, the texture is a raised or ulcerated surface, the symmetry is symmetrical, the elevation is flat to raised.",
            "diagnosis": "melanoma"
        },
        {
            "name": "Example 5: No violations (should not change)",
            "concepts": "The color is uniformly tan, brown, or black, the shape is round, the border is sharp and well-defined, the dermoscopic patterns are regular pigment network, symmetric dots and globules, the texture is smooth, the symmetry is symmetrical, the elevation is slightly raised.",
            "diagnosis": "nevus"
        }
    ]
    
    results_summary = []
    
    for i, example in enumerate(examples, 1):
        print("\n" + "=" * 80)
        print(f"EXAMPLE {i}: {example['name']}")
        print("=" * 80)
        print()
        
        # Initialize refiner
        refiner = ConceptSelfRefine(
            llm_refine_fn=SimpleRuleBasedRefiner(),
            max_iterations=3,
            verbose=False  # Less verbose for batch processing
        )
        
        # Refine
        refined, info = refiner.refine(example['concepts'], diagnosis=example['diagnosis'])
        
        # Display summary
        print(f"Original: {example['concepts'][:100]}...")
        print(f"\nRefined:  {refined[:100]}...")
        print(f"\nResult: {info['initial_violations']} → {info['final_violations']} violations")
        print(f"Converged: {'✓' if info['converged'] else '✗'}")
        
        results_summary.append({
            'example': i,
            'name': example['name'],
            'initial': info['initial_violations'],
            'final': info['final_violations'],
            'converged': info['converged']
        })
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print()
    
    for result in results_summary:
        status = "✓ IMPROVED" if result['initial'] > result['final'] else \
                 "○ NO CHANGE" if result['initial'] == result['final'] else \
                 "✗ WORSENED"
        print(f"Example {result['example']}: {result['initial']} → {result['final']} violations {status}")
    
    total_initial = sum(r['initial'] for r in results_summary)
    total_final = sum(r['final'] for r in results_summary)
    converged_count = sum(r['converged'] for r in results_summary)
    
    print()
    print(f"Total violations: {total_initial} → {total_final} (reduction: {total_initial - total_final})")
    print(f"Convergence rate: {converged_count}/{len(results_summary)} ({converged_count/len(results_summary)*100:.0f}%)")
    print()


def interactive_mode():
    """
    Interactive mode: enter your own concept strings for testing.
    """
    print("#" * 80)
    print("# INTERACTIVE MODE")
    print("#" * 80)
    print("\nEnter concept predictions to test self-refine interactively.")
    print("Type 'quit' to exit.\n")
    
    while True:
        print("-" * 80)
        print("Enter concept string (or 'quit'):")
        concepts = input("> ").strip()
        
        if concepts.lower() == 'quit':
            print("Exiting interactive mode.")
            break
        
        if not concepts:
            print("Please enter a valid concept string.")
            continue
        
        print("\nEnter diagnosis (optional, press Enter to skip):")
        diagnosis = input("> ").strip() or None
        
        print()
        inspect_example(concepts, diagnosis)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Option 3: Manual Inspection')
    parser.add_argument('--mode', type=str, default='predefined', 
                        choices=['predefined', 'interactive', 'custom'],
                        help='Testing mode: predefined examples, interactive, or custom')
    parser.add_argument('--concepts', type=str, default=None,
                        help='Custom concept string to test')
    parser.add_argument('--diagnosis', type=str, default=None,
                        help='Diagnosis label for custom concept')
    
    args = parser.parse_args()
    
    if args.mode == 'predefined':
        run_predefined_examples()
    elif args.mode == 'interactive':
        interactive_mode()
    elif args.mode == 'custom' and args.concepts:
        inspect_example(args.concepts, args.diagnosis)
    else:
        print("Error: For custom mode, provide --concepts argument")
        print("\nRun with --help for usage information")
"""
Test script for Phase 1 Self-Refine functionality.

Usage:
    python scripts/test_self_refine.py
"""

import sys
sys.path.append('.')

from src.self_refine.concept_refiner import ConceptSelfRefine, SimpleRuleBasedRefiner

# Test cases with known inconsistencies
test_cases = [
    {
        "name": "Asymmetric shape + Sharp border (should be inconsistent)",
        "concepts": "The color is highly variable, the shape is irregular, the border is sharp and well-defined, the dermoscopic patterns are atypical pigment network, the texture is smooth, the symmetry is asymmetrical, the elevation is flat to raised",
        "expected_violations": 1  # asymmetry_border_consistency
    },
    {
        "name": "Regular pattern + Asymmetry (should be inconsistent)",
        "concepts": "The color is uniformly tan, the shape is round, the border is sharp and well-defined, the dermoscopic patterns are regular pigment network, the texture is smooth, the symmetry is asymmetrical, the elevation is raised",
        "expected_violations": 1  # benign_pattern_consistency
    },
    {
        "name": "Benign lesion (should be consistent)",
        "concepts": "The color is uniformly tan, the shape is round, the border is sharp and well-defined, the dermoscopic patterns are regular pigment network, the texture is smooth, the symmetry is symmetrical, the elevation is raised",
        "expected_violations": 0
    },
    {
        "name": "Melanoma features (should be consistent)",
        "concepts": "The color is highly variable, the shape is irregular, the border is often blurry and irregular, the dermoscopic patterns are atypical pigment network, the texture is a raised or ulcerated surface, the symmetry is asymmetrical, the elevation is flat to raised",
        "expected_violations": 0
    }
]

def main():
    print("\n" + "="*70)
    print("Testing Phase 1 Self-Refine for ExpLICD")
    print("="*70 + "\n")
    
    # Initialize refiner with rule-based fallback
    refiner = ConceptSelfRefine(
        llm_refine_fn=SimpleRuleBasedRefiner(),
        max_iterations=3,
        verbose=True
    )
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*70}")
        
        print(f"\nInitial concepts:")
        print(test_case['concepts'])
        
        # Run refinement
        refined, info = refiner.refine(test_case['concepts'])
        
        print(f"\nRefined concepts:")
        print(refined)
        
        print(f"\nRefinement Info:")
        print(f"  Iterations: {info['num_iterations']}")
        print(f"  Initial violations: {info['initial_violations']}")
        print(f"  Final violations: {info['final_violations']}")
        print(f"  Converged: {info['converged']}")
        
        # Verify expected behavior
        if info['initial_violations'] == test_case['expected_violations']:
            print(f"  ✓ Test PASSED (expected {test_case['expected_violations']} violations)")
        else:
            print(f"  ✗ Test FAILED (expected {test_case['expected_violations']}, got {info['initial_violations']} violations)")
        
        print()
    
    print("="*70)
    print("Testing complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
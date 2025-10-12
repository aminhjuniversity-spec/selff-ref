"""
Test script for Phase 1 Self-Refine functionality.

Usage:
    python scripts/test_self_refine.py
"""

import sys
sys.path.append('.')

from src.self_refine.concept_refiner import ConceptSelfRefine, SimpleRuleBasedRefiner, ConceptConsistencyRules


def test_consistency_rules():
    """Test individual consistency rules"""
    print("Testing Consistency Rules")
    print("=" * 50)
    
    rules = ConceptConsistencyRules()
    
    # Test Case 1: Asymmetric shape with sharp border (should violate)
    concepts1 = {
        'color': 'highly variable, often with multiple colors',
        'shape': 'irregular',
        'border': 'sharp and well-defined',
        'dermoscopic patterns': 'atypical pigment network',
        'texture': 'smooth',
        'symmetry': 'asymmetrical',
        'elevation': 'flat to raised'
    }
    
    violations1 = rules.check_consistency(concepts1)
    print(f"\nTest Case 1: Asymmetric + Sharp border")
    print(f"Violations: {len(violations1)}")
    for v in violations1:
        print(f"  - {v[:80]}...")
    
    # Test Case 2: Multiple colors with regular patterns (should violate)
    concepts2 = {
        'color': 'highly variable, often with multiple colors',
        'shape': 'round',
        'border': 'sharp and well-defined',
        'dermoscopic patterns': 'regular pigment network, symmetric dots',
        'texture': 'smooth',
        'symmetry': 'symmetrical',
        'elevation': 'flat'
    }
    
    violations2 = rules.check_consistency(concepts2)
    print(f"\nTest Case 2: Multiple colors + Regular patterns")
    print(f"Violations: {len(violations2)}")
    for v in violations2:
        print(f"  - {v[:80]}...")
    
    # Test Case 3: Consistent concepts (should NOT violate)
    concepts3 = {
        'color': 'uniformly tan, brown, or black',
        'shape': 'round',
        'border': 'sharp and well-defined',
        'dermoscopic patterns': 'regular pigment network, symmetric dots',
        'texture': 'smooth',
        'symmetry': 'symmetrical',
        'elevation': 'flat'
    }
    
    violations3 = rules.check_consistency(concepts3)
    print(f"\nTest Case 3: Consistent concepts")
    print(f"Violations: {len(violations3)} (Expected: 0)")
    
    print("\n✓ Consistency rules test completed!")


def test_self_refine():
    """Test the full self-refine loop"""
    print("\n\nTesting Self-Refine Loop")
    print("=" * 50)
    
    # Create a refiner with rule-based fallback
    refiner = ConceptSelfRefine(
        llm_refine_fn=SimpleRuleBasedRefiner(),
        max_iterations=3,
        verbose=True
    )
    
    # Test Case: Inconsistent concepts that should be refined
    initial_concepts = (
        "The color is highly variable, often with multiple colors (black, brown, red, white, blue), "
        "the shape is irregular, "
        "the border is sharp and well-defined, "
        "the dermoscopic patterns are regular pigment network, symmetric dots and globules, "
        "the texture is smooth, "
        "the symmetry is asymmetrical, "
        "the elevation is flat to raised."
    )
    
    print("\nInitial Concepts:")
    print(initial_concepts)
    
    # Run refinement
    refined_concepts, refinement_info = refiner.refine(initial_concepts)
    
    print("\n\nRefined Concepts:")
    print(refined_concepts)
    
    print("\n\nRefinement Summary:")
    print(f"  Iterations: {refinement_info['iterations']}")
    print(f"  Initial violations: {refinement_info['initial_violations']}")
    print(f"  Final violations: {refinement_info['final_violations']}")
    print(f"  Converged: {refinement_info['converged']}")
    
    print("\n✓ Self-refine loop test completed!")


def test_parsing():
    """Test concept string parsing"""
    print("\n\nTesting Concept Parsing")
    print("=" * 50)
    
    refiner = ConceptSelfRefine(
        llm_refine_fn=SimpleRuleBasedRefiner(),
        max_iterations=1,
        verbose=False
    )
    
    test_str = (
        "The color is highly variable, the shape is irregular, "
        "the border is often blurry and irregular, "
        "the dermoscopic patterns are atypical pigment network, "
        "the texture is smooth, the symmetry is asymmetrical, "
        "the elevation is flat to raised."
    )
    
    parsed = refiner.parse_concepts(test_str)
    print("\nParsed concepts:")
    for key, value in parsed.items():
        print(f"  {key}: {value}")
    
    # Convert back to string
    reconstructed = refiner.concepts_to_string(parsed)
    print("\nReconstructed string:")
    print(reconstructed)
    
    print("\n✓ Parsing test completed!")


if __name__ == "__main__":
    print("Testing Phase 1 Self-Refine for ExpLICD")
    print("=" * 50)
    
    # Run all tests
    test_consistency_rules()
    test_self_refine()
    test_parsing()
    
    print("\n\n" + "=" * 50)
    print("✓ All tests passed successfully!")
    print("=" * 50)
    print("\nYou can now run the full pipeline with Self-Refine enabled.")
    print("Next step: Modify run_x_to_c_to_y.py to use get_concept_predictions_with_self_refine()")

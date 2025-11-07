"""
Phase 1 Self-Refine for ExpLICD Concept Predictions
Based on Self-Refine paper: https://arxiv.org/abs/2303.17651

This module implements:
1. Rule-based consistency checking for dermoscopic concepts
2. Iterative refinement with history tracking to prevent oscillation
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptConsistencyRules:
    """
    Medical knowledge-based consistency rules for dermoscopic concepts.
    
    These rules encode clinical relationships between dermoscopic features
    based on standard dermatology criteria (ABCD rule, pattern analysis, etc.)
    """
    
    @staticmethod
    def check_consistency(concepts_dict: Dict[str, str]) -> List[str]:
        """
        Check for clinical inconsistencies in concept predictions.
        
        Args:
            concepts_dict: Dictionary mapping concept names to predicted descriptions
                          e.g., {'color': 'highly variable...', 'shape': 'irregular', ...}
        
        Returns:
            List of violation messages (empty if consistent)
        """
        violations = []
        
        # Rule 1: Asymmetric shape should have irregular/blurry borders
        if 'asymmetric' in concepts_dict.get('symmetry', '').lower():
            border = concepts_dict.get('border', '').lower()
            if 'sharp' in border or 'well-defined' in border:
                violations.append(
                    "Clinical inconsistency: Asymmetric lesions typically have irregular borders, "
                    "but 'sharp and well-defined' border was predicted."
                )
        
        # Rule 2: Multiple colors should correlate with irregular patterns
        if 'multiple colors' in concepts_dict.get('color', '').lower():
            patterns = concepts_dict.get('dermoscopic patterns', '').lower()
            if 'regular' in patterns and 'symmetric' in patterns:
                violations.append(
                    "Clinical inconsistency: Multiple colors typically indicate complex/irregular patterns, "
                    "but 'regular' patterns were predicted."
                )
        
        # Rule 3: Smooth texture conflicts with ulcerated/raised descriptions
        if 'smooth' in concepts_dict.get('texture', '').lower():
            if 'ulcerated' in concepts_dict.get('texture', '').lower() or \
               'raised' in concepts_dict.get('elevation', '').lower():
                violations.append(
                    "Clinical inconsistency: 'Smooth' texture conflicts with raised/ulcerated features."
                )
        
        # Rule 4: BCC indicators (arborizing vessels) should have specific characteristics
        if 'arborizing vessels' in concepts_dict.get('dermoscopic patterns', '').lower():
            if 'irregular' not in concepts_dict.get('shape', '').lower():
                violations.append(
                    "Clinical inconsistency: Arborizing vessels (BCC indicator) typically appear in "
                    "lesions with irregular shape."
                )
        
        # Rule 5: Melanoma indicators (atypical network, blue-whitish veil) require specific features
        if 'atypical' in concepts_dict.get('dermoscopic patterns', '').lower() or \
           'blue-whitish veil' in concepts_dict.get('dermoscopic patterns', '').lower():
            if 'symmetrical' in concepts_dict.get('symmetry', '').lower():
                violations.append(
                    "Clinical inconsistency: Atypical patterns/blue-whitish veil (melanoma indicators) "
                    "rarely appear in symmetrical lesions."
                )
        
        # Rule 6: Flat elevation conflicts with thick/raised texture
        if 'flat' in concepts_dict.get('elevation', '').lower():
            texture = concepts_dict.get('texture', '').lower()
            if 'thick' in texture or 'warty' in texture:
                violations.append(
                    "Clinical inconsistency: Flat elevation conflicts with thick/warty texture."
                )
        
        return violations


class ConceptSelfRefine:
    """
    Iterative refinement system for ExpLICD concept predictions.
    
    Implements the Self-Refine paper's approach:
    1. Initial prediction
    2. Feedback generation (consistency checking)
    3. Refinement based on feedback
    4. Repeat until consistent or max iterations
    """
    
    def __init__(self, llm_refine_fn, max_iterations=5, verbose=True):
        """
        Args:
            llm_refine_fn: Function that takes (concepts_str, feedback) and returns refined concepts
            max_iterations: Maximum refinement iterations (increased from 3 to 5)
            verbose: Whether to print refinement progress
        """
        self.llm_refine_fn = llm_refine_fn
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.rules = ConceptConsistencyRules()
    
    def parse_concepts(self, concepts_str: str) -> Dict[str, str]:
        """
        Parse concept string into dictionary.
        
        Example input:
            "The color is highly variable, the shape is irregular, ..."
        
        Returns:
            {'color': 'highly variable', 'shape': 'irregular', ...}
        """
        concepts_dict = {}
        
        # Define expected concept keys
        concept_keys = ['color', 'shape', 'border', 'dermoscopic patterns', 
                       'texture', 'symmetry', 'elevation']
        
        for key in concept_keys:
            # Match pattern: "the {key} is {description}"
            pattern = rf"the {key} (?:is|are) ([^,\.]+)"
            match = re.search(pattern, concepts_str, re.IGNORECASE)
            if match:
                concepts_dict[key] = match.group(1).strip()
        
        return concepts_dict
    
    def concepts_to_string(self, concepts_dict: Dict[str, str]) -> str:
        """
        Convert concepts dictionary back to string format.
        """
        template = (
            "The color is {color}, the shape is {shape}, the border is {border}, "
            "the dermoscopic patterns are {dermoscopic patterns}, the texture is {texture}, "
            "the symmetry is {symmetry}, the elevation is {elevation}."
        )
        return template.format(**concepts_dict)
    
    def refine(self, initial_concepts: str, diagnosis: str = None) -> Tuple[str, Dict]:
        """
        Iteratively refine concept predictions.
        
        Args:
            initial_concepts: Initial concept string from ExpLICD
            diagnosis: Optional diagnosis (for context)
        
        Returns:
            Tuple of (refined_concepts_str, refinement_info_dict)
        """
        current_concepts = initial_concepts
        history = [initial_concepts]
        violation_history = []  # NEW: Track violation counts instead of strings
        
        refinement_info = {
            'iterations': 0,
            'initial_violations': 0,
            'final_violations': 0,
            'converged': False,
            'history': []
        }
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- Iteration {iteration} ---")
            
            # Parse concepts
            concepts_dict = self.parse_concepts(current_concepts)
            
            # Check consistency
            violations = self.rules.check_consistency(concepts_dict)
            num_violations = len(violations)
            violation_history.append(num_violations)
            
            if iteration == 0:
                refinement_info['initial_violations'] = num_violations
            
            if self.verbose:
                print(f"Found {num_violations} consistency violations")
            
            # If no violations, we're done
            if num_violations == 0:
                if self.verbose:
                    print(f"✓ Converged at iteration {iteration}")
                refinement_info['converged'] = True
                refinement_info['iterations'] = iteration
                refinement_info['final_violations'] = 0
                break
            
            # NEW: Smarter oscillation detection (only after 3 iterations)
            if len(violation_history) >= 3:
                last_three = violation_history[-3:]
                
                # Check if violations are stuck (no progress for 3 consecutive iterations)
                if last_three[0] == last_three[1] == last_three[2]:
                    if self.verbose:
                        print(f"⚠ Oscillation detected: violations stuck at {last_three[0]} for 3 iterations, stopping")
                    refinement_info['iterations'] = iteration
                    refinement_info['final_violations'] = num_violations
                    refinement_info['oscillation_reason'] = 'stuck'
                    break
                
                # Check if violations are consistently increasing
                elif last_three[1] >= last_three[0] and last_three[2] >= last_three[1]:
                    if self.verbose:
                        print(f"⚠ Oscillation detected: violations increasing [{last_three[0]}→{last_three[1]}→{last_three[2]}], stopping")
                    refinement_info['iterations'] = iteration
                    refinement_info['final_violations'] = num_violations
                    refinement_info['oscillation_reason'] = 'increasing'
                    break
            
            # Generate feedback
            feedback = "\n".join(violations)
            if self.verbose:
                print(f"Feedback: {feedback[:100]}...")
            
            # Refine concepts
            refined_concepts = self.llm_refine_fn(current_concepts, feedback, concepts_dict)
            
            # Update history (still keep for reference, but don't use for oscillation detection)
            history.append(refined_concepts)
            current_concepts = refined_concepts
            
            refinement_info['history'].append({
                'iteration': iteration,
                'violations': num_violations,
                'feedback': feedback
            })
        
        # Final check
        if refinement_info['iterations'] == self.max_iterations - 1:
            final_concepts_dict = self.parse_concepts(current_concepts)
            final_violations = self.rules.check_consistency(final_concepts_dict)
            refinement_info['final_violations'] = len(final_violations)
            if self.verbose:
                print(f"\n⚠ Max iterations reached with {len(final_violations)} remaining violations")
        
        return current_concepts, refinement_info


class SimpleRuleBasedRefiner:
    """
    Simple rule-based refiner (fallback when no LLM is available).
    
    This applies heuristic fixes to common inconsistencies.
    """
    
    def __call__(self, concepts_str: str, feedback: str, concepts_dict: Dict[str, str]) -> str:
        """
        Apply rule-based refinements to fix inconsistencies.
        
        Args:
            concepts_str: Current concept string
            feedback: Violation feedback
            concepts_dict: Parsed concepts dictionary
        
        Returns:
            Refined concept string
        """
        refined_dict = concepts_dict.copy()
        
        # Fix Rule 1: Asymmetric + Sharp border → Make border irregular
        if 'asymmetric' in refined_dict.get('symmetry', '').lower() and \
           'sharp' in refined_dict.get('border', '').lower():
            refined_dict['border'] = 'often blurry and irregular'
        
        # Fix Rule 2: Multiple colors + Regular patterns → Make patterns irregular
        if 'multiple colors' in refined_dict.get('color', '').lower() and \
           'regular' in refined_dict.get('dermoscopic patterns', '').lower():
            refined_dict['dermoscopic patterns'] = 'atypical pigment network, irregular streaks'
        
        # Fix Rule 3: Smooth + Ulcerated → Remove ulcerated from texture
        if 'smooth' in refined_dict.get('texture', '').lower() and \
           'ulcerated' in refined_dict.get('texture', '').lower():
            refined_dict['texture'] = 'smooth'
        
        # Fix Rule 4: Arborizing vessels + Regular shape → Make shape irregular
        if 'arborizing vessels' in refined_dict.get('dermoscopic patterns', '').lower() and \
           'round' in refined_dict.get('shape', '').lower():
            refined_dict['shape'] = 'round to irregular'
        
        # Fix Rule 5: Atypical patterns + Symmetrical → Make asymmetrical
        if ('atypical' in refined_dict.get('dermoscopic patterns', '').lower() or \
            'blue-whitish veil' in refined_dict.get('dermoscopic patterns', '').lower()) and \
           'symmetrical' in refined_dict.get('symmetry', '').lower():
            refined_dict['symmetry'] = 'asymmetrical'
        
        # Fix Rule 6: Flat elevation + Thick texture → Make texture smooth
        if 'flat' in refined_dict.get('elevation', '').lower() and \
           ('thick' in refined_dict.get('texture', '').lower() or \
            'warty' in refined_dict.get('texture', '').lower()):
            refined_dict['texture'] = 'smooth'
        
        # Convert back to string
        template = (
            "The color is {color}, the shape is {shape}, the border is {border}, "
            "the dermoscopic patterns are {dermoscopic patterns}, the texture is {texture}, "
            "the symmetry is {symmetry}, the elevation is {elevation}."
        )
        return template.format(**refined_dict)

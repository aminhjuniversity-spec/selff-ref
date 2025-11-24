# Updated concept_refiner.py with new refine() method inserted and old version removed

import numpy as np
from typing import Dict, List, Tuple
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptConsistencyRules:
    @staticmethod
    def check_consistency(concepts_dict: Dict[str, str]) -> List[str]:
        violations = []

        if 'asymmetric' in concepts_dict.get('symmetry', '').lower():
            border = concepts_dict.get('border', '').lower()
            if 'sharp' in border or 'well-defined' in border:
                violations.append(
                    "Clinical inconsistency: Asymmetric lesions typically have irregular borders, "
                    "but 'sharp and well-defined' border was predicted."
                )

        if 'multiple colors' in concepts_dict.get('color', '').lower():
            patterns = concepts_dict.get('dermoscopic patterns', '').lower()
            if 'regular' in patterns and 'symmetric' in patterns:
                violations.append(
                    "Clinical inconsistency: Multiple colors typically indicate complex/irregular patterns, "
                    "but 'regular' patterns were predicted."
                )

        if 'smooth' in concepts_dict.get('texture', '').lower():
            if 'ulcerated' in concepts_dict.get('texture', '').lower() or \
               'raised' in concepts_dict.get('elevation', '').lower():
                violations.append(
                    "Clinical inconsistency: 'Smooth' texture conflicts with raised/ulcerated features."
                )

        if 'arborizing vessels' in concepts_dict.get('dermoscopic patterns', '').lower():
            if 'irregular' not in concepts_dict.get('shape', '').lower():
                violations.append(
                    "Clinical inconsistency: Arborizing vessels typically appear in lesions with irregular shape."
                )

        if 'atypical' in concepts_dict.get('dermoscopic patterns', '').lower() or \
           'blue-whitish veil' in concepts_dict.get('dermoscopic patterns', '').lower():
            if 'symmetrical' in concepts_dict.get('symmetry', '').lower():
                violations.append(
                    "Clinical inconsistency: Atypical patterns rarely appear in symmetrical lesions."
                )

        if 'flat' in concepts_dict.get('elevation', '').lower():
            texture = concepts_dict.get('texture', '').lower()
            if 'thick' in texture or 'warty' in texture:
                violations.append(
                    "Clinical inconsistency: Flat elevation conflicts with thick/warty texture."
                )

        return violations


class ConceptSelfRefine:
    def __init__(self, llm_refine_fn, max_iterations=5, verbose=True):
        self.llm_refine_fn = llm_refine_fn
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.rules = ConceptConsistencyRules()

    def parse_concepts(self, concepts_str: str) -> Dict[str, str]:
        concepts_dict = {}
        concept_keys = ['color', 'shape', 'border', 'dermoscopic patterns', 'texture', 'symmetry', 'elevation']

        for key in concept_keys:
            pattern = rf"the {key} (?:is|are) ([^,\.]+)"
            match = re.search(pattern, concepts_str, re.IGNORECASE)
            if match:
                concepts_dict[key] = match.group(1).strip()

        return concepts_dict

    def concepts_to_string(self, concepts_dict: Dict[str, str]) -> str:
        template = (
            "The color is {color}, the shape is {shape}, the border is {border}, "
            "the dermoscopic patterns are {dermoscopic patterns}, the texture is {texture}, "
            "the symmetry is {symmetry}, the elevation is {elevation}."
        )
        return template.format(**concepts_dict)

    # --- NEW refine() (old version removed completely) ---
    def refine(self, initial_concepts: str, diagnosis: str = None) -> Tuple[str, Dict]:
        """Iteratively refine concept predictions with improved oscillation detection."""
        
        current_concepts = initial_concepts
        history = [initial_concepts]
        violation_history = []
        
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
            
            concepts_dict = self.parse_concepts(current_concepts)
            violations = self.rules.check_consistency(concepts_dict)
            num_violations = len(violations)
            violation_history.append(num_violations)
            
            if iteration == 0:
                refinement_info['initial_violations'] = num_violations
            
            if self.verbose:
                print(f"Found {num_violations} consistency violations")
            
            # SUCCESS
            if num_violations == 0:
                if self.verbose:
                    print(f"✓ Converged at iteration {iteration}")
                refinement_info['converged'] = True
                refinement_info['iterations'] = iteration
                refinement_info['final_violations'] = 0
                break
            
            # IMPROVED OSCILLATION DETECTION
            if len(violation_history) >= 4:
                last_four = violation_history[-4:]
                
                # Stuck
                if last_four[0] == last_four[1] == last_four[2] == last_four[3]:
                    if last_four[0] >= violation_history[0]:
                        if self.verbose:
                            print(f"⚠ Truly stuck at {last_four[0]} violations (4 iterations), stopping")
                        refinement_info['oscillation_reason'] = 'stuck'
                        break
                
                # Worsening
                elif all(last_four[i+1] > last_four[i] for i in range(3)):
                    if self.verbose:
                        print(f"⚠ Violations consistently increasing, stopping")
                    refinement_info['oscillation_reason'] = 'worsening'
                    break
                
                # Alternating pattern
                elif len(set(last_four)) == 2 and last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                    if self.verbose:
                        print(f"⚠ Alternating pattern detected: {last_four}, stopping")
                    refinement_info['oscillation_reason'] = 'alternating'
                    break
            
            # Generate feedback
            feedback = "\n".join(violations)
            
            # Refine with LLM
            try:
                refined_concepts = self.llm_refine_fn(current_concepts, feedback, concepts_dict)
                
                if refined_concepts == current_concepts:
                    if self.verbose:
                        print(f"⚠ LLM returned unchanged concepts, stopping")
                    refinement_info['oscillation_reason'] = 'no_change'
                    break
                
                current_concepts = refined_concepts
                history.append(refined_concepts)
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Refinement failed: {e}, stopping")
                refinement_info['oscillation_reason'] = 'error'
                break
            
            refinement_info['history'].append({
                'iteration': iteration,
                'violations': num_violations,
                'feedback': feedback
            })
        
        refinement_info['iterations'] = iteration
        refinement_info['final_violations'] = num_violations
        
        return current_concepts, refinement_info


class SimpleRuleBasedRefiner:
    def __call__(self, concepts_str: str, feedback: str, concepts_dict: Dict[str, str]) -> str:
        refined_dict = concepts_dict.copy()

        if 'asymmetric' in refined_dict.get('symmetry', '').lower() and \
           'sharp' in refined_dict.get('border', '').lower():
            refined_dict['border'] = 'often blurry and irregular'

        if 'multiple colors' in refined_dict.get('color', '').lower() and \
           'regular' in refined_dict.get('dermoscopic patterns', '').lower():
            refined_dict['dermoscopic patterns'] = 'atypical pigment network, irregular streaks'

        if 'smooth' in refined_dict.get('texture', '').lower():
            texture = refined_dict.get('texture', '').lower()
            elevation = refined_dict.get('elevation', '').lower()

            if 'ulcerated' in texture:
                refined_dict['texture'] = 'smooth'

            if 'raised' in elevation or 'ulcerat' in elevation:
                refined_dict['elevation'] = 'flat to slightly raised'

        if 'arborizing vessels' in refined_dict.get('dermoscopic patterns', '').lower() and \
           'round' in refined_dict.get('shape', '').lower():
            refined_dict['shape'] = 'round to irregular'

        if ('atypical' in refined_dict.get('dermoscopic patterns', '').lower() or \
            'blue-whitish veil' in refined_dict.get('dermoscopic patterns', '').lower()) and \
           'symmetrical' in refined_dict.get('symmetry', '').lower():
            refined_dict['symmetry'] = 'asymmetrical'

        if 'flat' in refined_dict.get('elevation', '').lower() and \
           ('thick' in refined_dict.get('texture', '').lower() or \
            'warty' in refined_dict.get('texture', '').lower()):
            refined_dict['texture'] = 'smooth'

        template = (
            "The color is {color}, the shape is {shape}, the border is {border}, "
            "the dermoscopic patterns are {dermoscopic patterns}, the texture is {texture}, "
            "the symmetry is {symmetry}, the elevation is {elevation}."
        )
        return template.format(**refined_dict)

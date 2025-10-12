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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptConsistencyRules:
    """
    Medical knowledge-based consistency rules for dermoscopic concepts.
    
    These rules encode clinical relationships between dermoscopic features
    based on standard dermatology criteria (ABCD rule, pattern analysis, etc.)
    """
    
    # Dermoscopic concept indices in ExpLICD output
    CONCEPT_INDICES = {
        'color': 0,
        'shape': 1,
        'border': 2,
        'dermoscopic_patterns': 3,
        'texture': 4,
        'symmetry': 5,
        'elevation': 6
    }
    
    # Clinical consistency rules
    # Each rule has: condition, expected_state, feedback, severity
    RULES = {
        "asymmetry_border_consistency": {
            "description": "Asymmetric lesions typically have irregular borders",
            "check": lambda concepts: not (
                "irregular" in concepts['shape'].lower() and 
                "sharp and well-defined" in concepts['border'].lower()
            ),
            "feedback": "Clinical inconsistency: Irregular/asymmetric shape usually correlates with irregular borders, not sharp and well-defined borders.",
            "fix_suggestion": "Consider if the border should be 'often blurry and irregular' instead of 'sharp and well-defined'.",
            "severity": "high"
        },
        
        "melanoma_pattern_consistency": {
            "description": "Atypical melanoma patterns should align with other malignant features",
            "check": lambda concepts: not (
                ("atypical" in concepts['dermoscopic_patterns'].lower() or 
                 "blue-whitish veil" in concepts['dermoscopic_patterns'].lower()) and
                "symmetrical" in concepts['symmetry'].lower() and
                "smooth" in concepts['texture'].lower()
            ),
            "feedback": "Clinical inconsistency: Atypical patterns (e.g., atypical network, blue-whitish veil) with symmetry and smooth texture is unusual for melanoma.",
            "fix_suggestion": "If atypical dermoscopic patterns present, expect asymmetry and/or raised/ulcerated texture.",
            "severity": "high"
        },
        
        "benign_pattern_consistency": {
            "description": "Regular benign patterns should align with symmetric features",
            "check": lambda concepts: not (
                "regular pigment network" in concepts['dermoscopic_patterns'].lower() and
                "asymmetrical" in concepts['symmetry'].lower()
            ),
            "feedback": "Clinical inconsistency: Regular pigment network typically indicates benign lesions, which are usually symmetrical.",
            "fix_suggestion": "Consider if symmetry should be 'symmetrical' rather than 'asymmetrical'.",
            "severity": "medium"
        },
        
        "bcc_pattern_consistency": {
            "description": "BCC-specific patterns should align with other BCC features",
            "check": lambda concepts: not (
                "arborizing vessels" in concepts['dermoscopic_patterns'].lower() and
                "uniformly tan, brown, or black" in concepts['color'].lower()
            ),
            "feedback": "Clinical inconsistency: Arborizing vessels (BCC feature) rarely presents with uniform tan/brown color.",
            "fix_suggestion": "BCC typically shows translucent, pearly white color, sometimes with blue, brown, or black areas.",
            "severity": "medium"
        },
        
        "raised_ulcerated_consistency": {
            "description": "Raised/ulcerated texture aligns with elevation",
            "check": lambda concepts: not (
                "raised or ulcerated" in concepts['texture'].lower() and
                "flat" in concepts['elevation'].lower() and
                "to raised" not in concepts['elevation'].lower()
            ),
            "feedback": "Clinical inconsistency: Texture described as 'raised or ulcerated' but elevation is 'flat'.",
            "fix_suggestion": "Consider if elevation should be 'raised' or 'flat to raised'.",
            "severity": "high"
        },
        
        "color_pattern_consistency": {
            "description": "Multiple colors should correlate with atypical patterns",
            "check": lambda concepts: not (
                "multiple colors" in concepts['color'].lower() and
                "regular pigment network" in concepts['dermoscopic_patterns'].lower() and
                "symmetric" in concepts['dermoscopic_patterns'].lower()
            ),
            "feedback": "Clinical inconsistency: Highly variable multiple colors rarely present with regular, symmetric patterns.",
            "fix_suggestion": "Multiple colors often indicate atypical patterns (irregular streaks, blue-whitish veil).",
            "severity": "medium"
        }
    }


class ConceptSelfRefine:
    """
    Phase 1 Self-Refine implementation for ExpLICD concept predictions.
    
    This class implements iterative refinement with:
    1. Rule-based consistency checking
    2. History tracking to prevent oscillation
    3. LLM-based refinement using feedback
    """
    
    def __init__(self, llm_refine_fn=None, max_iterations=3, verbose=True):
        """
        Args:
            llm_refine_fn: Function that takes (concepts_dict, feedback) and returns refined concepts
            max_iterations: Maximum number of refinement iterations
            verbose: Whether to print refinement progress
        """
        self.rules = ConceptConsistencyRules()
        self.llm_refine_fn = llm_refine_fn
        self.max_iterations = max_iterations
        self.verbose = verbose
    
    def parse_concept_string(self, concept_string: str) -> Dict[str, str]:
        """
        Parse ExpLICD concept string into structured dictionary.
        
        Input format: "The color is X, the shape is Y, the border is Z, ..."
        Output: {'color': 'X', 'shape': 'Y', 'border': 'Z', ...}
        """
        concepts = {}
        
        # Remove "Thus the diagnosis is..." suffix if present
        if "Thus the diagnosis is" in concept_string:
            concept_string = concept_string[:concept_string.find("Thus the diagnosis is")].strip()
        
        # Parse each concept
        parts = concept_string.split(", the ")
        
        for i, part in enumerate(parts):
            if i == 0:
                # First part: "The color is X"
                part = part.replace("The ", "")
            
            if " is " in part:
                key, value = part.split(" is ", 1)
                key = key.strip()
                value = value.strip().rstrip(',').rstrip('.')
                concepts[key] = value
        
        return concepts
    
    def format_concepts_to_string(self, concepts: Dict[str, str], add_diagnosis: str = None) -> str:
        """
        Convert structured concepts dictionary back to ExpLICD string format.
        
        Input: {'color': 'X', 'shape': 'Y', ...}
        Output: "The color is X, the shape is Y, ..."
        """
        template = "The color is {color}, the shape is {shape}, the border is {border}, the dermoscopic patterns are {dermoscopic_patterns}, the texture is {texture}, the symmetry is {symmetry}, the elevation is {elevation}"
        
        # Handle key name variations
        if 'dermoscopic patterns' in concepts:
            concepts['dermoscopic_patterns'] = concepts.pop('dermoscopic patterns')
        
        result = template.format(**concepts)
        
        if add_diagnosis:
            result += f". Thus the diagnosis is {add_diagnosis}."
        
        return result
    
    def check_consistency(self, concepts: Dict[str, str]) -> List[Dict]:
        """
        Check all consistency rules and return violations.
        
        Returns:
            List of violated rules with feedback
        """
        violations = []
        
        for rule_name, rule_config in self.rules.RULES.items():
            try:
                if not rule_config["check"](concepts):
                    violations.append({
                        "rule": rule_name,
                        "description": rule_config["description"],
                        "feedback": rule_config["feedback"],
                        "fix_suggestion": rule_config["fix_suggestion"],
                        "severity": rule_config["severity"]
                    })
            except KeyError as e:
                logger.warning(f"Rule {rule_name} failed due to missing concept: {e}")
                continue
        
        return violations
    
    def generate_refinement_feedback(self, violations: List[Dict]) -> str:
        """
        Generate natural language feedback from rule violations.
        
        This feedback will be passed to the LLM for refinement.
        """
        if not violations:
            return "No consistency issues detected. The concepts are clinically coherent."
        
        feedback_parts = ["The following clinical inconsistencies were detected:\n"]
        
        # Sort by severity
        high_severity = [v for v in violations if v['severity'] == 'high']
        medium_severity = [v for v in violations if v['severity'] == 'medium']
        
        for i, violation in enumerate(high_severity + medium_severity, 1):
            feedback_parts.append(f"{i}. {violation['feedback']}")
            feedback_parts.append(f"   Suggestion: {violation['fix_suggestion']}\n")
        
        return "\n".join(feedback_parts)
    
    def detect_oscillation(self, history: List[Dict[str, str]]) -> bool:
        """
        Detect if concepts are oscillating between iterations.
        
        Oscillation = concepts flip-flop between the same values repeatedly
        """
        if len(history) < 3:
            return False
        
        # Check if the last iteration is similar to iteration -3
        last_concepts = history[-1]
        third_last_concepts = history[-3]
        
        # Count how many concepts reverted to previous state
        reversions = sum(
            1 for key in last_concepts
            if key in third_last_concepts and 
            last_concepts[key] == third_last_concepts[key]
        )
        
        # If >50% of concepts reverted, we're oscillating
        return reversions > len(last_concepts) / 2
    
    def select_best_from_history(self, history: List[Dict], 
                                  violation_counts: List[int]) -> Dict[str, str]:
        """
        Select the best concept set from history based on fewest violations.
        """
        if not history:
            return None
        
        # Find iteration with minimum violations
        best_idx = np.argmin(violation_counts)
        
        if self.verbose:
            logger.info(f"Selected concepts from iteration {best_idx} with {violation_counts[best_idx]} violations")
        
        return history[best_idx]
    
    def refine(self, initial_concepts: str, diagnosis: str = None) -> Tuple[str, Dict]:
        """
        Main refinement loop.
        
        Args:
            initial_concepts: ExpLICD concept string
            diagnosis: Optional diagnosis to append
        
        Returns:
            Tuple of (refined_concept_string, refinement_info)
        """
        # Parse initial concepts
        current_concepts = self.parse_concept_string(initial_concepts)
        
        history = [current_concepts.copy()]
        violation_counts = []
        all_feedback = []
        
        if self.verbose:
            logger.info("="*60)
            logger.info("Starting Phase 1 Self-Refine")
            logger.info("="*60)
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                logger.info(f"\n--- Iteration {iteration} ---")
            
            # Check consistency
            violations = self.check_consistency(current_concepts)
            violation_counts.append(len(violations))
            
            if self.verbose:
                logger.info(f"Found {len(violations)} consistency violations")
            
            # Generate feedback
            feedback = self.generate_refinement_feedback(violations)
            all_feedback.append(feedback)
            
            # Check stopping condition
            if not violations:
                if self.verbose:
                    logger.info(f"✓ Converged at iteration {iteration}: No violations")
                break
            
            if self.verbose:
                logger.info(f"Feedback:\n{feedback}")
            
            # Check for oscillation
            if self.detect_oscillation(history):
                if self.verbose:
                    logger.info(f"⚠ Oscillation detected at iteration {iteration}")
                current_concepts = self.select_best_from_history(history, violation_counts)
                break
            
            # Refine with LLM
            if self.llm_refine_fn is not None:
                refined_concepts = self.llm_refine_fn(current_concepts, feedback)
                current_concepts = refined_concepts
                history.append(current_concepts.copy())
            else:
                # No LLM refiner provided, can't continue
                if self.verbose:
                    logger.warning("No LLM refiner provided, stopping at iteration {iteration}")
                break
        
        # Convert back to string format
        refined_string = self.format_concepts_to_string(current_concepts, add_diagnosis=diagnosis)
        
        # Prepare refinement info
        refinement_info = {
            "num_iterations": len(history),
            "initial_violations": violation_counts[0] if violation_counts else 0,
            "final_violations": violation_counts[-1] if violation_counts else 0,
            "history": history,
            "feedbacks": all_feedback,
            "converged": len(violations) == 0
        }
        
        if self.verbose:
            logger.info("="*60)
            logger.info(f"Refinement complete: {refinement_info['initial_violations']} → {refinement_info['final_violations']} violations")
            logger.info("="*60)
        
        return refined_string, refinement_info


# ============================================================================
# Simple LLM Refiner (Rule-based fallback for testing without LLM)
# ============================================================================

class SimpleRuleBasedRefiner:
    """
    Simple rule-based refiner for testing without an LLM.
    
    This applies heuristic fixes based on the feedback.
    For production, replace with actual LLM-based refinement.
    """
    
    def __init__(self):
        self.rules = ConceptConsistencyRules()
    
    def __call__(self, concepts: Dict[str, str], feedback: str) -> Dict[str, str]:
        """
        Apply simple heuristic fixes to concepts.
        
        This is a FALLBACK for testing. In production, use LLM refinement.
        """
        refined = concepts.copy()
        
        # Rule 1: Asymmetry + Sharp border → Irregular border
        if "irregular" in concepts['shape'].lower() and "sharp and well-defined" in concepts['border'].lower():
            refined['border'] = "often blurry and irregular"
        
        # Rule 2: Atypical patterns + Symmetry → Asymmetry
        if ("atypical" in concepts['dermoscopic_patterns'].lower() or 
            "blue-whitish veil" in concepts['dermoscopic_patterns'].lower()):
            if "symmetrical" in concepts['symmetry'].lower():
                refined['symmetry'] = "asymmetrical"
        
        # Rule 3: Regular network + Asymmetry → Symmetry
        if "regular pigment network" in concepts['dermoscopic_patterns'].lower():
            if "asymmetrical" in concepts['symmetry'].lower():
                refined['symmetry'] = "symmetrical"
        
        # Rule 4: Arborizing vessels + Tan/brown → Pearly white
        if "arborizing vessels" in concepts['dermoscopic_patterns'].lower():
            if "uniformly tan" in concepts['color'].lower():
                refined['color'] = "translucent, pearly white, sometimes with blue, brown, or black areas"
        
        # Rule 5: Raised texture + Flat elevation → Raised elevation
        if "raised or ulcerated" in concepts['texture'].lower():
"""
MMed-LLM Based Refiner for ExpLICD Self-Refine (Path D)
Uses MMed-Llama-3-8B instead of GPT-4o (FREE - runs on cluster)
"""

import sys
sys.path.append('.')

from src.models.MMed_Llama_3_8B import MMedLlama3
from typing import Dict


class MMedBasedRefiner:
    """
    MMed-LLM based refiner that uses MMed-Llama-3-8B to refine concept predictions.
    
    This is the FREE alternative to GPT-4o that runs on your cluster.
    """
    
    def __init__(self, ckpt="Henrychur/MMed-Llama-3-8B"):
        """
        Args:
            ckpt: MMed checkpoint to use
        """
        print(f"Loading MMed-LLM refiner: {ckpt}")
        self.model = MMedLlama3(ckpt=ckpt)
        print("✓ MMed-LLM refiner loaded")
    
    def __call__(self, concepts_str: str, feedback: str, concepts_dict: Dict[str, str]) -> str:
        """
        Refine concepts using MMed-LLM based on feedback.
        
        Args:
            concepts_str: Current concept description string
            feedback: Consistency violation feedback
            concepts_dict: Parsed concepts dictionary
        
        Returns:
            Refined concept description string
        """
        
        # Create the prompt for MMed-LLM
        instruction = """You are a dermatology expert. Your task is to refine dermoscopic concept descriptions to fix clinical inconsistencies.

CRITICAL RULES:
1. Only modify concepts that have consistency violations
2. Keep the same template format: "The color is ..., the shape is ..., etc."
3. Ensure concepts are medically accurate for melanoma vs nevus differentiation
4. Output ONLY the refined description, nothing else

DERMOSCOPIC CONCEPT VALUES (use these exact phrases):

Color options:
- "highly variable, often with multiple colors (black, brown, red, white, blue)"
- "uniformly tan, brown, or black"

Shape options:
- "irregular"
- "round"
- "round to irregular"

Border options:
- "often blurry and irregular"
- "sharp and well-defined"

Patterns options:
- "atypical pigment network, irregular streaks, blue-whitish veil, irregular"
- "regular pigment network, symmetric dots and globules"

Texture options:
- "a raised or ulcerated surface"
- "smooth"

Symmetry options:
- "asymmetrical"
- "symmetrical"

Elevation options:
- "flat to raised"
- "raised with possible central ulceration"
- "slightly raised"

MEDICAL CONSISTENCY RULES:
1. Asymmetric lesions → typically have irregular/blurry borders (not sharp)
2. Multiple colors → typically indicate irregular patterns (not regular)
3. Smooth texture → conflicts with raised/ulcerated features
4. Atypical patterns / blue-whitish veil → rarely symmetrical
"""

        query = f"""Current dermoscopic description:
{concepts_str}

Consistency feedback (violations to fix):
{feedback}

Provide the REFINED description that fixes these violations. Output ONLY the refined description in the exact same format, nothing else."""

        try:
            # Get prompt from MMed
            prompt = self.model.get_prompt(instruction=instruction, query=query, demos=None)
            
            # Generate refinement (max 300 tokens)
            refined_concepts = self.model.predict(prompt=prompt, max_new_tokens=300).strip()
            
            # Clean up output (remove any extra text)
            refined_concepts = self._clean_output(refined_concepts)
            
            # Validate format
            if not self._validate_format(refined_concepts):
                print(f"⚠ MMed output invalid format, falling back to original")
                return concepts_str
            
            return refined_concepts
            
        except Exception as e:
            print(f"⚠ MMed refinement failed: {e}, using original concepts")
            return concepts_str
    
    def _clean_output(self, output: str) -> str:
        """Clean MMed output to extract just the concept description"""
        # MMed sometimes adds extra text, extract just the concept sentence
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for line that starts with "The color is"
            if line.lower().startswith('the color is'):
                return line
        
        # If not found, return first non-empty line
        for line in lines:
            if line.strip():
                return line.strip()
        
        return output.strip()
    
    # NEW (LENIENT):
    def _validate_format(self, refined_str: str) -> bool:
    # Just check if 5 out of 7 concept keywords are present
        concept_keywords = ['color', 'shape', 'border', 'pattern', 'texture', 'symmetry', 'elevation']
        found_count = sum(1 for keyword in concept_keywords if keyword in refined_str.lower())
        return found_count >= 5  # ✅ 5/7 is good enough


# Example usage for testing
if __name__ == "__main__":
    # Test the MMed refiner
    print("Testing MMed-LLM Refiner")
    print("=" * 80)
    
    refiner = MMedBasedRefiner(ckpt="Henrychur/MMed-Llama-3-8B")
    
    test_concepts = (
        "The color is highly variable, often with multiple colors (black, brown, red, white, blue), "
        "the shape is irregular, "
        "the border is sharp and well-defined, "
        "the dermoscopic patterns are regular pigment network, symmetric dots and globules, "
        "the texture is smooth, "
        "the symmetry is asymmetrical, "
        "the elevation is flat to raised."
    )
    
    test_feedback = """Clinical inconsistency: Asymmetric lesions typically have irregular borders, but 'sharp and well-defined' border was predicted.
Clinical inconsistency: Multiple colors typically indicate complex/irregular patterns, but 'regular' patterns were predicted."""
    
    test_dict = {
        'color': 'highly variable, often with multiple colors (black, brown, red, white, blue)',
        'shape': 'irregular',
        'border': 'sharp and well-defined',
        'dermoscopic patterns': 'regular pigment network, symmetric dots and globules',
        'texture': 'smooth',
        'symmetry': 'asymmetrical',
        'elevation': 'flat to raised'
    }
    
    print("\nOriginal:")
    print(test_concepts)
    print("\nFeedback:")
    print(test_feedback)
    print("\nRefined:")
    refined = refiner(test_concepts, test_feedback, test_dict)
    print(refined)
    print("\n✓ Test complete!")

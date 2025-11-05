"""
LLM-Based Refiner for ExpLICD Self-Refine (Path D)
Uses GPT-4o to generate refinements based on consistency feedback.
"""

from openai import OpenAI
import os
from typing import Dict


class LLMBasedRefiner:
    """
    LLM-based refiner that uses GPT-4o to refine concept predictions.
    
    This is the ADVANCED version for Path D that we'll compare against
    the rule-based fallback to show LLM contributions.
    """
    
    def __init__(self, model="gpt-4o-mini", temperature=0.0):
        """
        Args:
            model: OpenAI model to use (gpt-4o-mini is cheaper, gpt-4o for best quality)
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
    
    def __call__(self, concepts_str: str, feedback: str, concepts_dict: Dict[str, str]) -> str:
        """
        Refine concepts using GPT-4o based on feedback.
        
        Args:
            concepts_str: Current concept description string
            feedback: Consistency violation feedback
            concepts_dict: Parsed concepts dictionary
        
        Returns:
            Refined concept description string
        """
        
        # Create the prompt for GPT-4o
        system_prompt = """You are a dermatology expert assisting with refining dermoscopic concept descriptions.

Your task: Given a description of dermoscopic concepts and feedback about clinical inconsistencies, refine the description to fix the inconsistencies while maintaining medical accuracy.

CRITICAL RULES:
1. Only modify concepts that have consistency violations
2. Keep the same template format: "The color is ..., the shape is ..., etc."
3. Ensure concepts are medically accurate for melanoma vs nevus differentiation
4. Use only the concept values provided in the knowledge base below

DERMOSCOPIC CONCEPT KNOWLEDGE BASE:
- Color options: 
  * "highly variable, often with multiple colors (black, brown, red, white, blue)"
  * "uniformly tan, brown, or black"
  * "translucent, pearly white, sometimes with blue, brown, or black areas"
  * "red, pink, or brown, often with a scale"
  * "light brown to black"
  * "pink brown or red"
  * "red, purple, or blue"

- Shape options:
  * "irregular"
  * "round"
  * "round to irregular"
  * "variable"

- Border options:
  * "often blurry and irregular"
  * "sharp and well-defined"
  * "rolled edges, often indistinct"

- Dermoscopic patterns options:
  * "atypical pigment network, irregular streaks, blue-whitish veil, irregular"
  * "regular pigment network, symmetric dots and globules"
  * "arborizing vessels, leaf-like areas, blue-gray avoid nests"
  * "strawberry pattern, glomerular vessels, scale"
  * "cerebriform pattern, milia-like cysts, comedo-like openings"
  * "central white patch, peripheral pigment network"
  * "depends on type (e.g., cherry angiomas have red lacunae; spider angiomas have a central red dot with radiating legs"

- Texture options:
  * "a raised or ulcerated surface"
  * "smooth"
  * "smooth, possibly with telangiectasias"
  * "rough, scaly"
  * "warty or greasy surface"
  * "firm, may dimple when pinched"

- Symmetry options:
  * "asymmetrical"
  * "symmetrical"
  * "can be symmetrical or asymmetrical depending on type"

- Elevation options:
  * "flat to raised"
  * "raised with possible central ulceration"
  * "slightly raised"
  * "slightly raised maybe thick"

MEDICAL CONSISTENCY RULES:
1. Asymmetric lesions → typically have irregular/blurry borders (not sharp)
2. Multiple colors → typically indicate irregular patterns (not regular)
3. Smooth texture → conflicts with raised/ulcerated features
4. Arborizing vessels → typically irregular shape
5. Atypical patterns / blue-whitish veil → rarely symmetrical
6. Flat elevation → conflicts with thick/warty texture"""

        user_prompt = f"""Current dermoscopic description:
{concepts_str}

Consistency feedback (violations to fix):
{feedback}

Please provide the REFINED description that fixes these violations. Output ONLY the refined description in the same format, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=300
            )
            
            refined_concepts = response.choices[0].message.content.strip()
            
            # Validate that the output follows the expected format
            if not self._validate_format(refined_concepts):
                print(f"⚠ LLM output invalid format, falling back to original")
                return concepts_str
            
            return refined_concepts
            
        except Exception as e:
            print(f"⚠ LLM refinement failed: {e}, using original concepts")
            return concepts_str
    
    def _validate_format(self, refined_str: str) -> bool:
        """Check if refined concepts follow expected format"""
        required_keys = ['color', 'shape', 'border', 'dermoscopic patterns', 
                        'texture', 'symmetry', 'elevation']
        
        for key in required_keys:
            if f"the {key}" not in refined_str.lower():
                return False
        
        return True


# Example usage for testing
if __name__ == "__main__":
    # Test the LLM refiner
    refiner = LLMBasedRefiner(model="gpt-4o-mini")
    
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
    
    print("Testing LLM Refiner")
    print("=" * 80)
    print("\nOriginal:")
    print(test_concepts)
    print("\nFeedback:")
    print(test_feedback)
    print("\nRefined:")
    refined = refiner(test_concepts, test_feedback, test_dict)
    print(refined)

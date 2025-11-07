"""
MMed-LLM Based Refiner for ExpLICD Self-Refine (Path D)
Uses MMed-Llama-3-8B instead of GPT-4o (FREE - runs on cluster)

ENHANCED VERSION with:
- Relaxed format validation
- Better output cleaning
- Concept extraction/salvaging
- Targeted refinement (only fix violated concepts)
"""

import sys
sys.path.append('.')

from src.models.MMed_Llama_3_8B import MMedLlama3
from typing import Dict
import re


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
        
        # NEW: Extract violated concepts from feedback for targeted refinement
        violated_concepts = self._extract_violated_concepts(feedback)
        
        # Create the prompt for MMed-LLM with targeted refinement
        instruction = self._build_instruction(violated_concepts)
        query = self._build_query(concepts_str, feedback, violated_concepts)
        
        try:
            # Get prompt from MMed
            prompt = self.model.get_prompt(instruction=instruction, query=query, demos=None)
            
            # Generate refinement (max 300 tokens)
            refined_concepts = self.model.predict(prompt=prompt, max_new_tokens=300).strip()
            
            # Clean up output (remove any extra text)
            refined_concepts = self._clean_output(refined_concepts)
            
            # NEW: Try to extract concepts if format is partially valid
            if not self._validate_format(refined_concepts):
                print(f"⚠ MMed output invalid, trying to extract concepts...")
                extracted = self._try_extract_concepts(refined_concepts, concepts_dict)
                if extracted and self._validate_format(extracted):
                    print(f"✓ Successfully extracted concepts")
                    return extracted
                else:
                    print(f"⚠ Extraction failed, falling back to original")
                    return concepts_str
            
            return refined_concepts
            
        except Exception as e:
            print(f"⚠ MMed refinement failed: {e}, using original concepts")
            return concepts_str
    
    def _extract_violated_concepts(self, feedback: str) -> set:
        """
        Extract which concepts have violations from feedback.
        
        Args:
            feedback: Violation feedback string
        
        Returns:
            Set of violated concept names (e.g., {'border', 'symmetry'})
        """
        violated = set()
        feedback_lower = feedback.lower()
        
        # Map feedback keywords to concept names
        concept_keywords = {
            'border': ['border'],
            'color': ['color', 'multiple colors'],
            'shape': ['shape'],
            'symmetry': ['symmetry', 'asymmetric', 'symmetrical'],
            'texture': ['texture', 'smooth', 'ulcerated'],
            'elevation': ['elevation', 'flat', 'raised'],
            'dermoscopic patterns': ['pattern', 'network', 'veil', 'streak']
        }
        
        for concept, keywords in concept_keywords.items():
            if any(kw in feedback_lower for kw in keywords):
                violated.add(concept)
        
        return violated
    
    def _build_instruction(self, violated_concepts: set) -> str:
        """Build targeted instruction based on violated concepts."""
        
        base_instruction = """You are a dermatology expert. Your task is to refine dermoscopic concept descriptions to fix clinical inconsistencies.

CRITICAL RULES:
1. ONLY modify the concepts mentioned in the feedback
2. Keep ALL other concepts exactly as they are
3. Maintain the exact template format: "The color is ..., the shape is ..., the border is ..., the dermoscopic patterns are ..., the texture is ..., the symmetry is ..., the elevation is ..."
4. Output ONLY the complete refined description, nothing else (no explanations, no preambles)

"""
        
        # Add concept-specific guidance for violated concepts
        if violated_concepts:
            base_instruction += "\nFOCUS ON FIXING THESE CONCEPTS:\n"
            
            if 'border' in violated_concepts:
                base_instruction += "- Border: Use 'often blurry and irregular' for asymmetric lesions\n"
            
            if 'color' in violated_concepts or 'dermoscopic patterns' in violated_concepts:
                base_instruction += "- Patterns: Use 'atypical pigment network, irregular streaks' for multiple colors\n"
            
            if 'symmetry' in violated_concepts:
                base_instruction += "- Symmetry: Use 'asymmetrical' for atypical patterns/blue-whitish veil\n"
            
            if 'texture' in violated_concepts or 'elevation' in violated_concepts:
                base_instruction += "- Texture/Elevation: 'smooth' conflicts with 'raised' or 'ulcerated'\n"
        
        base_instruction += """
VALID CONCEPT VALUES (use these exact phrases):

Color: "highly variable, often with multiple colors (black, brown, red, white, blue)" OR "uniformly tan, brown, or black"
Shape: "irregular" OR "round" OR "round to irregular"
Border: "often blurry and irregular" OR "sharp and well-defined"
Patterns: "atypical pigment network, irregular streaks, blue-whitish veil, irregular" OR "regular pigment network, symmetric dots and globules"
Texture: "a raised or ulcerated surface" OR "smooth"
Symmetry: "asymmetrical" OR "symmetrical"
Elevation: "flat to raised" OR "raised with possible central ulceration" OR "slightly raised"
"""
        
        return base_instruction
    
    def _build_query(self, concepts_str: str, feedback: str, violated_concepts: set) -> str:
        """Build targeted query emphasizing which concepts to fix."""
        
        query = f"""Current description:
{concepts_str}

Violations to fix:
{feedback}

IMPORTANT: Only modify the concepts mentioned in the violations above ({', '.join(violated_concepts) if violated_concepts else 'as indicated'}). Keep all other concepts EXACTLY as they are.

Provide the complete refined description in this exact format:
The color is ..., the shape is ..., the border is ..., the dermoscopic patterns are ..., the texture is ..., the symmetry is ..., the elevation is ...

OUTPUT ONLY THE REFINED DESCRIPTION (no explanations):"""
        
        return query
    
    def _clean_output(self, output: str) -> str:
        """
        Clean MMed output to extract just the concept description.
        
        Removes common LLM artifacts like:
        - "Refined description:"
        - "Here is the refined..."
        - Multiple newlines
        - Extra explanations
        """
        # Remove common preambles
        output = re.sub(r'^(refined description|here is|the refined)[\s:]*', '', output, flags=re.IGNORECASE)
        
        # Split by newlines and find the concept sentence
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for line that starts with "The color is"
            if line.lower().startswith('the color is'):
                return line
        
        # If not found, look for any line with multiple concept keywords
        for line in lines:
            line = line.strip()
            if len(line) > 50:  # Reasonable length for full description
                keyword_count = sum(1 for kw in ['color', 'shape', 'border', 'pattern', 'texture', 'symmetry', 'elevation'] 
                                   if kw in line.lower())
                if keyword_count >= 4:
                    return line
        
        # Last resort: return first substantial line
        for line in lines:
            if len(line.strip()) > 50:
                return line.strip()
        
        return output.strip()
    
    def _validate_format(self, refined_str: str) -> bool:
        """
        Lenient format validation - checks if 5 out of 7 concept keywords are present.
        
        Args:
            refined_str: Refined concept string to validate
        
        Returns:
            True if at least 5/7 concept keywords found
        """
        concept_keywords = ['color', 'shape', 'border', 'pattern', 'texture', 'symmetry', 'elevation']
        found_count = sum(1 for keyword in concept_keywords if keyword in refined_str.lower())
        return found_count >= 5  # ✅ 5/7 is good enough
    
    def _try_extract_concepts(self, output: str, original_dict: Dict[str, str]) -> str:
        """
        NEW: Try to salvage partially valid output by extracting concepts.
        
        If MMed output is partially valid (e.g., some concepts present but not all),
        try to extract what we can and fill in missing concepts from original.
        
        Args:
            output: MMed output that failed validation
            original_dict: Original concepts dictionary as fallback
        
        Returns:
            Extracted/salvaged concept string, or empty string if failed
        """
        extracted_dict = {}
        
        # Define expected concept keys
        concept_keys = ['color', 'shape', 'border', 'dermoscopic patterns', 
                       'texture', 'symmetry', 'elevation']
        
        # Try to extract each concept
        for key in concept_keys:
            # Try multiple patterns
            patterns = [
                rf"(?:the\s+)?{re.escape(key)}\s+(?:is|are)\s+([^,\.]+)",  # "the color is ..."
                rf"{re.escape(key)}:\s*([^,\.]+)",  # "color: ..."
            ]
            
            found = False
            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    extracted_dict[key] = match.group(1).strip()
                    found = True
                    break
            
            # If not found, use original
            if not found and key in original_dict:
                extracted_dict[key] = original_dict[key]
        
        # Check if we extracted enough concepts
        if len(extracted_dict) >= 5:
            # Convert to string
            template = (
                "The color is {color}, the shape is {shape}, the border is {border}, "
                "the dermoscopic patterns are {dermoscopic patterns}, the texture is {texture}, "
                "the symmetry is {symmetry}, the elevation is {elevation}."
            )
            try:
                return template.format(**extracted_dict)
            except KeyError:
                return ""
        
        return ""


# Example usage for testing
if __name__ == "__main__":
    # Test the MMed refiner
    print("Testing MMed-LLM Refiner (Enhanced Version)")
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
    
    # Test extraction function
    print("\n--- Testing Violated Concepts Extraction ---")
    violated = refiner._extract_violated_concepts(test_feedback)
    print(f"Violated concepts: {violated}")
    
    print("\n--- Testing Refinement ---")
    refined = refiner(test_concepts, test_feedback, test_dict)
    print("Refined:")
    print(refined)
    
    # Test salvage function with bad output
    print("\n--- Testing Salvage Function ---")
    bad_output = """Here's the refined version:
Color is multiple colors, shape irregular, border is blurry, patterns are atypical.
Hope this helps!"""
    print("Bad output:")
    print(bad_output)
    salvaged = refiner._try_extract_concepts(bad_output, test_dict)
    print("Salvaged:")
    print(salvaged)
    
    print("\n✓ Test complete!")

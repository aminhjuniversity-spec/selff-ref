"""
Master Test Script for Self-Refine Evaluation
Run all 3 testing options from a single interface.
"""

import sys
import os

def print_header():
    print("\n" + "=" * 80)
    print(" " * 20 + "SELF-REFINE TESTING SUITE")
    print("=" * 80)
    print("\nThis suite tests the self-refine mechanism for ExpLICD concept predictions.")
    print("It includes 3 testing options to evaluate if consistency-based refinement works.")
    print()

def print_menu():
    print("\n" + "-" * 80)
    print("AVAILABLE TESTS:")
    print("-" * 80)
    print("\n[1] Option 1: Quick Violation Comparison")
    print("    → Tests self-refine on 10 samples and compares violation counts")
    print("    → Fast (5-10 minutes)")
    print("    → Best for: Quick sanity check")
    
    print("\n[2] Option 2: Full Report Generation")
    print("    → Generates complete concept reports with/without self-refine")
    print("    → Moderate speed (20-30 minutes for full dataset)")
    print("    → Best for: Comprehensive analysis before downloading LLM")
    
    print("\n[3] Option 3: Manual Inspection")
    print("    → Inspect how self-refine works on specific examples")
    print("    → Very fast (seconds)")
    print("    → Best for: Understanding the refinement process")
    
    print("\n[4] Run ALL tests")
    print("    → Runs Option 1 and Option 3 (Option 2 is optional due to time)")
    
    print("\n[0] Exit")
    print("-" * 80)

def run_option1(dataset='PH2', split=0, num_samples=10):
    print("\n" + "=" * 80)
    print("RUNNING OPTION 1: Quick Violation Comparison")
    print("=" * 80)
    
    from test_option1_violation_comparison import test_violation_reduction
    
    results = test_violation_reduction(
        dataset=dataset,
        split=split,
        num_samples=num_samples
    )
    
    return results

def run_option2(dataset='PH2', split=0):
    print("\n" + "=" * 80)
    print("RUNNING OPTION 2: Full Report Generation")
    print("=" * 80)
    print("\n⚠ WARNING: This may take 20-30 minutes for a full dataset!")
    
    confirm = input("Do you want to continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Skipping Option 2.")
        return None
    
    from test_option2_report_comparison import generate_concepts_comparison
    
    results = generate_concepts_comparison(
        dataset=dataset,
        split=split
    )
    
    return results

def run_option3():
    print("\n" + "=" * 80)
    print("RUNNING OPTION 3: Manual Inspection")
    print("=" * 80)
    
    from test_option3_manual_inspection import run_predefined_examples
    
    run_predefined_examples()

def run_all_tests(dataset='PH2', split=0):
    print("\n" + "=" * 80)
    print("RUNNING ALL TESTS")
    print("=" * 80)
    
    # Run Option 1
    print("\n[1/2] Running Option 1...")
    run_option1(dataset=dataset, split=split, num_samples=10)
    
    # Run Option 3
    print("\n[2/2] Running Option 3...")
    run_option3()
    
    # Ask about Option 2
    print("\n" + "=" * 80)
    print("Option 2 (Full Report Generation) is optional and time-consuming.")
    confirm = input("Do you want to run Option 2 as well? (y/n): ").strip().lower()
    if confirm == 'y':
        run_option2(dataset=dataset, split=split)
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED!")
    print("=" * 80)

def main():
    print_header()
    
    # Default parameters
    dataset = 'PH2'
    split = 0
    
    while True:
        print_menu()
        choice = input("\nEnter your choice [0-4]: ").strip()
        
        if choice == '0':
            print("\nExiting. Thank you!")
            break
        
        elif choice == '1':
            # Option 1: Quick test
            print("\nConfiguring Option 1...")
            num_samples = input("Number of samples to test (default 10): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else 10
            
            run_option1(dataset=dataset, split=split, num_samples=num_samples)
            
            input("\nPress Enter to continue...")
        
        elif choice == '2':
            # Option 2: Full report generation
            run_option2(dataset=dataset, split=split)
            
            input("\nPress Enter to continue...")
        
        elif choice == '3':
            # Option 3: Manual inspection
            run_option3()
            
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            # Run all tests
            run_all_tests(dataset=dataset, split=split)
            
            input("\nPress Enter to continue...")
        
        else:
            print("\n⚠ Invalid choice. Please enter 0-4.")

if __name__ == "__main__":
    main()
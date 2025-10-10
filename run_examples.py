"""
Runner script for XGBoost interpretability examples.

This script helps you run the different examples easily.
"""

import os
import sys
import subprocess


def run_example(example_name):
    """Run a specific example."""
    example_path = f"xgboost_interp/examples/{example_name}.py"
    
    if not os.path.exists(example_path):
        print(f"❌ Example not found: {example_path}")
        return False
    
    print(f"🚀 Running {example_name}...")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, example_path], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ {example_name} completed successfully!")
            return True
        else:
            print(f"\n❌ {example_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {example_name}: {e}")
        return False


def main():
    """Main function to run examples."""
    print("XGBoost Interpretability Package - Example Runner")
    print("=" * 55)
    
    examples = {
        "1": ("sklearn_dataset_example", "California Housing (Regression)"),
        "2": ("iris_classification_example", "Iris Classification"),
        "3": ("basic_analysis", "Basic Analysis (requires your model)"),
        "4": ("advanced_analysis", "Advanced Analysis (requires your model)")
    }
    
    print("\nAvailable examples:")
    for key, (name, description) in examples.items():
        print(f"  {key}. {description}")
    
    print("\nRecommended order:")
    print("  - Start with example 1 or 2 (they include data and model training)")
    print("  - Then try examples 3 and 4 with your own models")
    
    while True:
        choice = input(f"\nEnter example number (1-{len(examples)}) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            print("Goodbye!")
            break
        
        if choice in examples:
            example_name, description = examples[choice]
            print(f"\nSelected: {description}")
            
            # Check dependencies for examples 3 and 4
            if choice in ["3", "4"]:
                print("⚠️  Note: Examples 3 and 4 require you to:")
                print("   1. Update the model_path in the example file")
                print("   2. Ensure your data is available")
                proceed = input("   Do you want to continue? (y/n): ").strip().lower()
                if proceed != 'y':
                    continue
            
            success = run_example(example_name)
            
            if success:
                print(f"\n📁 Check the output directory for generated plots and files")
            
        else:
            print("❌ Invalid choice. Please enter a number 1-4 or 'q'.")


if __name__ == "__main__":
    # Change to the package directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()

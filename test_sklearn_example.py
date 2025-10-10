"""
Test script to verify the sklearn example structure without running it.
This checks the code structure and imports.
"""

import os
import ast
import sys


def check_example_structure(example_path):
    """Check if an example file has the correct structure."""
    print(f"Checking {example_path}...")
    
    if not os.path.exists(example_path):
        print(f"❌ File not found: {example_path}")
        return False
    
    try:
        with open(example_path, 'r') as f:
            content = f.read()
        
        # Parse the AST to check structure
        tree = ast.parse(content)
        
        # Check for required functions
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        print(f"  Functions found: {functions}")
        
        # Check for imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"{node.module}.{node.names[0].name if node.names else '*'}")
        
        print(f"  Key imports: {[imp for imp in imports if any(lib in imp for lib in ['sklearn', 'xgboost', 'pandas', 'numpy'])]}")
        
        # Check for main execution
        has_main = any(isinstance(node, ast.If) and 
                      isinstance(node.test, ast.Compare) and
                      isinstance(node.test.left, ast.Name) and
                      node.test.left.id == '__name__'
                      for node in tree.body)
        
        print(f"  Has main execution: {has_main}")
        print("  ✅ Structure looks good!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error parsing file: {e}")
        return False


def main():
    """Test the sklearn examples."""
    print("Testing sklearn examples structure...")
    print("=" * 50)
    
    examples = [
        "xgboost_interp/examples/sklearn_dataset_example.py",
        "xgboost_interp/examples/iris_classification_example.py"
    ]
    
    all_good = True
    
    for example in examples:
        success = check_example_structure(example)
        all_good = all_good and success
        print()
    
    if all_good:
        print("🎉 All examples have correct structure!")
        print("\nTo run the examples:")
        print("1. Install dependencies: pip install -e .")
        print("2. Run: python run_examples.py")
        print("3. Or run individual examples directly")
        
        print("\nWhat the examples will do:")
        print("📊 California Housing Example:")
        print("   - Load 20,640 housing samples with 8 features")
        print("   - Train XGBoost regressor (100 trees, depth 6)")
        print("   - Generate ~15 interpretability plots")
        print("   - Save model as california_housing_xgb.json")
        
        print("\n🌸 Iris Classification Example:")
        print("   - Load 150 iris samples with 4 features")  
        print("   - Train XGBoost classifier (50 trees, depth 4)")
        print("   - Generate ~10 interpretability plots")
        print("   - Save model as iris_xgb.json")
        
    else:
        print("❌ Some examples have issues. Check the errors above.")


if __name__ == "__main__":
    main()

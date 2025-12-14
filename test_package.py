"""
Simple test to verify the package imports correctly.
"""

def test_imports():
    """Test that all main components can be imported."""
    try:
        from xgboost_interp import TreeAnalyzer, ModelAnalyzer
        print("✅ Core classes imported successfully")
        
        from xgboost_interp.plotting import BasePlotter, FeaturePlotter, TreePlotter, InteractivePlotter
        print("✅ Plotting classes imported successfully")
        
        from xgboost_interp.utils import ModelLoader, DataLoader
        print("✅ Utility classes imported successfully")
        
        print("🎉 All imports successful! Package is ready to use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without requiring actual model files."""
    try:
        from xgboost_interp.utils.model_utils import ModelLoader
        
        # Test utility functions
        assert ModelLoader._safe_int("123") == 123
        assert ModelLoader._safe_int("invalid") is None
        assert ModelLoader._safe_float("12.34") == 12.34
        
        print("✅ Basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False


if __name__ == "__main__":
    print("Testing XGBoost Interpretability Package")
    print("=" * 40)
    
    import_success = test_imports()
    func_success = test_basic_functionality()
    
    if import_success and func_success:
        print("\n🎉 All tests passed! The package is working correctly.")
        print("\nNext steps:")
        print("1. Replace 'your_model.json' in examples with your actual model path")
        print("2. Run: python -m xgboost_interp.examples.basic_analysis")
        print("3. Or use the package in your own scripts")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

"""
Demo script showing how to analyze a specific class in a multi-class XGBoost model.

This script demonstrates the new multi-class support that allows you to:
1. Analyze the predicted probability for a specific class
2. Generate class-specific PDPs and marginal impact plots
3. Compare different classes side-by-side
"""

from xgboost_interp import TreeAnalyzer, ModelAnalyzer


def analyze_single_class_example():
    """Example: Analyze only class 0 (setosa) from the Iris model."""
    
    print("="*60)
    print("DEMO: Analyzing Only Class 0 (Setosa)")
    print("="*60)
    
    # Load the model
    model_path = "iris_xgb.json"
    tree_analyzer = TreeAnalyzer(model_path)
    
    # Initialize ModelAnalyzer with target_class=0 for setosa
    model_analyzer = ModelAnalyzer(tree_analyzer, target_class=0)
    
    # Load data
    model_analyzer.load_data_from_parquets("iris_data", num_files_to_read=1)
    model_analyzer.load_xgb_model()
    
    print("\n✅ Model loaded - analyzing probability of class 0 (setosa)")
    print("\nGenerating analysis plots...")
    
    # Generate PDP for a feature - this will show how it affects P(setosa)
    model_analyzer.plot_partial_dependence(
        feature_name="petal length (cm)",
        grid_points=30,
        n_curves=150
    )
    
    # Generate marginal impact - shows how splits affect P(setosa)
    model_analyzer.plot_marginal_impact_univariate("petal length (cm)")
    
    print(f"\n✅ Class 0 analysis complete!")
    print(f"   Plots saved to: {tree_analyzer.plotter.save_dir}")


def compare_classes_example():
    """Example: Compare how features affect different classes."""
    
    print("\n" + "="*60)
    print("DEMO: Comparing Feature Impact Across Classes")
    print("="*60)
    
    model_path = "iris_xgb.json"
    tree_analyzer = TreeAnalyzer(model_path)
    
    class_names = ['setosa', 'versicolor', 'virginica']
    feature = "petal length (cm)"
    
    print(f"\nAnalyzing how '{feature}' affects each class probability:")
    
    for class_idx in range(3):
        print(f"\n  Analyzing class {class_idx} ({class_names[class_idx]})...")
        
        # Create analyzer for this specific class
        model_analyzer = ModelAnalyzer(tree_analyzer, target_class=class_idx)
        model_analyzer.load_data_from_parquets("iris_data", num_files_to_read=1)
        model_analyzer.load_xgb_model()
        
        # Generate class-specific PDP
        model_analyzer.plot_partial_dependence(
            feature_name=feature,
            grid_points=30,
            n_curves=150
        )
        
        print(f"    ✅ PDP shows impact on P({class_names[class_idx]})")
    
    print(f"\n✅ Comparison complete! Check the plots to see how the same")
    print(f"   feature affects different class probabilities differently.")


def main():
    """Run all demo examples."""
    
    print("\n" + "="*60)
    print("XGBoost Multi-Class Interpretability Demo")
    print("="*60)
    print("\nThis demo shows how to analyze specific classes in a")
    print("multi-class XGBoost model.\n")
    
    # Demo 1: Single class analysis
    analyze_single_class_example()
    
    # Demo 2: Compare across classes
    compare_classes_example()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. Use ModelAnalyzer(tree_analyzer, target_class=0) to analyze class 0
2. All PDPs and marginal impact plots will show effects on that class
3. You can create multiple analyzers to compare different classes
4. Each class has its own set of trees in the XGBoost model

USAGE:
    tree_analyzer = TreeAnalyzer("model.json")
    
    # Analyze class 0
    analyzer_class0 = ModelAnalyzer(tree_analyzer, target_class=0)
    analyzer_class0.load_xgb_model()
    analyzer_class0.plot_partial_dependence("feature_name")
    
    # Analyze class 1
    analyzer_class1 = ModelAnalyzer(tree_analyzer, target_class=1)
    analyzer_class1.load_xgb_model()
    analyzer_class1.plot_partial_dependence("feature_name")
""")


if __name__ == "__main__":
    main()


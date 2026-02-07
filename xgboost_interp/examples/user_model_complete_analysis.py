"""
Complete analysis example for user-provided XGBoost model.

This example demonstrates how to run ALL available analysis and plotting functions
on your own pre-trained XGBoost model saved as JSON.

Usage:
    python user_model_complete_analysis.py /path/to/your/model.json [/path/to/data/]

Requirements:
    - XGBoost model saved as JSON file
    - (Optional) Data directory with parquet files for data-dependent analysis
"""

import os
import sys
import argparse
import time

# Add the package to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xgboost_interp import TreeAnalyzer, ModelAnalyzer
from xgboost_interp.plotting import FeaturePlotter, TreePlotter, InteractivePlotter
from xgboost_interp.utils import AnalysisTracker


def run_all_tree_level_analysis(tree_analyzer, tracker=None):
    """
    Run ALL tree-level analysis functions (no data required).
    
    This includes:
    - Model summary
    - Feature importance plots (multiple types)
    - Tree structure analysis
    - Feature co-occurrence analysis
    - Advanced tree statistics
    - Marginal impact analysis (tree-level)
    """
    if tracker is None:
        tracker = AnalysisTracker()
    
    start_time = time.time()
    print("\n" + "="*70)
    print("PART 1: TREE-LEVEL ANALYSIS (No Data Required)")
    print("="*70)
    print("Note: Marginal impact plots are included here - they only need tree structure!")
    
    # 1. Print model summary
    print("\n[1/15] Printing model summary...")
    tree_analyzer.print_model_summary()
    
    # 2. Combined feature importance (weight, gain, cover)
    print("\n[2/15] Generating combined feature importance plot...")
    try:
        tree_analyzer.plot_feature_importance_combined()
        print("Generated: feature_importance_combined.png")
        tracker.success("Feature importance combined")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Feature importance combined", e)
    
    # 3. Feature importance distributions (boxplots)
    print("\n[3/15] Generating feature importance distributions...")
    try:
        tree_analyzer.plot_feature_importance_distributions(log_scale=True)
        print("Generated: feature_weight.png")
        print("Generated: feature_gain_distribution.png")
        print("Generated: feature_cover_distribution.png")
        tracker.success("Feature importance distributions")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Feature importance distributions", e)
    
    # 4. Feature importance scatter plot
    print("\n[4/15] Generating feature importance scatter plot...")
    try:
        tree_analyzer.plot_feature_importance_scatter()
        print("Generated: feature_importance_scatter.png")
        tracker.success("Feature importance scatter")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Feature importance scatter", e)
    
    # 5. Tree depth histogram
    print("\n[5/15] Generating tree depth histogram...")
    try:
        tree_analyzer.plot_tree_depth_histogram()
        print("Generated: tree_depth_histogram.png")
        tracker.success("Tree depth histogram")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Tree depth histogram", e)
    
    # 6. Cumulative gain
    print("\n[6/15] Generating cumulative gain plot...")
    try:
        tree_analyzer.plot_cumulative_gain()
        print("Generated: cumulative_gain.png")
        tracker.success("Cumulative gain")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Cumulative gain", e)
    
    # 7. Cumulative prediction shift
    print("\n[7/15] Generating cumulative prediction shift plot...")
    try:
        tree_analyzer.plot_cumulative_prediction_shift()
        print("Generated: cumulative_prediction_shift.png")
        tracker.success("Cumulative prediction shift")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Cumulative prediction shift", e)
    
    # 8. Tree-level feature co-occurrence
    print("\n[8/15] Generating tree-level feature co-occurrence heatmap...")
    try:
        tree_analyzer.plot_tree_level_feature_cooccurrence()
        print("Generated: feature_cooccurrence_tree_level.png")
        tracker.success("Tree-level feature co-occurrence")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Tree-level feature co-occurrence", e)
    
    # 9. Path-level feature co-occurrence
    print("\n[9/16] Generating path-level feature co-occurrence heatmap...")
    try:
        tree_analyzer.plot_path_level_feature_cooccurrence()
        print("Generated: feature_cooccurrence_path_level.png")
        tracker.success("Path-level feature co-occurrence")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Path-level feature co-occurrence", e)
    
    # 10. Sequential feature co-occurrence
    print("\n[10/16] Generating sequential feature co-occurrence heatmap...")
    try:
        tree_analyzer.plot_sequential_feature_dependency()
        print("Generated: feature_cooccurrence_sequential.png")
        tracker.success("Sequential feature dependency")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Sequential feature dependency", e)
    
    # Initialize advanced plotters
    feature_plotter = FeaturePlotter(tree_analyzer.plotter.save_dir)
    tree_plotter = TreePlotter(tree_analyzer.plotter.save_dir)
    
    # 11. Feature usage heatmap
    print("\n[11/16] Generating feature usage heatmap...")
    try:
        feature_plotter.plot_feature_usage_heatmap(
            tree_analyzer.trees, 
            tree_analyzer.feature_names, 
            log_scale=True
        )
        print("Generated: feature_usage_heatmap.png")
        tracker.success("Feature usage heatmap")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Feature usage heatmap", e)
    
    # 12. Split depth per feature
    print("\n[12/16] Generating split depth per feature plot...")
    try:
        feature_plotter.plot_split_depth_per_feature(
            tree_analyzer.trees, 
            tree_analyzer.feature_names
        )
        print("Generated: split_depth_per_feature.png")
        tracker.success("Split depth per feature")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Split depth per feature", e)
    
    # 13. Feature split impact
    print("\n[13/16] Generating feature split impact plot...")
    try:
        feature_plotter.plot_feature_split_impact(
            tree_analyzer.trees, 
            tree_analyzer.feature_names, 
            log_scale=False
        )
        print("Generated: feature_split_impact.png")
        tracker.success("Feature split impact")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Feature split impact", e)
    
    # 13. Prediction and gain statistics
    print("\n[14/16] Generating prediction and gain statistics plots...")
    try:
        tree_plotter.plot_prediction_and_gain_stats(
            tree_analyzer.trees, 
            log_scale=False
        )
        print("Generated: prediction_stats_per_tree.png")
        print("Generated: prediction_stats_by_depth.png")
        print("Generated: gain_stats_per_tree.png")
        print("Generated: gain_stats_by_depth.png")
        tracker.success("Prediction and gain statistics")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Prediction and gain statistics", e)
    
    # 14. Gain heatmap
    print("\n[15/16] Generating gain heatmap...")
    try:
        tree_plotter.plot_gain_heatmap(
            tree_analyzer.trees, 
            tree_analyzer.feature_names
        )
        print("Generated: gain_heatmap.png")
        tracker.success("Gain heatmap")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Gain heatmap", e)
    
    # 15. Marginal impact analysis (NO DATA REQUIRED)
    print("\n[16/16] Generating marginal impact plots for all features...")
    print("(This analyzes tree structure only - no data needed)")
    
    # Create a temporary ModelAnalyzer just for marginal impact (doesn't need data loaded)
    from xgboost_interp import ModelAnalyzer
    temp_analyzer = ModelAnalyzer(tree_analyzer, target_class=0)
    
    marginal_success_count = 0
    feature_names = tree_analyzer.feature_names
    
    for i, feature in enumerate(feature_names, 1):
        try:
            print(f"  [{i}/{len(feature_names)}] Computing marginal impact for '{feature}'...")
            temp_analyzer.plot_marginal_impact_univariate(feature, scale="linear")
            marginal_success_count += 1
            print(f"  Generated: marginal_impact/{feature}.png")
            tracker.success(f"Marginal impact: {feature}")
        except Exception as e:
            print(f"  Failed for {feature}: {e}")
            tracker.failure(f"Marginal impact: {feature}", e)
    
    print(f"\nGenerated {marginal_success_count}/{len(feature_names)} marginal impact plots")
    print(f"   Saved in: {tree_analyzer.plotter.save_dir}/marginal_impact/")
    
    # 16. Interactive tree visualization
    print("\n[16/16] Generating interactive tree visualizations...")
    try:
        interactive_plotter = InteractivePlotter(tree_analyzer.plotter.save_dir)
        # Plot first 5 trees individually
        num_trees_to_plot = min(5, len(tree_analyzer.trees))
        interactive_plotter.plot_interactive_trees(
            tree_analyzer.trees[:num_trees_to_plot], 
            tree_analyzer.feature_names, 
            top_k=num_trees_to_plot, 
            combined=False
        )
        print(f"Generated: {num_trees_to_plot} interactive tree PNG files")
        tracker.success("Interactive tree plots")
    except ImportError:
        print("Plotly not installed - skipping interactive plots")
        print("   Install with: pip install plotly networkx")
        tracker.failure("Interactive tree plots", "Plotly not installed")
    except Exception as e:
        print(f"Error: {e}")
        tracker.failure("Interactive tree plots", e)
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"PART 1 COMPLETE - Tree-level analysis finished in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("="*70)


def run_all_data_dependent_analysis(model_analyzer, tree_analyzer, data_dir, target_class=None, plotting_mode='raw', target_column=None, tracker=None):
    """
    Run ALL data-dependent analysis functions.
    
    This includes:
    - Partial dependence plots (PDP with ICE curves)
    - Prediction evolution analysis (probability across trees)
    - ALE plots (if pyALE installed)
    
    Note: Marginal impact analysis is in Part 1 (tree-level) - it doesn't need data!
    
    Args:
        model_analyzer: ModelAnalyzer instance
        tree_analyzer: TreeAnalyzer instance
        data_dir: Directory containing parquet files
        target_class: Target class for multi-class models (None for binary/regression)
        plotting_mode: Y-axis mode for PDPs and score plots ('raw', 'probability', or 'logit')
        tracker: Optional AnalysisTracker instance for failure tracking
    """
    if tracker is None:
        tracker = AnalysisTracker()
    start_time = time.time()
    print("\n" + "="*70)
    print("PART 2: DATA-DEPENDENT ANALYSIS (Requires Data)")
    print("="*70)
    print(f"Plotting mode: {plotting_mode}")
    
    # Load data
    print("\n[1/5] Loading data from parquet files...")
    try:
        model_analyzer.load_data_from_parquets(data_dir, num_files_to_read=1000)
        print(f"Loaded data: {len(model_analyzer.df)} records")
        tracker.success("Load data from parquets")
    except Exception as e:
        print(f"Failed to load data: {e}")
        tracker.failure("Load data from parquets", e)
        return False
    
    # Load XGBoost model
    print("\n[2/5] Loading XGBoost model for predictions...")
    try:
        model_analyzer.load_xgb_model()
        print("Model loaded successfully")
        tracker.success("Load XGBoost model")
    except Exception as e:
        print(f"Failed to load model: {e}")
        tracker.failure("Load XGBoost model", e)
        return False
    
    # Model performance metrics (if target column provided)
    if target_column and target_column in model_analyzer.df.columns:
        print(f"\n[2.5/5] Computing model performance metrics...")
        y_true = model_analyzer.df[target_column].values
        feature_names = tree_analyzer.feature_names
        X = model_analyzer.df[feature_names].values
        
        # Get predictions based on model type
        is_binary = False
        if model_analyzer.is_regression:
            y_pred = model_analyzer.xgb_model.predict(X)
        else:
            y_pred = model_analyzer.xgb_model.predict_proba(X)
            if y_pred.ndim == 2 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]  # Binary: use positive class proba
                is_binary = True
        
        metrics = model_analyzer.evaluate_model_performance(y_true, y_pred)
        print("Model Performance Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {round(v, 6)}")
        print(f"  Saved to: {tree_analyzer.plotter.save_dir}/model_performance_metrics.txt")
        
        # Calibration curves (binary classification only)
        if is_binary:
            print("\nGenerating calibration curves...")
            model_analyzer.generate_calibration_curves(y_true, y_pred, X=X, n_bins=10)
    elif target_column:
        print(f"\n[2.5/5] Target column '{target_column}' not found in data. Skipping metrics.")
    
    # Partial Dependence Plots for all features
    print("\n[3/4] Generating Partial Dependence Plots (PDPs) for all features...")
    print("This may take a few minutes depending on dataset size...")
    pdp_start_time = time.time()
    
    feature_names = tree_analyzer.feature_names
    pdp_success_count = 0
    
    for i, feature in enumerate(feature_names, 1):
        try:
            print(f"  [{i}/{len(feature_names)}] Computing PDP for '{feature}'...")
            model_analyzer.plot_partial_dependence(
                feature_name=feature,
                n_curves=min(1000, len(model_analyzer.df)),
                mode=plotting_mode
            )
            pdp_success_count += 1
            print(f"  Generated: pdp/{feature}.png")
            tracker.success(f"PDP: {feature}")
        except Exception as e:
            print(f"  Failed for {feature}: {e}")
            tracker.failure(f"PDP: {feature}", e)
    
    pdp_elapsed = time.time() - pdp_start_time
    print(f"\nGenerated {pdp_success_count}/{len(feature_names)} PDP plots in {pdp_elapsed:.2f}s")
    print(f"   Saved in: {tree_analyzer.plotter.save_dir}/pdp/")
    
    # Prediction evolution across trees
    print("\n[4/5] Generating prediction evolution analysis...")
    try:
        # Create tree indices using quantiles (1, 20%, 40%, 60%, 80%, 100%)
        tree_indices = model_analyzer._get_default_tree_indices()
        
        print(f"  Analyzing predictions at tree indices: {tree_indices}")
        model_analyzer.plot_scores_across_trees(
            tree_indices=tree_indices,
            n_records=min(1000, len(model_analyzer.df)),
            mode=plotting_mode
        )
        print("  Generated: scores_across_trees.png")
        tracker.success("Scores across trees")
    except Exception as e:
        print(f"  Failed: {e}")
        tracker.failure("Scores across trees", e)
    
    # Early exit performance analysis
    print("\n[5/5] Generating early exit performance analysis...")
    try:
        model_analyzer.analyze_early_exit_performance(
            n_records=min(5000, len(model_analyzer.df)),
            n_detailed_curves=1000
        )
        tracker.success("Early exit analysis")
    except Exception as e:
        print(f"Could not generate early exit analysis: {e}")
        tracker.failure("Early exit analysis", e)
    
    # ALE plots (optional - requires pyALE)
    print("\n[BONUS] Attempting to generate ALE plots (requires pyALE)...")
    ale_start_time = time.time()
    try:
        from PyALE import ale
        print("  pyALE detected! Generating ALE plots for all features...")
        
        # Generate ALE for ALL features
        all_features = tree_analyzer.feature_names
        ale_success_count = 0
        
        for i, feature in enumerate(all_features, 1):
            try:
                print(f"  [{i}/{len(all_features)}] Computing ALE for '{feature}'...")
                model_analyzer.plot_ale(
                    feature_name=feature,
                    grid_size=100,
                    include_CI=True,
                    n_curves=min(10000, len(model_analyzer.df))
                )
                ale_success_count += 1
                print(f"  Generated: ALE_analysis/{feature}.png")
                tracker.success(f"ALE: {feature}")
            except Exception as e:
                print(f"  Failed for {feature}: {e}")
                tracker.failure(f"ALE: {feature}", e)
        
        ale_elapsed = time.time() - ale_start_time
        print(f"\n   Generated {ale_success_count}/{len(all_features)} ALE plots in {ale_elapsed:.2f}s")
        print(f"   ALE plots saved in: {tree_analyzer.plotter.save_dir}/ALE_analysis/")
        
    except ImportError:
        print("  pyALE not installed - skipping ALE plots")
        print("     Install with: pip install pyALE")
        tracker.failure("ALE plots", "pyALE not installed")
    
    # SHAP Analysis
    print("\n[BONUS] Attempting to generate SHAP analysis plots...")
    shap_start_time = time.time()
    try:
        import shap
        import matplotlib.pyplot as plt
        print("  SHAP detected! Generating comprehensive SHAP analysis...")
        
        # Sample 1000 rows for SHAP computation - only use model features
        X_sample = model_analyzer.df[tree_analyzer.feature_names].sample(n=min(1000, len(model_analyzer.df)), random_state=42)
        
        # Create SHAP explainer using TreeExplainer
        print(f"  Creating TreeExplainer for {len(X_sample)} samples...")
        explainer = shap.TreeExplainer(model_analyzer.xgb_model)
        print("  Computing SHAP values...")
        shap_values = explainer(X_sample)
        
        # Create output directories
        shap_dir = os.path.join(tree_analyzer.plotter.save_dir, 'SHAP_analysis')
        shap_dep_dir = os.path.join(shap_dir, 'SHAP_dependence_plots')
        os.makedirs(shap_dir, exist_ok=True)
        os.makedirs(shap_dep_dir, exist_ok=True)
        
        # 1. Summary Bar Plot (mean absolute SHAP values)
        print("  [1/4] Generating summary bar plot...")
        plt.figure()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, 'summary_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Generated: SHAP_analysis/summary_bar.png")
        
        # 2. Summary Beeswarm Plot (SHAP value distributions)
        print("  [2/4] Generating summary beeswarm plot...")
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_dir, 'summary_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Generated: SHAP_analysis/summary_beeswarm.png")
        
        # 3. Dependence Plots (one per feature)
        print(f"  [3/4] Generating dependence plots for {len(tree_analyzer.feature_names)} features...")
        for i, feature in enumerate(tree_analyzer.feature_names):
            plt.figure()
            shap.dependence_plot(i, shap_values.values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dep_dir, f'dependence_{feature}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        print(f"  Generated: SHAP_analysis/SHAP_dependence_plots/ ({len(tree_analyzer.feature_names)} plots)")
        
        # 4. Waterfall Plots (first 5 samples)
        print("  [4/4] Generating waterfall plots for first 5 samples...")
        for idx in range(min(5, len(X_sample))):
            plt.figure()
            shap.waterfall_plot(shap_values[idx], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(shap_dir, f'waterfall_sample_{idx}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        print(f"  Generated: SHAP_analysis/waterfall_sample_0-{min(5, len(X_sample))-1}.png")
        
        shap_elapsed = time.time() - shap_start_time
        print(f"\n   SHAP analysis complete in {shap_elapsed:.2f}s")
        print(f"   SHAP analysis plots saved in: {shap_dir}")
        tracker.success("SHAP analysis")
        
    except ImportError:
        print("  shap not installed - skipping SHAP analysis")
        print("     Install with: pip install shap")
        tracker.failure("SHAP analysis", "shap not installed")
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"PART 2 COMPLETE - Data-dependent analysis finished in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("="*70)
    
    return True


def generate_summary_report(tree_analyzer, data_analysis_done):
    """Generate a summary report of all generated files."""
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY REPORT")
    print("="*70)
    
    output_dir = tree_analyzer.plotter.save_dir
    
    if os.path.exists(output_dir):
        # Count PNG files in main directory
        main_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        
        # Count files in subdirectories
        pdp_dir = os.path.join(output_dir, 'pdp')
        marginal_dir = os.path.join(output_dir, 'marginal_impact')
        ale_dir = os.path.join(output_dir, 'ale')
        
        pdp_count = len(os.listdir(pdp_dir)) if os.path.exists(pdp_dir) else 0
        marginal_count = len(os.listdir(marginal_dir)) if os.path.exists(marginal_dir) else 0
        ale_count = len(os.listdir(ale_dir)) if os.path.exists(ale_dir) else 0
        
        total_plots = len(main_files) + pdp_count + marginal_count + ale_count
        
        print(f"\nOutput Directory: {output_dir}")
        print(f"Total Plots Generated: {total_plots}")
        
        # Categorize main directory plots
        tree_plots = [f for f in main_files if any(x in f for x in ['tree', 'depth', 'gain', 'cumulative', 'prediction_stats'])]
        feature_plots = [f for f in main_files if 'feature' in f or 'split' in f or 'cooccurrence' in f]
        other_plots = [f for f in main_files if f not in tree_plots + feature_plots]
        
        print(f"\nPlot Categories:")
        print(f"  Tree Structure Plots: {len(tree_plots)} (in main directory)")
        print(f"  Feature Analysis Plots: {len(feature_plots)} (in main directory)")
        
        # Always show marginal impact (Part 1 - no data needed)
        if marginal_count > 0:
            print(f"  Marginal Impact Plots: {marginal_count} (in marginal_impact/)")
        
        if data_analysis_done:
            if pdp_count > 0:
                print(f"  Partial Dependence Plots: {pdp_count} (in pdp/)")
            if ale_count > 0:
                print(f"  ALE Plots: {ale_count} (in ale/)")
            if other_plots:
                print(f"  Other Plots: {len(other_plots)} (in main directory)")
        
        print(f"\nDirectory Structure:")
        print(f"  {output_dir}/")
        print(f"    ├── {len(main_files)} plots (tree structure, feature importance, etc.)")
        if marginal_count > 0:
            print(f"    ├── marginal_impact/ ({marginal_count} plots)")
        if pdp_count > 0:
            print(f"    ├── pdp/ ({pdp_count} plots)")
        if ale_count > 0:
            print(f"    └── ale/ ({ale_count} plots)")
        
    else:
        print(f"\nOutput directory not found: {output_dir}")
    
    print("\n" + "="*70)


def main():
    """Main function to run complete analysis."""
    parser = argparse.ArgumentParser(
        description="Complete XGBoost model analysis with all available functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tree-level analysis only (no data needed)
  python user_model_complete_analysis.py model.json
  
  # Complete analysis with data
  python user_model_complete_analysis.py model.json data_directory/
  
  # Multi-class model - analyze specific class
  python user_model_complete_analysis.py model.json data_directory/ --target-class 0
  
  # Use different plotting mode (probability with corrected base_score)
  python user_model_complete_analysis.py model.json data_directory/ --plotting_mode probability
  
  # Use logit scale for plots
  python user_model_complete_analysis.py model.json data_directory/ --plotting_mode logit

For multi-class models, you can run this script multiple times with different
--target-class values to analyze each class separately.
        """
    )
    
    parser.add_argument(
        'model_path',
        help='Path to XGBoost model JSON file'
    )
    
    parser.add_argument(
        'data_dir',
        nargs='?',
        default=None,
        help='(Optional) Directory containing parquet files for data-dependent analysis'
    )
    
    parser.add_argument(
        '--target-class',
        type=int,
        default=None,
        help='Target class index for multi-class models (default: None for binary/regression, 0 for multi-class)'
    )
    
    parser.add_argument(
        '--plotting_mode',
        type=str,
        default='raw',
        choices=['raw', 'probability', 'logit'],
        help='Y-axis mode for PDPs and score evolution plots (default: raw). Options: raw, probability, logit'
    )
    
    parser.add_argument(
        '--target-column',
        type=str,
        default=None,
        help='Name of target column in data for computing model performance metrics'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if args.data_dir and not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Print header
    total_start_time = time.time()
    print("\n" + "="*70)
    print("XGBoost Model Complete Analysis")
    print("="*70)
    print(f"Model: {args.model_path}")
    if args.data_dir:
        print(f"Data:  {args.data_dir}")
    else:
        print(f"Data:  None (tree-level analysis only)")
    if args.target_class is not None:
        print(f"Target Class: {args.target_class}")
    print("="*70)
    
    # Initialize analysis tracker
    tracker = AnalysisTracker()
    
    # Initialize TreeAnalyzer
    print("\nInitializing TreeAnalyzer...")
    try:
        tree_analyzer = TreeAnalyzer(args.model_path)
        print("TreeAnalyzer initialized successfully")
    except Exception as e:
        print(f"Failed to initialize TreeAnalyzer: {e}")
        sys.exit(1)
    
    # Run all tree-level analysis
    run_all_tree_level_analysis(tree_analyzer, tracker=tracker)
    
    # Run data-dependent analysis if data directory provided
    data_analysis_done = False
    if args.data_dir:
        print("\nInitializing ModelAnalyzer for data-dependent analysis...")
        try:
            model_analyzer = ModelAnalyzer(tree_analyzer, target_class=args.target_class or 0)
            print("ModelAnalyzer initialized successfully")
            
            data_analysis_done = run_all_data_dependent_analysis(
                model_analyzer, 
                tree_analyzer, 
                args.data_dir,
                args.target_class,
                args.plotting_mode,
                args.target_column,
                tracker=tracker
            )
        except Exception as e:
            print(f"Failed to run data-dependent analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "="*70)
        print(" SKIPPING DATA-DEPENDENT ANALYSIS")
        print("="*70)
        print("No data directory provided. To run complete analysis including:")
        print("  - Partial Dependence Plots (PDP)")
        print("  - Marginal Impact Analysis")
        print("  - Prediction Evolution")
        print("  - ALE Plots")
        print("\nRe-run with: python user_model_complete_analysis.py MODEL.json DATA_DIR/")
        print("="*70)
    
    # Print failure/success summary
    tracker.print_summary()
    
    # Generate summary report
    generate_summary_report(tree_analyzer, data_analysis_done)
    
    # Final message
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {tree_analyzer.plotter.save_dir}/")
    print("\nYou can now:")
    print("  1. Review the generated plots")
    print("  2. Share them in reports or presentations")
    print("  3. Use them to understand your model's behavior")
    
    if not args.data_dir:
        print("\nTip: Run with data directory for complete analysis including PDPs!")
    
    # Check if model is multi-class
    is_multiclass = False
    num_classes = 0
    
    # Detect multi-class from objective
    objective_str = str(tree_analyzer.objective)
    if 'multi:' in objective_str or 'softmax' in objective_str or 'softprob' in objective_str:
        is_multiclass = True
        # Try to infer number of classes from tree count
        if tree_analyzer.num_trees_total and tree_analyzer.num_trees_outer:
            num_classes = tree_analyzer.num_trees_total // tree_analyzer.num_trees_outer
            if num_classes < 2:
                num_classes = 3  # Default guess
    
    if is_multiclass and args.target_class is None and num_classes > 0:
        print(f"\nTip: This appears to be a multi-class model (~{num_classes} classes).")
        print("    Re-run with --target-class to analyze different classes:")
        for i in range(num_classes):
            print(f"      python user_model_complete_analysis.py {args.model_path} {args.data_dir or ''} --target-class {i}")
    
    total_elapsed_time = time.time() - total_start_time
    print("\n" + "="*70)
    print(f" TOTAL EXECUTION TIME: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


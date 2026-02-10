"""
Model diffing CLI for comparing two user-provided XGBoost models.

This script runs the full ModelDiff comparison between two XGBoost models
saved as JSON files. It produces tree-structure comparisons automatically,
and data-dependent comparisons (PDP, predictions, Q-Q plot) when a data
directory is provided.

Usage:
    # Tree-level comparison only (no data needed)
    python user_model_diff.py MODEL_A.json MODEL_B.json

    # Full comparison with data
    python user_model_diff.py MODEL_A.json MODEL_B.json DATA_DIR/

    # Custom labels and output directory
    python user_model_diff.py MODEL_A.json MODEL_B.json DATA_DIR/ \
        --label-a "Baseline v3.6" --label-b "Candidate v3.7" \
        --output-dir /path/to/output/

Requirements:
    - Two XGBoost models saved as JSON files
    - (Optional) Data directory with parquet files for data-dependent analysis
"""

import os
import time

from xgboost_interp import TreeAnalyzer, ModelAnalyzer
from xgboost_interp.core import ModelDiff
from xgboost_interp.utils import AnalysisTracker


def run_tree_level_comparison(model_diff, tracker=None):
    """
    Run all tree-level ModelDiff comparisons (no data required).

    This includes:
    - Side-by-side model metadata summary
    - Feature change detection (new / dropped / shared)
    - Cumulative gain overlay
    - Feature importance scatter plots (gain, weight, cover)
    """
    if tracker is None:
        tracker = AnalysisTracker()
    start_time = time.time()
    print("\n" + "=" * 70)
    print("PART 1: TREE-LEVEL COMPARISON (No Data Required)")
    print("=" * 70)

    # 1. Print summary
    print("\n[1/4] Model summary...")
    model_diff.print_summary()

    # 2. Feature changes
    print("\n[2/4] Feature changes...")
    feature_changes = model_diff.find_feature_changes()
    print(f"  New in {model_diff.label_b}: {feature_changes['new_in_b'] or '(none)'}")
    print(f"  Dropped in {model_diff.label_b}: {feature_changes['dropped_in_b'] or '(none)'}")
    print(f"  In both models: {len(feature_changes['in_both'])} features")

    # 3. Cumulative gain comparison
    print("\n[3/4] Cumulative gain comparison...")
    try:
        model_diff.compare_cumulative_gain()
        print("  Generated: cumulative_gain_comparison.png")
        tracker.success("Cumulative gain comparison")
    except Exception as e:
        print(f"  Error: {e}")
        tracker.failure("Cumulative gain comparison", e)

    # 4. Feature importance scatters
    print("\n[4/4] Feature importance scatter plots...")
    try:
        model_diff.plot_all_importance_scatters()
        print("  Generated: importance_scatter_gain.png")
        print("  Generated: importance_scatter_weight.png")
        print("  Generated: importance_scatter_cover.png")
        tracker.success("Feature importance scatters")
    except Exception as e:
        print(f"  Error: {e}")
        tracker.failure("Feature importance scatters", e)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PART 1 COMPLETE - Tree-level comparison finished in {elapsed:.2f} seconds")
    print("=" * 70)


def run_data_dependent_comparison(
    model_diff,
    tree_analyzer_a,
    tree_analyzer_b,
    model_a_path,
    model_b_path,
    data_dir,
    target_class,
    plotting_mode,
    target_column,
    tracker=None,
):
    """
    Run all data-dependent ModelDiff comparisons.

    This includes:
    - PDP comparisons for all common features
    - Prediction comparison (scatter, histogram, correlations, Q-Q plot)
    - Classification agreement matrix (if target column provided)
    """
    if tracker is None:
        tracker = AnalysisTracker()
    start_time = time.time()
    print("\n" + "=" * 70)
    print("PART 2: DATA-DEPENDENT COMPARISON (Requires Data)")
    print("=" * 70)
    print(f"Plotting mode: {plotting_mode}")

    # Initialize ModelAnalyzers
    print("\n[1/3] Loading data and models...")
    try:
        model_analyzer_a = ModelAnalyzer(tree_analyzer_a, target_class=target_class or 0)
        model_analyzer_a.load_data_from_parquets(data_dir, num_files_to_read=1000)
        model_analyzer_a.load_xgb_model(model_a_path)
        print(f"  {model_diff.label_a}: loaded {len(model_analyzer_a.df)} records")
        tracker.success(f"Load data + model: {model_diff.label_a}")
    except Exception as e:
        print(f"  Failed to load {model_diff.label_a}: {e}")
        tracker.failure(f"Load data + model: {model_diff.label_a}", e)
        return False

    try:
        model_analyzer_b = ModelAnalyzer(tree_analyzer_b, target_class=target_class or 0)
        model_analyzer_b.load_data_from_parquets(data_dir, num_files_to_read=1000)
        model_analyzer_b.load_xgb_model(model_b_path)
        print(f"  {model_diff.label_b}: loaded {len(model_analyzer_b.df)} records")
        tracker.success(f"Load data + model: {model_diff.label_b}")
    except Exception as e:
        print(f"  Failed to load {model_diff.label_b}: {e}")
        tracker.failure(f"Load data + model: {model_diff.label_b}", e)
        return False

    # PDP comparisons for all common features
    print("\n[2/3] Comparing PDP for all common features...")
    print("  This may take a few minutes depending on dataset size...")
    pdp_start = time.time()
    try:
        model_diff.compare_all_pdp(
            analyzer_a=model_analyzer_a,
            analyzer_b=model_analyzer_b,
            n_curves=1000,
            mode=plotting_mode,
        )
        tracker.success("PDP comparisons (all common features)")
    except Exception as e:
        print(f"  Error during PDP comparison: {e}")
        tracker.failure("PDP comparisons (all common features)", e)
    pdp_elapsed = time.time() - pdp_start
    print(f"  PDP comparisons finished in {pdp_elapsed:.2f} seconds")

    # Prediction comparison
    print("\n[3/3] Comparing predictions...")
    y_true = None
    if target_column:
        if target_column in model_analyzer_a.df.columns:
            y_true = model_analyzer_a.df[target_column].values
            print(f"  Using target column '{target_column}' for agreement analysis")
        else:
            print(f"  Warning: target column '{target_column}' not found in data. "
                  f"Skipping agreement matrix.")

    try:
        pred_stats = model_diff.compare_predictions(
            analyzer_a=model_analyzer_a,
            analyzer_b=model_analyzer_b,
            y_true=y_true,
            n_samples=None,
        )
        print("  Generated: prediction_scatter.png")
        print("  Generated: prediction_diff_histogram.png")
        print("  Generated: prediction_correlation_summary.txt")
        print("  Generated: score_qq_plot.png")
        if y_true is not None:
            print("  Generated: prediction_agreement_matrix.png")
        tracker.success("Prediction comparison")
    except Exception as e:
        print(f"  Error during prediction comparison: {e}")
        tracker.failure("Prediction comparison", e)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PART 2 COMPLETE - Data-dependent comparison finished in "
          f"{elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print("=" * 70)

    return True


def generate_summary_report(output_dir, data_analysis_done):
    """Count and list all generated files."""
    print("\n" + "=" * 70)
    print("MODEL DIFF SUMMARY REPORT")
    print("=" * 70)

    if not os.path.exists(output_dir):
        print(f"\nOutput directory not found: {output_dir}")
        return

    all_pngs = []
    all_txts = []
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), output_dir)
            if f.endswith(".png"):
                all_pngs.append(rel)
            elif f.endswith(".txt"):
                all_txts.append(rel)

    # Only count diff-related files (prefixed with diff_ or containing
    # prediction/importance/cumulative/pdp/qq)
    diff_pngs = [f for f in all_pngs if any(
        kw in f for kw in [
            "diff_", "cumulative_gain_comparison", "importance_scatter",
            "pdp_comparison", "prediction_", "score_qq", "agreement",
        ]
    )]
    diff_txts = [f for f in all_txts if any(
        kw in f for kw in ["prediction_correlation", "diff_"]
    )]

    print(f"\nOutput Directory: {output_dir}")
    print(f"Total Diff Plots Generated: {len(diff_pngs)}")
    if diff_txts:
        print(f"Text Reports: {len(diff_txts)}")

    if diff_pngs:
        print("\nGenerated plots:")
        for p in sorted(diff_pngs):
            print(f"  - {p}")

    if diff_txts:
        print("\nGenerated reports:")
        for t in sorted(diff_txts):
            print(f"  - {t}")

    if not data_analysis_done:
        print("\nNote: Data-dependent plots (PDP, predictions, Q-Q) were skipped.")
        print("  Re-run with a data directory for the full comparison.")

    print("\n" + "=" * 70)


def main():
    """Main function to run model diff CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two XGBoost models (JSON) using ModelDiff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tree-level comparison only (no data needed)
  python user_model_diff.py model_a.json model_b.json

  # Full comparison with data
  python user_model_diff.py model_a.json model_b.json data_directory/

  # Custom labels
  python user_model_diff.py model_a.json model_b.json data_directory/ \\
      --label-a "Baseline v3.6" --label-b "Candidate v3.7"

  # Specify target column for agreement matrix
  python user_model_diff.py model_a.json model_b.json data_directory/ \\
      --target-column target

  # Override output directory
  python user_model_diff.py model_a.json model_b.json --output-dir /tmp/diff_output/
        """,
    )

    parser.add_argument(
        "model_a_path",
        help="Path to the first (baseline) XGBoost model JSON file",
    )
    parser.add_argument(
        "model_b_path",
        help="Path to the second (candidate) XGBoost model JSON file",
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="(Optional) Directory containing parquet files for data-dependent analysis",
    )
    parser.add_argument(
        "--label-a",
        type=str,
        default="Model A",
        help='Display label for the first model (default: "Model A")',
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default="Model B",
        help='Display label for the second model (default: "Model B")',
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Target class index for multi-class models (default: None for binary/regression)",
    )
    parser.add_argument(
        "--plotting-mode",
        type=str,
        default="raw",
        choices=["raw", "probability", "logit"],
        help="Y-axis mode for PDP plots (default: raw)",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Name of the target column in data (for prediction agreement matrix)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: derived from model filenames)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_a_path):
        print(f"Error: Model A file not found: {args.model_a_path}")
        sys.exit(1)

    if not os.path.exists(args.model_b_path):
        print(f"Error: Model B file not found: {args.model_b_path}")
        sys.exit(1)

    if args.data_dir and not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        stem_a = os.path.splitext(os.path.basename(args.model_a_path))[0]
        stem_b = os.path.splitext(os.path.basename(args.model_b_path))[0]
        parent = os.path.dirname(os.path.abspath(args.model_a_path))
        output_dir = os.path.join(parent, f"model_diff_{stem_a}_vs_{stem_b}")

    os.makedirs(output_dir, exist_ok=True)

    # Print header
    total_start = time.time()
    print("\n" + "=" * 70)
    print("XGBoost Model Diff")
    print("=" * 70)
    print(f"{args.label_a}: {args.model_a_path}")
    print(f"{args.label_b}: {args.model_b_path}")
    if args.data_dir:
        print(f"Data:     {args.data_dir}")
    else:
        print(f"Data:     None (tree-level comparison only)")
    print(f"Output:   {output_dir}")
    print("=" * 70)

    # Initialize analysis tracker
    tracker = AnalysisTracker()

    # Initialize TreeAnalyzers and ModelDiff
    print("\nInitializing...")
    try:
        tree_analyzer_a = TreeAnalyzer(args.model_a_path, save_dir=output_dir)
        tree_analyzer_b = TreeAnalyzer(args.model_b_path, save_dir=output_dir)
    except Exception as e:
        print(f"Failed to initialize TreeAnalyzers: {e}")
        sys.exit(1)

    model_diff = ModelDiff(
        analyzer_a=tree_analyzer_a,
        analyzer_b=tree_analyzer_b,
        label_a=args.label_a,
        label_b=args.label_b,
        save_dir=output_dir,
    )

    # Part 1: Tree-level comparison (always runs)
    run_tree_level_comparison(model_diff, tracker=tracker)

    # Part 2: Data-dependent comparison (only if data provided)
    data_analysis_done = False
    if args.data_dir:
        data_analysis_done = run_data_dependent_comparison(
            model_diff=model_diff,
            tree_analyzer_a=tree_analyzer_a,
            tree_analyzer_b=tree_analyzer_b,
            model_a_path=args.model_a_path,
            model_b_path=args.model_b_path,
            data_dir=args.data_dir,
            target_class=args.target_class,
            plotting_mode=args.plotting_mode,
            target_column=args.target_column,
            tracker=tracker,
        )
    else:
        print("\n" + "=" * 70)
        print("SKIPPING DATA-DEPENDENT COMPARISON")
        print("=" * 70)
        print("No data directory provided. To run full comparison including:")
        print("  - Partial Dependence Plot comparisons")
        print("  - Prediction scatter, histogram, and correlations")
        print("  - Score Q-Q plot")
        print("  - Classification agreement matrix")
        print(f"\nRe-run with: python user_model_diff.py {args.model_a_path} "
              f"{args.model_b_path} DATA_DIR/")
        print("=" * 70)

    # Print failure/success summary
    tracker.print_summary()

    # Summary report
    generate_summary_report(output_dir, data_analysis_done)

    # Final message
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("MODEL DIFF COMPLETE!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {output_dir}/")
    print(f"Total execution time: {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

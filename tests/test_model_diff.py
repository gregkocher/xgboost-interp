"""
Unit tests for the ModelDiff class.

Uses the existing model_diffing_example.py to generate artifacts
(models, data, plots) once per session, then runs focused assertions.
"""

import os
import sys
import subprocess
import tempfile

import numpy as np
import pytest

# Add the package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost_interp import TreeAnalyzer, ModelAnalyzer
from xgboost_interp.core import ModelDiff
from xgboost_interp.examples.model_diffing_example import (
    ALL_FEATURES,
    MODEL_A_DROPS,
    MODEL_B_DROPS,
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
)

# Timeout in seconds (10 minutes)
TIMEOUT_SECONDS = 600


# ---------------------------------------------------------------------------
# Session-scoped fixture: run the example once, reuse artifacts everywhere
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model_diff_env(tmp_path_factory):
    """
    Run model_diffing_example.py once and return a namespace with:
    - model_diff: ModelDiff instance
    - analyzer_a / analyzer_b: ModelAnalyzer instances (data + model loaded)
    - output_dir: path to generated plots
    - pred_stats: dict returned by compare_predictions
    """
    tmpdir = str(tmp_path_factory.mktemp("model_diff"))

    # Run the full example script
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "xgboost_interp", "examples", "model_diffing_example.py",
    )
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=tmpdir,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"model_diffing_example.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # Paths inside the temp directory (mirrors the example's OUTPUT_DIR layout)
    output_dir = os.path.join(tmpdir, "examples", "model_diff", "output")
    model_a_path = os.path.join(tmpdir, "examples", "model_diff", "model_a.json")
    model_b_path = os.path.join(tmpdir, "examples", "model_diff", "model_b.json")
    data_dir = os.path.join(tmpdir, "examples", "model_diff", "data")

    # Reconstruct analyzers from saved artifacts
    tree_analyzer_a = TreeAnalyzer(model_a_path, save_dir=output_dir)
    tree_analyzer_b = TreeAnalyzer(model_b_path, save_dir=output_dir)

    model_diff = ModelDiff(
        analyzer_a=tree_analyzer_a,
        analyzer_b=tree_analyzer_b,
        label_a="Model A",
        label_b="Model B",
        save_dir=output_dir,
    )

    model_analyzer_a = ModelAnalyzer(tree_analyzer_a, target_class=1)
    model_analyzer_a.load_data_from_parquets(data_dir, num_files_to_read=1)
    model_analyzer_a.load_xgb_model(model_a_path)

    model_analyzer_b = ModelAnalyzer(tree_analyzer_b, target_class=1)
    model_analyzer_b.load_data_from_parquets(data_dir, num_files_to_read=1)
    model_analyzer_b.load_xgb_model(model_b_path)

    # Run compare_predictions to get the stats dict
    y_true = model_analyzer_a.df["target"].values
    pred_stats = model_diff.compare_predictions(
        analyzer_a=model_analyzer_a,
        analyzer_b=model_analyzer_b,
        y_true=y_true,
        n_samples=None,
    )

    return {
        "model_diff": model_diff,
        "analyzer_a": model_analyzer_a,
        "analyzer_b": model_analyzer_b,
        "output_dir": output_dir,
        "pred_stats": pred_stats,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFindFeatureChanges:
    def test_find_feature_changes(self, model_diff_env):
        """find_feature_changes returns the correct added/dropped/shared sets."""
        md = model_diff_env["model_diff"]
        changes = md.find_feature_changes()

        # Features that Model B has but Model A doesn't = MODEL_A_DROPS minus MODEL_B_DROPS
        expected_new_in_b = sorted(
            set(ALL_FEATURES) - set(FEATURES_MODEL_A) - (set(ALL_FEATURES) - set(FEATURES_MODEL_B))
        )
        # Simpler: new_in_b = features in B but not in A
        expected_new_in_b = sorted(set(FEATURES_MODEL_B) - set(FEATURES_MODEL_A))
        expected_dropped = sorted(set(FEATURES_MODEL_A) - set(FEATURES_MODEL_B))
        expected_in_both = sorted(set(FEATURES_MODEL_A) & set(FEATURES_MODEL_B))

        assert changes["new_in_b"] == expected_new_in_b
        assert changes["dropped_in_b"] == expected_dropped
        assert changes["in_both"] == expected_in_both


class TestBuildQQPercentiles:
    def test_build_qq_percentiles(self):
        """_build_qq_percentiles(5000) includes tail points, is sorted, length 103."""
        pcts = ModelDiff._build_qq_percentiles(5000)

        assert len(pcts) == 103
        assert 0.1 in pcts
        assert 0.5 in pcts
        assert 99.5 in pcts
        assert 99.9 in pcts
        assert pcts.min() > 0
        assert pcts.max() < 100
        # Monotonically increasing
        assert np.all(np.diff(pcts) > 0)


class TestComparePredictions:
    def test_compare_predictions_stats(self, model_diff_env):
        """compare_predictions returns valid correlation and difference stats."""
        stats = model_diff_env["pred_stats"]

        # Required keys
        for key in ["pearson_r", "spearman_r", "kendall_tau", "diff_mean", "diff_std"]:
            assert key in stats, f"Missing key: {key}"

        # Correlations must be in [-1, 1]
        assert -1 <= stats["pearson_r"] <= 1
        assert -1 <= stats["spearman_r"] <= 1
        assert -1 <= stats["kendall_tau"] <= 1

        # Standard deviation must be non-negative
        assert stats["diff_std"] >= 0

    def test_compare_predictions_generates_plots(self, model_diff_env):
        """compare_predictions creates the expected output PNGs."""
        output_dir = model_diff_env["output_dir"]

        expected_files = [
            "prediction_scatter.png",
            "prediction_diff_histogram.png",
            "prediction_agreement_matrix.png",
            "score_qq_plot.png",
        ]
        for fname in expected_files:
            fpath = os.path.join(output_dir, fname)
            assert os.path.isfile(fpath), f"Missing output file: {fname}"
            assert os.path.getsize(fpath) > 0, f"Empty output file: {fname}"


class TestCumulativeGain:
    def test_compare_cumulative_gain_generates_plot(self, model_diff_env):
        """compare_cumulative_gain creates its output PNG."""
        output_dir = model_diff_env["output_dir"]
        fpath = os.path.join(output_dir, "cumulative_gain_comparison.png")
        assert os.path.isfile(fpath), "Missing cumulative_gain_comparison.png"
        assert os.path.getsize(fpath) > 0, "Empty cumulative_gain_comparison.png"


class TestImportanceScatter:
    def test_plot_importance_scatter_generates_plots(self, model_diff_env):
        """plot_all_importance_scatters creates 3 scatter PNGs."""
        output_dir = model_diff_env["output_dir"]

        expected_files = [
            "importance_scatter_gain.png",
            "importance_scatter_weight.png",
            "importance_scatter_cover.png",
        ]
        for fname in expected_files:
            fpath = os.path.join(output_dir, fname)
            assert os.path.isfile(fpath), f"Missing output file: {fname}"
            assert os.path.getsize(fpath) > 0, f"Empty output file: {fname}"

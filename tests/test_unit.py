"""
Fast unit tests for xgboost-interp.

These tests train tiny models (10 trees, < 1 second each) and verify
core functionality without running full example scripts.

Total runtime: < 30 seconds.
"""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_classification
import xgboost as xgb

from xgboost_interp import TreeAnalyzer, ModelAnalyzer, ModelDiff
from xgboost_interp.utils import AnalysisTracker


# ---------------------------------------------------------------------------
# Fixtures: tiny models trained once per session (< 2 seconds total)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def iris_env(tmp_path_factory):
    """Train a tiny 10-tree iris model for testing."""
    tmpdir = str(tmp_path_factory.mktemp("iris"))

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)

    model_path = os.path.join(tmpdir, "iris_test.json")
    model.save_model(model_path)

    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir, exist_ok=True)

    return {
        "model_path": model_path,
        "output_dir": output_dir,
        "X": X,
        "y": y,
        "feature_names": list(iris.feature_names),
    }


@pytest.fixture(scope="session")
def diff_env(tmp_path_factory):
    """Train two 10-tree models with overlapping but different features."""
    tmpdir = str(tmp_path_factory.mktemp("diff"))

    feature_names = [f"feat_{i}" for i in range(10)]
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    df = pd.DataFrame(X, columns=feature_names)

    # Model A uses feat_0..feat_7, Model B uses feat_2..feat_9
    names_a = feature_names[:8]
    names_b = feature_names[2:]

    model_a = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model_a.fit(df[names_a], y)
    path_a = os.path.join(tmpdir, "model_a.json")
    model_a.save_model(path_a)

    model_b = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model_b.fit(df[names_b], y)
    path_b = os.path.join(tmpdir, "model_b.json")
    model_b.save_model(path_b)

    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir, exist_ok=True)

    return {
        "model_a_path": path_a,
        "model_b_path": path_b,
        "output_dir": output_dir,
        "names_a": names_a,
        "names_b": names_b,
    }


# ---------------------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_package(self):
        from xgboost_interp import TreeAnalyzer, ModelAnalyzer, ModelDiff
        assert TreeAnalyzer is not None
        assert ModelAnalyzer is not None
        assert ModelDiff is not None

    def test_import_core(self):
        from xgboost_interp.core import TreeAnalyzer, ModelAnalyzer, ModelDiff
        assert TreeAnalyzer is not None

    def test_import_plotting(self):
        from xgboost_interp.plotting import (
            BasePlotter, FeaturePlotter, TreePlotter, InteractivePlotter,
        )
        assert BasePlotter is not None

    def test_import_utils(self):
        from xgboost_interp.utils import (
            ModelLoader, DataLoader, compute_model_metrics, AnalysisTracker,
        )
        assert AnalysisTracker is not None


# ---------------------------------------------------------------------------
# 2. AnalysisTracker tests
# ---------------------------------------------------------------------------

class TestAnalysisTracker:
    def test_success_and_failure(self):
        t = AnalysisTracker()
        t.success("step1")
        t.failure("step2", "some error")
        t.success("step3")

        assert len(t.results) == 3
        failures = [(n, e) for n, ok, e in t.results if not ok]
        successes = [n for n, ok, _ in t.results if ok]
        assert len(failures) == 1
        assert len(successes) == 2
        assert failures[0] == ("step2", "some error")

    def test_print_summary(self, capsys):
        t = AnalysisTracker()
        t.success("ok_step")
        t.failure("bad_step", "boom")
        t.print_summary()

        captured = capsys.readouterr()
        assert "Total:     2" in captured.out
        assert "Succeeded: 1" in captured.out
        assert "Failed:    1" in captured.out
        assert "bad_step: boom" in captured.out


# ---------------------------------------------------------------------------
# 3. TreeAnalyzer tests (using iris fixture)
# ---------------------------------------------------------------------------

class TestTreeAnalyzer:
    def test_load_model(self, iris_env):
        ta = TreeAnalyzer(iris_env["model_path"], save_dir=iris_env["output_dir"])
        assert ta.feature_names is not None
        assert len(ta.feature_names) == 4  # iris has 4 features

    def test_tree_count(self, iris_env):
        ta = TreeAnalyzer(iris_env["model_path"], save_dir=iris_env["output_dir"])
        # 10 estimators * 3 classes = 30 trees for multi-class
        assert len(ta.trees) == 30

    def test_feature_importance_plot(self, iris_env):
        ta = TreeAnalyzer(iris_env["model_path"], save_dir=iris_env["output_dir"])
        ta.plot_feature_importance_combined()
        fpath = os.path.join(iris_env["output_dir"], "feature_importance_combined.png")
        assert os.path.isfile(fpath)
        assert os.path.getsize(fpath) > 0

    def test_cumulative_gain_plot(self, iris_env):
        ta = TreeAnalyzer(iris_env["model_path"], save_dir=iris_env["output_dir"])
        ta.plot_cumulative_gain()
        fpath = os.path.join(iris_env["output_dir"], "cumulative_gain.png")
        assert os.path.isfile(fpath)
        assert os.path.getsize(fpath) > 0

    def test_analyze_feature_freeze(self, iris_env):
        ta = TreeAnalyzer(iris_env["model_path"], save_dir=iris_env["output_dir"])
        result = ta.analyze_feature_freeze("petal length (cm)", 2.5, verbose=False)

        # Check all expected keys are present
        expected_keys = {
            "feature_name", "value",
            "total_leaves", "reachable_leaves", "frac_leaves_reachable",
            "total_splits", "reachable_splits", "frozen_splits", "active_splits",
            "frac_splits_reachable", "frac_splits_still_deciding",
            "total_cover", "reachable_cover", "frac_cover_reachable",
        }
        assert set(result.keys()) == expected_keys

        # Fractions should be in [0, 1]
        assert 0.0 <= result["frac_leaves_reachable"] <= 1.0
        assert 0.0 <= result["frac_splits_reachable"] <= 1.0
        assert 0.0 <= result["frac_splits_still_deciding"] <= 1.0
        assert 0.0 <= result["frac_cover_reachable"] <= 1.0

        # Reachable counts should not exceed totals
        assert result["reachable_leaves"] <= result["total_leaves"]
        assert result["reachable_splits"] <= result["total_splits"]

        # Frozen + active should equal reachable splits
        assert result["frozen_splits"] + result["active_splits"] == result["reachable_splits"]

        # Totals should be positive for a real model
        assert result["total_leaves"] > 0
        assert result["total_splits"] > 0

    def test_analyze_feature_freeze_invalid_feature(self, iris_env):
        ta = TreeAnalyzer(iris_env["model_path"], save_dir=iris_env["output_dir"])
        with pytest.raises(ValueError, match="not found in model"):
            ta.analyze_feature_freeze("nonexistent_feature", 1.0)

    def test_analyze_feature_freeze_verbose(self, iris_env, capsys):
        ta = TreeAnalyzer(iris_env["model_path"], save_dir=iris_env["output_dir"])
        ta.analyze_feature_freeze("petal width (cm)", 1.0, verbose=True)
        captured = capsys.readouterr()
        assert "Feature Freeze Analysis" in captured.out
        assert "petal width (cm)" in captured.out
        assert "Reachable leaves" in captured.out
        assert "Reachable cover" in captured.out


# ---------------------------------------------------------------------------
# 4. ModelDiff tests (using diff fixture)
# ---------------------------------------------------------------------------

class TestModelDiff:
    def test_find_feature_changes(self, diff_env):
        ta_a = TreeAnalyzer(diff_env["model_a_path"], save_dir=diff_env["output_dir"])
        ta_b = TreeAnalyzer(diff_env["model_b_path"], save_dir=diff_env["output_dir"])
        md = ModelDiff(
            analyzer_a=ta_a, analyzer_b=ta_b,
            label_a="A", label_b="B",
            save_dir=diff_env["output_dir"],
        )
        changes = md.find_feature_changes()

        expected_new = sorted(set(diff_env["names_b"]) - set(diff_env["names_a"]))
        expected_dropped = sorted(set(diff_env["names_a"]) - set(diff_env["names_b"]))
        expected_shared = sorted(set(diff_env["names_a"]) & set(diff_env["names_b"]))

        assert changes["new_in_b"] == expected_new
        assert changes["dropped_in_b"] == expected_dropped
        assert changes["in_both"] == expected_shared

    def test_build_qq_percentiles_large(self):
        pcts = ModelDiff._build_qq_percentiles(5000)
        assert len(pcts) == 103
        assert 0.1 in pcts
        assert 99.9 in pcts
        assert np.all(np.diff(pcts) > 0)

    def test_build_qq_percentiles_small(self):
        pcts = ModelDiff._build_qq_percentiles(500)
        assert len(pcts) == 99
        assert 0.1 not in pcts  # no tail points for small samples
        assert np.all(np.diff(pcts) > 0)

    def test_compare_cumulative_gain_plot(self, diff_env):
        ta_a = TreeAnalyzer(diff_env["model_a_path"], save_dir=diff_env["output_dir"])
        ta_b = TreeAnalyzer(diff_env["model_b_path"], save_dir=diff_env["output_dir"])
        md = ModelDiff(
            analyzer_a=ta_a, analyzer_b=ta_b,
            label_a="A", label_b="B",
            save_dir=diff_env["output_dir"],
        )
        md.compare_cumulative_gain()
        fpath = os.path.join(diff_env["output_dir"], "cumulative_gain_comparison.png")
        assert os.path.isfile(fpath)
        assert os.path.getsize(fpath) > 0

    def test_importance_scatter_plots(self, diff_env):
        ta_a = TreeAnalyzer(diff_env["model_a_path"], save_dir=diff_env["output_dir"])
        ta_b = TreeAnalyzer(diff_env["model_b_path"], save_dir=diff_env["output_dir"])
        md = ModelDiff(
            analyzer_a=ta_a, analyzer_b=ta_b,
            label_a="A", label_b="B",
            save_dir=diff_env["output_dir"],
        )
        md.plot_all_importance_scatters()
        for metric in ["gain", "weight", "cover"]:
            fpath = os.path.join(diff_env["output_dir"], f"importance_scatter_{metric}.png")
            assert os.path.isfile(fpath), f"Missing importance_scatter_{metric}.png"
            assert os.path.getsize(fpath) > 0

    def test_print_summary(self, diff_env, capsys):
        ta_a = TreeAnalyzer(diff_env["model_a_path"], save_dir=diff_env["output_dir"])
        ta_b = TreeAnalyzer(diff_env["model_b_path"], save_dir=diff_env["output_dir"])
        md = ModelDiff(
            analyzer_a=ta_a, analyzer_b=ta_b,
            label_a="A", label_b="B",
            save_dir=diff_env["output_dir"],
        )
        md.print_summary()
        captured = capsys.readouterr()
        assert len(captured.out) > 0  # should print something

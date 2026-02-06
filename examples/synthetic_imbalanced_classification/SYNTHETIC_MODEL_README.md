# Synthetic Imbalanced Classification Example

This example generates synthetic data for binary classification with known feature-target relationships, allowing validation of the interpretability tools.

## Dataset Overview

- **Samples**: 50,000
- **Positive Rate**: ~10% (imbalanced)
- **Random Seed**: 10 (for reproducibility)
- **Total Features**: 39

## Feature Groups

### 1. Normal IID Features (9)

Independent normally distributed features with varying effects on the target.

| Feature Name | Mean | Std | Effect |
|--------------|------|-----|--------|
| `norm_iid_pos_strong` | 0 | 1 | +0.8 (strong positive) |
| `norm_iid_pos_medium` | 0 | 1 | +0.5 (medium positive) |
| `norm_iid_pos_weak` | 0 | 1 | +0.2 (weak positive) |
| `norm_iid_neg_strong` | 0 | 1 | -0.8 (strong negative) |
| `norm_iid_neg_medium` | 0 | 1 | -0.5 (medium negative) |
| `norm_iid_neg_weak` | 0 | 1 | -0.2 (weak negative) |
| `norm_iid_zero_1` | 0 | 1 | 0 (no effect) |
| `norm_iid_zero_2` | 2 | 0.5 | 0 (no effect) |
| `norm_iid_zero_3` | -1 | 2 | 0 (no effect) |

### 2. Correlated Normal Features (7)

Normally distributed features with correlation structure (adjacent features have ρ ≈ 0.6).

| Feature Name | Effect |
|--------------|--------|
| `norm_corr_1_pos` | +0.5 (medium positive) |
| `norm_corr_2_pos` | +0.2 (weak positive) |
| `norm_corr_3_neg` | -0.5 (medium negative) |
| `norm_corr_4_neg` | -0.2 (weak negative) |
| `norm_corr_5_zero` | 0 (no effect) |
| `norm_corr_6_zero` | 0 (no effect) |
| `norm_corr_7_zero` | 0 (no effect) |

### 3. Categorical Features (8)

Each category has a specific effect on positive label probability.

| Feature Name | Cardinality | Effect Pattern |
|--------------|-------------|----------------|
| `cat_15_strong` | 15 | Linear from -0.8 to +0.8 |
| `cat_30_medium` | 30 | Linear from -0.5 to +0.5 |
| `cat_50_weak` | 50 | Linear from -0.2 to +0.2 |
| `cat_75_mixed` | 75 | Random effects per category |
| `cat_100_strong` | 100 | Linear from -0.8 to +0.8 |
| `cat_150_medium` | 150 | Linear from -0.5 to +0.5 |
| `cat_180_weak` | 180 | Linear from -0.2 to +0.2 |
| `cat_200_none` | 200 | No effect |

### 4. Binary Features (4)

Binary (0/1) features with 50% base rate.

| Feature Name | Effect |
|--------------|--------|
| `bin_pos_strong` | +0.8 when =1 |
| `bin_pos_weak` | +0.2 when =1 |
| `bin_neg_strong` | -0.8 when =1 |
| `bin_neg_weak` | -0.2 when =1 |

### 5. Uniform Features (6)

Uniformly distributed with linear or nonlinear relationships.

| Feature Name | Range | Relationship |
|--------------|-------|--------------|
| `unif_linear_pos` | [0, 1] | Linear positive (+0.5) |
| `unif_linear_neg` | [0, 1] | Linear negative (-0.5) |
| `unif_quad_pos` | [0, 1] | Quadratic U-shaped (+1.5, very strong) |
| `unif_quad_neg` | [0, 1] | Quadratic inverted-U (-1.5, very strong) |
| `unif_none_1` | [0, 10] | No effect |
| `unif_none_2` | [-5, 5] | No effect |

**Quadratic relationships**: 
- `unif_quad_pos`: Both extremes (0 and 1) increase P(y=1), center (0.5) decreases it (U-shaped)
- `unif_quad_neg`: Center increases P(y=1), both extremes decrease it (inverted-U)

### 6. Trigonometric Features (2)

Periodic features with sinusoidal relationships to the target.

| Feature Name | Range | Relationship |
|--------------|-------|--------------|
| `trig_sin_pos_x` | [0, 2π] | sin(3x) with effect +0.8 |
| `trig_cos_pos_x` | [0, 2π] | cos(3x) with effect +0.8 |

These features complete **3 full cycles** in their range, creating periodic wave patterns:
- At peaks of sin/cos, P(y=1) increases
- At troughs of sin/cos, P(y=1) decreases
- Tests whether XGBoost can learn non-monotonic, periodic relationships

### 7. Noise Features (3)

Pure noise with no signal - these should have minimal feature importance.

| Feature Name | Distribution |
|--------------|--------------|
| `noise_norm` | Normal(0, 1) |
| `noise_unif` | Uniform(0, 100) |
| `noise_cat` | Categorical (50 levels) |

## Target Generation

1. Compute log-odds as sum of all feature contributions
2. Adjust intercept to achieve ~10% positive rate
3. Apply sigmoid to get P(y=1)
4. Sample binary target from Bernoulli(P(y=1))

## Expected Results

When analyzing this model:

- **High importance**: Features with "strong" in the name, quadratic features (very strong effect)
- **Medium importance**: Features with "medium" in the name, trigonometric features
- **Low importance**: Features with "weak" in the name
- **Near-zero importance**: Features with "zero", "none", or "noise" in the name
- **Correlated features**: May share importance due to substitutability
- **Categorical features**: Should show step patterns in PDP/ALE
- **Quadratic features**: Should show U-shaped or inverted-U relationships (very obvious due to strong effect)
- **Trigonometric features**: Should show periodic wave patterns with 3 cycles in PDP/ALE

## Running the Example

```bash
cd /path/to/xgboost-interp
uv run python xgboost_interp/examples/synthetic_imbalanced_classification_example.py
```

Outputs are saved to `examples/synthetic_imbalanced_classification/output/`.


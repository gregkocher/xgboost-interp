"""
Criteo Conversion Prediction Example
=====================================

This example demonstrates XGBoost interpretability analysis on the Criteo
Conversion Logs dataset - a real-world dataset from display advertising.

Dataset: Criteo Conversion Logs
- Source: Criteo Labs (http://labs.criteo.com/2013/12/conversion-logs-dataset/)
- Task: Predict whether an ad impression leads to a conversion
- Features: 39 features (13 numerical + 26 categorical)
- Size: ~1GB compressed, ~16 million records
- Domain: Display advertising / retargeting

This is a realistic dataset with:
- High cardinality categorical features
- Missing values (NULL entries)
- Class imbalance (low conversion rate)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost_interp import TreeAnalyzer, ModelAnalyzer

def download_instructions():
    """Print instructions for downloading the dataset."""
    print("\n" + "=" * 70)
    print("CRITEO CONVERSION LOGS DATASET")
    print("=" * 70)
    print("\n📥 To download the dataset:")
    print("\n  1. Go to: http://labs.criteo.com/2013/12/conversion-logs-dataset/")
    print("  2. Accept the research terms")
    print("  3. Download 'dac_sample.tar.gz' (~1GB)")
    print("  4. Extract it to get 'dac_sample.txt'")
    print("  5. Place 'dac_sample.txt' in the current directory")
    print("  6. Run this script again")
    print("\n" + "=" * 70)

def load_criteo_data(filepath='dac_sample.txt', nrows=1000000):
    """
    Load Criteo dataset.
    
    Args:
        filepath: Path to the Criteo data file
        nrows: Number of rows to load (default 1M for demo)
    
    Returns:
        DataFrame with loaded data
    """
    print("\n" + "=" * 70)
    print("CRITEO CONVERSION PREDICTION - XGBOOST INTERPRETABILITY")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        print(f"\n❌ Data file not found: {filepath}")
        download_instructions()
        return None
    
    print(f"\n✅ Found data file: {filepath}")
    print(f"📦 Loading {nrows:,} records...")
    
    # Define column names
    # 13 integer features (I1-I13) + 26 categorical features (C1-C26) + 1 label
    int_cols = [f'I{i}' for i in range(1, 14)]
    cat_cols = [f'C{i}' for i in range(1, 27)]
    cols = ['Label'] + int_cols + cat_cols
    
    # Load data
    df = pd.read_csv(
        filepath,
        sep='\t',
        names=cols,
        nrows=nrows,
        na_values=['', 'NULL']
    )
    
    print(f"\n📊 Loaded {len(df):,} records")
    print(f"📊 Shape: {df.shape}")
    print(f"📊 Features: {len(int_cols)} numerical + {len(cat_cols)} categorical = {len(cols)-1} total")
    
    return df

def prepare_features(df, sample_frac=0.2):
    """
    Prepare features for XGBoost training.
    
    Args:
        df: Raw dataframe
        sample_frac: Fraction of data to use for faster training
    
    Returns:
        X, y, feature_names
    """
    print("\n" + "=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)
    
    # Sample data for faster training
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"\n📊 Using {sample_frac*100}% sample: {len(df):,} records")
    
    print(f"\n📊 Conversion rate: {df['Label'].mean():.4f} ({df['Label'].mean()*100:.2f}%)")
    print(f"📊 Positive samples: {df['Label'].sum():,}")
    print(f"📊 Negative samples: {(df['Label']==0).sum():,}")
    
    # Separate features
    int_cols = [f'I{i}' for i in range(1, 14)]
    cat_cols = [f'C{i}' for i in range(1, 27)]
    
    # Handle missing values for numerical features (fill with -1)
    print(f"\n🔧 Handling missing values...")
    for col in int_cols:
        missing_pct = df[col].isna().sum() / len(df) * 100
        if missing_pct > 0:
            print(f"  {col}: {missing_pct:.1f}% missing")
        df[col] = df[col].fillna(-1).astype(int)
    
    # Encode categorical features
    print(f"\n🔄 Label encoding categorical features...")
    le_dict = {}
    
    for col in cat_cols:
        # Fill missing and convert to string
        df[col] = df[col].fillna('MISSING').astype(str)
        
        # Label encode
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
        
        n_unique = len(le.classes_)
        print(f"  {col}: {n_unique:,} unique values")
    
    # Prepare X and y
    feature_cols = int_cols + cat_cols
    X = df[feature_cols]
    y = df['Label']
    
    print(f"\n✅ Features prepared: {X.shape}")
    
    return X, y, feature_cols

def train_xgboost_model(X, y, feature_names):
    """
    Train XGBoost model for conversion prediction.
    
    Args:
        X: Features
        y: Target (conversion label)
        feature_names: List of feature names
    
    Returns:
        model, X_test, y_test
    """
    print("\n" + "=" * 70)
    print("XGBOOST MODEL TRAINING")
    print("=" * 70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train set: {len(X_train):,} samples")
    print(f"📊 Test set: {len(X_test):,} samples")
    print(f"📊 Train conversion rate: {y_train.mean():.4f}")
    print(f"📊 Test conversion rate: {y_test.mean():.4f}")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Train XGBoost model optimized for conversion prediction
    print("\n🚀 Training XGBoost model...")
    print("Settings:")
    print(f"  - Trees: 1000")
    print(f"  - Max depth: 8")
    print(f"  - Learning rate: 0.1")
    print(f"  - Objective: binary:logistic")
    print(f"  - Scale pos weight: {scale_pos_weight:.2f} (for imbalanced data)")
    print(f"  - Early stopping: 50 rounds")
    
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.1,
        objective='binary:logistic',
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50,
        tree_method='hist'  # Faster for large datasets
    )
    
    print("\n⏳ Training in progress...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    # Evaluate
    from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model Performance:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Total trees: {len(model.get_booster().get_dump())}")
    
    return model, X_test, y_test

def save_model_and_data(model, X_test, y_test):
    """Save model as JSON and test data for analysis."""
    print("\n" + "=" * 70)
    print("SAVING MODEL AND DATA")
    print("=" * 70)
    
    # Save model as JSON
    model_path = 'criteo_conversion_xgb.json'
    model.save_model(model_path)
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n✅ Saved XGBoost model: {model_path} ({file_size:.1f} MB)")
    
    # Save test data for analysis
    os.makedirs('criteo_data', exist_ok=True)
    
    # Sample test data for faster analysis
    sample_size = min(10000, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    y_sample = y_test.iloc[:sample_size]
    
    df_sample = X_sample.copy()
    df_sample['Label'] = y_sample.values
    
    df_sample.to_parquet('criteo_data/criteo_test_sample.parquet', index=False)
    print(f"✅ Saved test data sample ({sample_size:,} records): criteo_data/criteo_test_sample.parquet")
    
    return model_path

def run_interpretability_analysis(model_path):
    """Run comprehensive interpretability analysis."""
    print("\n" + "=" * 70)
    print("INTERPRETABILITY ANALYSIS")
    print("=" * 70)
    
    # Initialize tree analyzer
    tree_analyzer = TreeAnalyzer(model_path, save_dir='criteo_conversion_xgb')
    
    # Print model summary
    print("\n📊 Model Summary:")
    tree_analyzer.print_model_summary()
    
    # Tree structure analysis
    print("\n🌳 Generating tree structure analysis...")
    tree_analyzer.plot_feature_importance_combined(top_n=30)
    tree_analyzer.plot_feature_importance_distributions(log_scale=True, top_n=20)
    tree_analyzer.plot_tree_depth_histogram()
    tree_analyzer.plot_cumulative_gain()
    
    # Feature co-occurrence analysis
    print("\n🔗 Generating feature co-occurrence analysis...")
    tree_analyzer.plot_tree_level_feature_cooccurrence()
    tree_analyzer.plot_path_level_feature_cooccurrence()
    
    print("\n✅ Tree structure analysis complete!")
    
    # Data-dependent analysis
    print("\n📈 Generating data-dependent analysis...")
    model_analyzer = ModelAnalyzer(tree_analyzer)
    
    # Load test data
    model_analyzer.load_data_from_parquets('criteo_data/')
    model_analyzer.load_xgb_model()
    
    # Get top features for detailed analysis
    importance_df = pd.DataFrame({
        'feature': tree_analyzer.feature_names,
        'weight': [tree_analyzer.feature_weight.get(f, 0) for f in tree_analyzer.feature_names],
        'gain': [tree_analyzer.feature_gain.get(f, 0) for f in tree_analyzer.feature_names]
    }).sort_values('gain', ascending=False)
    
    print("\n📊 Top 10 Features by Gain:")
    print(importance_df.head(10).to_string(index=False))
    
    # Analyze top numerical and categorical features
    top_numerical = [f for f in importance_df['feature'].head(10) if f.startswith('I')][:3]
    top_categorical = [f for f in importance_df['feature'].head(10) if f.startswith('C')][:2]
    
    top_features = top_numerical + top_categorical
    print(f"\n📊 Analyzing top features: {top_features}")
    
    # Generate PDP and marginal impact for top features
    for feature in top_features:
        try:
            print(f"\n  Analyzing {feature}...")
            model_analyzer.plot_partial_dependence(feature)
            model_analyzer.plot_marginal_impact_univariate(feature)
        except Exception as e:
            print(f"  ⚠️  Could not analyze {feature}: {e}")
    
    print("\n✅ Data-dependent analysis complete!")
    
    return tree_analyzer

def main():
    """Main execution function."""
    # Load data
    df = load_criteo_data(filepath='dac_sample.txt', nrows=1000000)
    
    if df is None:
        print("\n❌ Could not load data. Please download the dataset first.")
        return
    
    # Prepare features
    X, y, feature_names = prepare_features(df, sample_frac=0.2)
    
    # Train model
    model, X_test, y_test = train_xgboost_model(X, y, feature_names)
    
    # Save model and data
    model_path = save_model_and_data(model, X_test, y_test)
    
    # Run interpretability analysis
    analyzer = run_interpretability_analysis(model_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output directory: criteo_conversion_xgb/")
    print(f"📊 Model file: {model_path}")
    print(f"💾 Data file: criteo_data/criteo_test_sample.parquet")
    
    # Count output files
    import glob
    plots = glob.glob('criteo_conversion_xgb/*.png')
    print(f"\n✅ Generated {len(plots)} visualization files:")
    for plot in sorted(plots):
        print(f"  - {os.path.basename(plot)}")
    
    print("\n🎉 Criteo Conversion Example completed successfully!")
    print("\nThis demonstrates interpretability analysis on a real-world")
    print("advertising dataset with 39 features (13 numerical + 26 categorical)")
    print("and 1000 trees trained on conversion prediction.")

if __name__ == "__main__":
    main()


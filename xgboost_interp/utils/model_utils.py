"""
Utilities for loading and parsing XGBoost models.
"""

import json
import re
from typing import Optional, List, Dict, Any


class ModelLoader:
    """Utility class for loading and parsing XGBoost model JSON files."""
    
    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """Safely convert value to int, returning None if conversion fails."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    
    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert value to float, returning None if conversion fails."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    
    @staticmethod
    def _regex_int(text: str, pattern: str) -> Optional[int]:
        """Extract integer from text using regex pattern."""
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None
    
    @classmethod
    def load_model_json(cls, json_path: str) -> Dict[str, Any]:
        """
        Load XGBoost model from JSON file.
        
        Args:
            json_path: Path to the JSON model file
            
        Returns:
            Dictionary containing the parsed model JSON
        """
        with open(json_path, "r") as f:
            return json.load(f)
    
    @classmethod
    def extract_model_metadata(cls, model_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from XGBoost model JSON.
        
        Args:
            model_json: Parsed model JSON dictionary
            
        Returns:
            Dictionary containing extracted metadata
        """
        learner = model_json.get("learner", {})
        
        # Extract basic parameters
        num_trees_total = cls._safe_int(
            learner.get("gradient_booster", {})
            .get("model", {})
            .get("gbtree_model_param", {})
            .get("num_trees")
        )
        
        # Extract scikit-learn attributes
        skl_attr = learner.get("attributes", {}).get("scikit_learn", "")
        num_trees_outer = cls._regex_int(skl_attr, r'"n_estimators":\s*(\d+)')
        max_depth = cls._regex_int(skl_attr, r'"max_depth":\s*(\d+)')
        
        # Extract training parameters
        learning_rate = cls._safe_float(
            learner.get("learner_train_param", {}).get("learning_rate")
        )
        
        # Extract model parameters
        base_score = cls._safe_float(
            learner.get("learner_model_param", {}).get("base_score")
        )
        
        # Extract feature names
        feature_names = learner.get("feature_names", [])
        if not feature_names:
            num_feature = cls._safe_int(
                learner.get("learner_model_param", {}).get("num_feature")
            )
            if num_feature is not None:
                feature_names = [f"f{i}" for i in range(num_feature)]
        
        # Extract objective
        objective = learner.get("objective", "Unknown")
        
        return {
            "num_trees_total": num_trees_total,
            "num_trees_outer": num_trees_outer,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "base_score": base_score,
            "feature_names": feature_names,
            "objective": objective,
        }
    
    @classmethod
    def extract_trees(cls, model_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tree structures from model JSON.
        
        Args:
            model_json: Parsed model JSON dictionary
            
        Returns:
            List of tree dictionaries
        """
        return model_json.get("learner", {}).get("gradient_booster", {}).get("model", {}).get("trees", [])

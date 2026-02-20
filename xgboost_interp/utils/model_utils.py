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
    
    @staticmethod
    def _compute_tree_depth(left_children: List[int], right_children: List[int]) -> int:
        """Compute the depth of a single tree from its child arrays."""
        if not left_children:
            return 0
        from collections import deque
        max_depth = 0
        queue = deque([(0, 0)])  # (node_index, depth)
        while queue:
            node, depth = queue.popleft()
            if left_children[node] == -1:
                # Leaf node
                max_depth = max(max_depth, depth)
            else:
                left = left_children[node]
                right = right_children[node]
                if left < len(left_children):
                    queue.append((left, depth + 1))
                if right < len(right_children):
                    queue.append((right, depth + 1))
        return max_depth
    
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
        gb_model = (
            learner.get("gradient_booster", {})
            .get("model", {})
        )
        gbtree_param = gb_model.get("gbtree_model_param", {})
        learner_model_param = learner.get("learner_model_param", {})
        
        # Extract basic parameters
        num_trees_total = cls._safe_int(gbtree_param.get("num_trees"))
        
        # --- num_trees_outer ---
        # Try scikit-learn attribute first, then compute from tree counts
        skl_attr = learner.get("attributes", {}).get("scikit_learn", "")
        num_trees_outer = cls._regex_int(skl_attr, r'"n_estimators":\s*(\d+)')
        if num_trees_outer is None and num_trees_total is not None:
            num_parallel_tree = cls._safe_int(gbtree_param.get("num_parallel_tree")) or 1
            num_class = cls._safe_int(learner_model_param.get("num_class")) or 0
            # For multiclass (num_class >= 2) each boosting round has num_class trees
            class_multiplier = num_class if num_class >= 2 else 1
            num_trees_outer = num_trees_total // (class_multiplier * num_parallel_tree)
        
        # --- max_depth ---
        # Try scikit-learn attribute first, then compute from actual tree structures
        max_depth = cls._regex_int(skl_attr, r'"max_depth":\s*(\d+)')
        if max_depth is None:
            trees = gb_model.get("trees", [])
            if trees:
                max_depth = max(
                    cls._compute_tree_depth(
                        t.get("left_children", []),
                        t.get("right_children", [])
                    )
                    for t in trees
                )
        
        # --- learning_rate ---
        # Try learner_train_param, then check attributes for eta
        learning_rate = cls._safe_float(
            learner.get("learner_train_param", {}).get("learning_rate")
        )
        if learning_rate is None:
            learning_rate = cls._safe_float(
                learner.get("learner_train_param", {}).get("eta")
            )
        
        # Extract model parameters
        base_score = cls._safe_float(learner_model_param.get("base_score"))
        
        # Extract feature names
        feature_names = learner.get("feature_names", [])
        if not feature_names:
            num_feature = cls._safe_int(learner_model_param.get("num_feature"))
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

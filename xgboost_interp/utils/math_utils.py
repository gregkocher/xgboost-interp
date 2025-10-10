"""
Mathematical utility functions.
"""

import numpy as np
from typing import List


class MathUtils:
    """Mathematical utility functions for XGBoost analysis."""
    
    @staticmethod
    def normalize_vector(x: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(x)
        return x / norm if norm > 0 else x
    
    @staticmethod
    def compute_tree_depth(left_children: List[int], right_children: List[int], 
                          start_node: int = 0) -> int:
        """
        Compute the depth of a tree given its structure.
        
        Args:
            left_children: List of left child indices (-1 for leaf)
            right_children: List of right child indices (-1 for leaf)
            start_node: Starting node index (usually 0 for root)
            
        Returns:
            Maximum depth of the tree
        """
        def _compute_depth_recursive(node: int) -> int:
            if node == -1 or node >= len(left_children):
                return 0
            
            left_depth = _compute_depth_recursive(left_children[node])
            right_depth = _compute_depth_recursive(right_children[node])
            return 1 + max(left_depth, right_depth)
        
        return _compute_depth_recursive(start_node)
    
    @staticmethod
    def compute_stats(data_list: List[float]) -> dict:
        """
        Compute basic statistics for a list of values.
        
        Args:
            data_list: List of numerical values
            
        Returns:
            Dictionary with mean, median, std, min, max
        """
        if not data_list:
            return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
        
        arr = np.array(data_list)
        return {
            "mean": np.mean(arr),
            "median": np.median(arr),
            "std": np.std(arr),
            "min": np.min(arr),
            "max": np.max(arr)
        }

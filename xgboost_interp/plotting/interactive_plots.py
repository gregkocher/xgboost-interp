"""
Interactive plotting functionality using Plotly.
"""

from typing import List, Dict, Optional, Tuple
from .base_plotter import BasePlotter


class InteractivePlotter(BasePlotter):
    """Interactive plotting using Plotly for tree visualization."""
    
    def plot_interactive_trees(self, trees: List[Dict], feature_names: List[str],
                             top_k: int = 30, combined: bool = False,
                             vertical_spacing: int = 12) -> None:
        """
        Create interactive tree visualizations using Plotly.
        
        Args:
            trees: List of tree dictionaries
            feature_names: List of feature names
            top_k: Number of trees to visualize
            combined: Whether to show all trees in one plot or separate plots
            vertical_spacing: Vertical spacing between trees (for combined view)
        """
        try:
            import plotly.graph_objects as go
            import networkx as nx
        except ImportError:
            print("Plotly and NetworkX are required for interactive plots")
            return
        
        trees = trees[:top_k]
        
        if combined:
            self._plot_combined_trees(trees, feature_names, vertical_spacing)
        else:
            self._plot_separate_trees(trees, feature_names)
    
    def _build_tree_data(self, tree: Dict, feature_names: List[str], 
                        y_shift: float = 0) -> Tuple[List, List, List, List, List, List, List, List, int]:
        """
        Build graph data for visualizing a tree.
        
        Args:
            tree: Tree dictionary from XGBoost model
            feature_names: List of feature names
            y_shift: Vertical offset for positioning (used in combined view)
            
        Returns:
            tuple: (edge_x, edge_y, node_x, node_y, node_text, node_colors, hover_colors, node_sizes, max_depth)
        """
        import networkx as nx
        
        G = nx.DiGraph()
        split_indices = tree.get("split_indices", [])
        base_weights = tree.get("base_weights", [])
        loss_changes = tree.get("loss_changes", [])
        split_conditions = tree.get("split_conditions", [])
        sum_hessians = tree.get("sum_hessian", [])  # Cover data
        lefts = tree.get("left_children", [])
        rights = tree.get("right_children", [])
        
        pos = {}
        labels = {}
        node_info = {}  # Store info for color/size computation
        max_depth = [0]
        
        def build_graph(node: int, depth: int, x_offset: float) -> None:
            if node == -1 or node >= len(split_indices):
                return
            
            max_depth[0] = max(max_depth[0], depth)
            
            # Get cover value
            cover = sum_hessians[node] if node < len(sum_hessians) else 0
            
            # Determine if leaf or split node
            is_leaf = lefts[node] == -1
            
            # Create node label
            if not is_leaf:  # Split node
                feat_name = (feature_names[split_indices[node]] 
                           if split_indices[node] < len(feature_names) 
                           else f"f{split_indices[node]}")
                gain = loss_changes[node] if node < len(loss_changes) else 0
                threshold = (split_conditions[node] 
                           if node < len(split_conditions) else 0)
                
                label = (f"<b>Feature:</b> {feat_name}<br>"
                        f"<b>Threshold:</b> &lt;{threshold:,.4f}<br>"
                        f"<b>Gain:</b> {gain:,.4f}<br>"
                        f"<b>Cover:</b> {cover:,.0f}")
                leaf_value = None
            else:  # Leaf node
                value = base_weights[node] if node < len(base_weights) else 0
                label = (f"<b>Δ Score:</b> {value:,.4f}<br>"
                        f"<b>Cover:</b> {cover:,.0f}")
                leaf_value = value
            
            pos[node] = (x_offset, -depth + y_shift)
            labels[node] = label
            node_info[node] = {'is_leaf': is_leaf, 'leaf_value': leaf_value, 'cover': cover}
            
            # Add edges and recurse
            if node < len(lefts) and lefts[node] != -1:
                G.add_edge(node, lefts[node])
                build_graph(lefts[node], depth + 1, x_offset - 1 / (2 ** depth))
            
            if node < len(rights) and rights[node] != -1:
                G.add_edge(node, rights[node])
                build_graph(rights[node], depth + 1, x_offset + 1 / (2 ** depth))
        
        build_graph(0, 0, 0)
        
        # For size normalization
        cover_values = [info['cover'] for info in node_info.values()]
        min_cover = min(cover_values) if cover_values else 0
        max_cover = max(cover_values) if cover_values else 1
        cover_range = max_cover - min_cover if max_cover > min_cover else 1
        
        # For leaf color normalization (gradient opacity)
        leaf_values = [info['leaf_value'] for info in node_info.values() if info['is_leaf'] and info['leaf_value'] is not None]
        max_abs_leaf = max(abs(v) for v in leaf_values) if leaf_values else 1
        if max_abs_leaf == 0:
            max_abs_leaf = 1  # Avoid division by zero
        
        # Size range in pixels
        min_size, max_size = 8, 25
        
        # Opacity range for gradient
        min_opacity, max_opacity = 0.2, 0.9
        
        # Color constants
        SPLIT_GRAY = '#808080'  # Gray for split nodes
        
        # Hover colors (categorical, fixed)
        HOVER_GRAY = '#e8e8e8'
        HOVER_GREEN = 'rgba(144, 238, 144, 0.9)'  # Light green
        HOVER_RED = 'rgba(255, 182, 182, 0.9)'    # Light red
        
        # Create edge arrays
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        
        # Create node arrays with colors and sizes
        node_x, node_y, node_text, node_colors, hover_colors, node_sizes = [], [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(labels[node])
            
            info = node_info[node]
            
            # Compute colors
            if info['is_leaf']:
                value = info['leaf_value']
                if value > 0:
                    # Dynamic opacity based on magnitude (per-tree normalized)
                    opacity = min_opacity + (max_opacity - min_opacity) * (abs(value) / max_abs_leaf)
                    marker_color = f'rgba(0, 200, 0, {opacity:.2f})'
                    hover_color = HOVER_GREEN
                elif value < 0:
                    # Dynamic opacity based on magnitude (per-tree normalized)
                    opacity = min_opacity + (max_opacity - min_opacity) * (abs(value) / max_abs_leaf)
                    marker_color = f'rgba(200, 0, 0, {opacity:.2f})'
                    hover_color = HOVER_RED
                else:  # value == 0
                    marker_color = 'rgba(128, 128, 128, 0.4)'
                    hover_color = HOVER_GRAY
            else:
                # Split node: gray
                marker_color = SPLIT_GRAY
                hover_color = HOVER_GRAY
            
            node_colors.append(marker_color)
            hover_colors.append(hover_color)
            
            # Compute size based on cover
            normalized_cover = (info['cover'] - min_cover) / cover_range
            size = min_size + (max_size - min_size) * normalized_cover
            node_sizes.append(size)
        
        return edge_x, edge_y, node_x, node_y, node_text, node_colors, hover_colors, node_sizes, max_depth[0]
    
    def _plot_separate_trees(self, trees: List[Dict], feature_names: List[str]) -> None:
        """Plot each tree in a separate interactive plot."""
        for t_idx, tree in enumerate(trees):
            fig = self._create_tree_plot(tree, feature_names, t_idx)
            fig.show()
    
    def _plot_combined_trees(self, trees: List[Dict], feature_names: List[str],
                           vertical_spacing: int) -> None:
        """Plot all trees in a single interactive plot."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        y_shift = 0
        
        for t_idx, tree in enumerate(trees):
            self._add_tree_to_figure(fig, tree, feature_names, t_idx, y_shift)
            y_shift -= vertical_spacing
        
        fig.update_layout(
            title=f"Interactive View of {len(trees)} Trees (Zoom + Scroll Enabled)",
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, 
                      scaleanchor="x", scaleratio=1),
            height=max(600, len(trees) * vertical_spacing * 10)
        )
        
        fig.show()
    
    def _create_tree_plot(self, tree: Dict, feature_names: List[str], tree_idx: int):
        """Create a single tree plot."""
        import plotly.graph_objects as go
        
        edge_x, edge_y, node_x, node_y, node_text, node_colors, hover_colors, node_sizes, max_depth = self._build_tree_data(
            tree, feature_names
        )
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes with dynamic colors, sizes, and hover backgrounds
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='black')),
            hovertext=node_text,
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor=hover_colors,
                font=dict(color='black'),
                bordercolor='black'
            ),
            showlegend=False
        ))
        
        # Create y-axis tick labels (Depth 0, Depth 1, ...)
        y_tickvals = [-d for d in range(max_depth + 1)]
        y_ticktext = [f"Depth {d}" for d in range(max_depth + 1)]
        
        fig.update_layout(
            title=f"Tree {tree_idx}",
            showlegend=False,
            margin=dict(l=60, r=10, t=30, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, 
                      tickmode='array', tickvals=y_tickvals, ticktext=y_ticktext),
            height=600
        )
        
        return fig
    
    def _add_tree_to_figure(self, fig, tree: Dict, feature_names: List[str],
                          tree_idx: int, y_shift: float) -> None:
        """Add a single tree to an existing figure."""
        import plotly.graph_objects as go
        
        edge_x, edge_y, node_x, node_y, node_text, node_colors, hover_colors, node_sizes, _ = self._build_tree_data(
            tree, feature_names, y_shift
        )
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes with dynamic colors, sizes, and hover backgrounds
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='black')),
            hoverinfo='text',
            hovertext=node_text,
            hoverlabel=dict(
                bgcolor=hover_colors,
                font=dict(color='black'),
                bordercolor='black'
            ),
            showlegend=False
        ))

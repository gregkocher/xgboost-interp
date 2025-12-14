"""
Interactive plotting functionality using Plotly.
"""

from typing import List, Dict, Optional
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
            print("⚠️ Plotly and NetworkX are required for interactive plots")
            return
        
        trees = trees[:top_k]
        
        if combined:
            self._plot_combined_trees(trees, feature_names, vertical_spacing)
        else:
            self._plot_separate_trees(trees, feature_names)
    
    def _plot_separate_trees(self, trees: List[Dict], feature_names: List[str]) -> None:
        """Plot each tree in a separate interactive plot."""
        import plotly.graph_objects as go
        import networkx as nx
        
        for t_idx, tree in enumerate(trees):
            fig = self._create_tree_plot(tree, feature_names, t_idx)
            fig.show()
    
    def _plot_combined_trees(self, trees: List[Dict], feature_names: List[str],
                           vertical_spacing: int) -> None:
        """Plot all trees in a single interactive plot."""
        import plotly.graph_objects as go
        import networkx as nx
        
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
            height=max(600, len(trees) * vertical_spacing * 10),
            hoverlabel=dict(bgcolor="white", font_size=10)
        )
        
        fig.show()
    
    def _create_tree_plot(self, tree: Dict, feature_names: List[str], tree_idx: int):
        """Create a single tree plot."""
        import plotly.graph_objects as go
        import networkx as nx
        
        G = nx.DiGraph()
        split_indices = tree.get("split_indices", [])
        base_weights = tree.get("base_weights", [])
        loss_changes = tree.get("loss_changes", [])
        split_conditions = tree.get("split_conditions", [])
        lefts = tree.get("left_children", [])
        rights = tree.get("right_children", [])
        
        pos = {}
        labels = {}
        max_depth = [0]  # Track max depth for y-axis labels
        
        def build_graph(node: int, depth: int, x_offset: float) -> None:
            if node == -1 or node >= len(split_indices):
                return
            
            max_depth[0] = max(max_depth[0], depth)
            
            # Create node label
            if node < len(split_indices) and lefts[node] != -1:  # Split node
                feat_name = (feature_names[split_indices[node]] 
                           if split_indices[node] < len(feature_names) 
                           else f"f{split_indices[node]}")
                gain = loss_changes[node] if node < len(loss_changes) else 0
                threshold = (split_conditions[node] 
                           if node < len(split_conditions) else 0)
                
                label = (f"Feature: {feat_name}<br>"
                        f"Threshold: &lt;{threshold:.4f}<br>"
                        f"Gain: {gain:.4f}")
            else:  # Leaf node
                value = base_weights[node] if node < len(base_weights) else 0
                label = f"<b>Leaf</b><br>Δ Score: {value:.4f}"
            
            pos[node] = (x_offset, -depth)
            labels[node] = label
            
            # Add edges and recurse
            if node < len(lefts) and lefts[node] != -1:
                G.add_edge(node, lefts[node])
                build_graph(lefts[node], depth + 1, x_offset - 1 / (2 ** depth))
            
            if node < len(rights) and rights[node] != -1:
                G.add_edge(node, rights[node])
                build_graph(rights[node], depth + 1, x_offset + 1 / (2 ** depth))
        
        build_graph(0, 0, 0)
        
        # Create edges
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        
        # Create nodes
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(labels[node])
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=10, color='lightblue'),
            hovertext=node_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        # Create y-axis tick labels (Depth 0, Depth 1, ...)
        y_tickvals = [-d for d in range(max_depth[0] + 1)]
        y_ticktext = [f"Depth {d}" for d in range(max_depth[0] + 1)]
        
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
        import networkx as nx
        
        G = nx.DiGraph()
        split_indices = tree.get("split_indices", [])
        base_weights = tree.get("base_weights", [])
        loss_changes = tree.get("loss_changes", [])
        split_conditions = tree.get("split_conditions", [])
        lefts = tree.get("left_children", [])
        rights = tree.get("right_children", [])
        
        pos = {}
        labels = {}
        
        def build_graph(node: int, depth: int, x_offset: float) -> None:
            if node == -1 or node >= len(split_indices):
                return
            
            # Create node label (similar to _create_tree_plot)
            if node < len(split_indices) and lefts[node] != -1:  # Split node
                feat_name = (feature_names[split_indices[node]] 
                           if split_indices[node] < len(feature_names) 
                           else f"f{split_indices[node]}")
                gain = loss_changes[node] if node < len(loss_changes) else 0
                threshold = (split_conditions[node] 
                           if node < len(split_conditions) else 0)
                
                label = (f"Feature: {feat_name}<br>"
                        f"Threshold: &lt;{threshold:.4f}<br>"
                        f"Gain: {gain:.4f}")
            else:  # Leaf node
                value = base_weights[node] if node < len(base_weights) else 0
                label = f"<b>Leaf</b><br>Δ Score: {value:.4f}"
            
            pos[node] = (x_offset, -depth + y_shift)
            labels[node] = label
            
            # Add edges and recurse
            if node < len(lefts) and lefts[node] != -1:
                G.add_edge(node, lefts[node])
                build_graph(lefts[node], depth + 1, x_offset - 1 / (2 ** depth))
            
            if node < len(rights) and rights[node] != -1:
                G.add_edge(node, rights[node])
                build_graph(rights[node], depth + 1, x_offset + 1 / (2 ** depth))
        
        build_graph(0, 0, 0)
        
        # Add edges to figure
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes to figure
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(labels[node])
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=10, color='lightblue'),
            hoverinfo='text',
            hovertext=node_text,
            showlegend=False
        ))

"""
Grid-based graph display widget showing multiple visualizations simultaneously.
"""
from PyQt6.QtWidgets import QWidget, QGridLayout, QScrollArea, QVBoxLayout, QPushButton, QLabel, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import theme
from utils import visualize_architecture
from visual_style import (
    apply_professional_style, style_axes, style_line_plot, 
    style_scatter_plot, style_bar_plot, add_value_labels,
    PROFESSIONAL_COLORS, GRAPH_COLORS
)


class GraphGridWidget(QWidget):
    """Grid widget displaying multiple graphs simultaneously."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.graphs = {}
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        # Make widget resizable
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Scroll area for responsiveness
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Main container
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(12)
        self.grid_layout.setContentsMargins(15, 15, 15, 15)
        container.setLayout(self.grid_layout)
        
        scroll.setWidget(container)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)
        main_layout.addWidget(scroll, 1)  # Stretch factor for resizing
        
        # Export all button
        export_btn = QPushButton("Export All Graphs")
        export_btn.clicked.connect(self.export_all)
        export_btn.setMaximumHeight(40)
        main_layout.addWidget(export_btn)
        
        self.setLayout(main_layout)
        self.container = container
        
    def create_graph_widget(self, title):
        """Create a graph widget with title."""
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {theme.DARK_THEME['primary']}; margin-bottom: 3px; padding: 3px;")
        layout.addWidget(title_label)
        
        # Figure - responsive sizing
        fig = Figure(figsize=(7, 5))
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(320)  # Minimum height
        canvas.setMinimumWidth(400)   # Minimum width
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(canvas, 1)  # Stretch factor for resizing
        
        widget.setLayout(layout)
        return widget, fig, canvas
        
    def update_graphs(self, model, history, dataset, device):
        """Update all graphs with current data."""
        # Clear existing graphs
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)
        
        self.graphs = {}
        
        if model is None or history is None or len(history) == 0:
            return
            
        # Apply professional style
        apply_professional_style()
        
        # 1. Loss Curves - Professional styled
        widget, fig, canvas = self.create_graph_widget("Loss Curves")
        best_idx = np.argmin([h[1] for h in history])
        train_losses = history[best_idx][2]
        val_losses = history[best_idx][3]
        epochs = np.array(range(1, len(train_losses) + 1))
        
        ax = fig.add_subplot(111)
        style_axes(ax, "Best Model Convergence", "Epoch", "Loss (Log Scale)")
        
        # Styled line plots with gradients
        style_line_plot(ax, epochs, train_losses, 'Training Loss', 
                       GRAPH_COLORS['line_primary'], linewidth=3, 
                       marker='o', markersize=6, gradient=True)
        style_line_plot(ax, epochs, val_losses, 'Validation Loss', 
                       GRAPH_COLORS['line_secondary'], linewidth=3, 
                       marker='s', markersize=6, gradient=True)
        
        ax.set_yscale('log')
        ax.legend(loc='upper right', framealpha=0.95)
        fig.patch.set_facecolor(PROFESSIONAL_COLORS['background'])
        canvas.draw()
        self.grid_layout.addWidget(widget, 0, 0)
        self.graphs['loss'] = (fig, canvas)
        
        # 2. Architecture - make it larger with modern styling
        widget, fig, canvas = self.create_graph_widget("Network Architecture")
        fig.set_size_inches(14, 8)  # Larger figure for new design
        ax = fig.add_subplot(111)
        from architecture_viz import visualize_architecture_modern
        visualize_architecture_modern(model, ax=ax, figsize=(14, 8))
        fig.patch.set_facecolor(PROFESSIONAL_COLORS['background'])
        canvas.draw()
        # Span 2 columns for architecture
        self.grid_layout.addWidget(widget, 0, 1, 1, 2)  # row, col, rowspan, colspan
        self.graphs['architecture'] = (fig, canvas)
        
        # 3. Search Trajectory - Professional styled
        widget, fig, canvas = self.create_graph_widget("Search Trajectory")
        losses = np.array([h[1] for h in history])
        trials = np.array(range(1, len(losses) + 1))
        best_trial = np.argmin(losses) + 1
        
        ax = fig.add_subplot(111)
        style_axes(ax, "Hyperoptimization Progress", "Trial ID", "Validation MSE Loss")
        
        # Main trajectory line with gradient
        style_line_plot(ax, trials, losses, 'Validation Loss', 
                       GRAPH_COLORS['line_primary'], linewidth=3, 
                       marker='o', markersize=8, gradient=True)
        
        # Highlight best model
        ax.scatter([best_trial], [min(losses)], 
                  color=GRAPH_COLORS['best_model'], s=300, 
                  marker='*', zorder=5, label='Best Model',
                  edgecolors=PROFESSIONAL_COLORS['text_primary'],
                  linewidths=2)
        
        ax.legend(loc='upper right', framealpha=0.95)
        fig.patch.set_facecolor(PROFESSIONAL_COLORS['background'])
        canvas.draw()
        self.grid_layout.addWidget(widget, 1, 0)
        self.graphs['trajectory'] = (fig, canvas)
        
        # 4. Prediction Accuracy - Professional styled
        if dataset:
            from utils import calculate_metrics
            metrics, preds, actuals = calculate_metrics(model, dataset, device)
            widget, fig, canvas = self.create_graph_widget("Prediction Accuracy")
            subset_idx = np.random.choice(len(actuals), min(500, len(actuals)), replace=False)
            
            ax = fig.add_subplot(111)
            r2 = metrics['Root1_Real']['R2']
            style_axes(ax, f"Root 1 Real (R² = {r2:.4f})", "Ground Truth", "AI Prediction")
            
            # Professional scatter plot
            style_scatter_plot(ax, actuals[subset_idx, 0], preds[subset_idx, 0],
                             GRAPH_COLORS['scatter_primary'], s=40, alpha=0.7)
            
            # Perfect fit line
            min_val, max_val = actuals[:, 0].min(), actuals[:, 0].max()
            ax.plot([min_val, max_val], [min_val, max_val], 
                   color=GRAPH_COLORS['line_accent'], linewidth=3, 
                   linestyle='--', label='Perfect Fit', zorder=2)
            
            ax.legend(loc='upper left', framealpha=0.95)
            fig.patch.set_facecolor(PROFESSIONAL_COLORS['background'])
            canvas.draw()
            self.grid_layout.addWidget(widget, 1, 1)
            self.graphs['accuracy'] = (fig, canvas)
        
        # 5. Topology Comparison - Professional styled
        widget, fig, canvas = self.create_graph_widget("Topology Comparison")
        # Reconstruct topology data
        topology_data = []
        for i, (config_dict, val_loss, _, _) in enumerate(history):
            hidden_layers = config_dict.get('hidden_layers', 3)
            hidden_units = config_dict.get('hidden_units', 64)
            input_dim = 3
            output_dim = 4
            params = (input_dim * hidden_units + hidden_units * hidden_units * (hidden_layers - 1) + 
                     hidden_units * output_dim + hidden_units * hidden_layers + output_dim)
            topology_data.append({
                'params': params,
                'val_loss': val_loss
            })
        
        if topology_data:
            params = np.array([t['params'] for t in topology_data])
            val_losses = np.array([t['val_loss'] for t in topology_data])
            best_idx = np.argmin(val_losses)
            
            ax = fig.add_subplot(111)
            style_axes(ax, "Model Size vs Performance", "Total Parameters (approx)", "Validation Loss")
            
            # Color code: best model in green, others in blue
            colors = [GRAPH_COLORS['best_model'] if i == best_idx else GRAPH_COLORS['scatter_primary'] 
                     for i in range(len(params))]
            
            # Professional scatter with edge colors
            for i, (p, v, c) in enumerate(zip(params, val_losses, colors)):
                edge_color = PROFESSIONAL_COLORS['text_primary'] if i == best_idx else PROFESSIONAL_COLORS['border']
                ax.scatter(p, v, c=c, s=120, alpha=0.8, 
                          edgecolors=edge_color, linewidths=2, zorder=3)
            
            fig.patch.set_facecolor(PROFESSIONAL_COLORS['background'])
            canvas.draw()
            self.grid_layout.addWidget(widget, 1, 2)
            self.graphs['topology'] = (fig, canvas)
        
        # 6. Benchmark Results - Professional styled
        widget, fig, canvas = self.create_graph_widget("Benchmark Results")
        from utils import run_benchmark
        benchmark = run_benchmark(model, dataset, device=device)
        methods = ['Traditional\nCalculation', 'AI\nInference']
        means = [benchmark['trad_mean'], benchmark['ai_mean']]
        stds = [benchmark['trad_std'], benchmark['ai_std']]
        
        ax = fig.add_subplot(111)
        style_axes(ax, f"Inference Latency (Speedup: {benchmark['speedup']:.2f}×)", 
                  None, "Time (ms)")
        
        # Professional bar chart
        x_pos = np.arange(len(methods))
        bar_colors = [GRAPH_COLORS['bar_accent'], GRAPH_COLORS['bar_primary']]
        bars = style_bar_plot(ax, x_pos, means, bar_colors, width=0.6, alpha=0.9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        
        # Error bars
        ax.errorbar(methods, means, yerr=stds, fmt='none', 
                   color=PROFESSIONAL_COLORS['text_primary'], 
                   capsize=12, capthick=2.5, elinewidth=2.5, zorder=4)
        
        # Value labels
        add_value_labels(ax, bars, means, '{:.2f}ms')
        
        # Speedup annotation
        ax.text(0.5, 0.95, f"Speedup: {benchmark['speedup']:.2f}×", 
               ha='center', va='top', transform=ax.transAxes,
               fontsize=14, weight='bold', 
               color=GRAPH_COLORS['best_model'],
               bbox=dict(boxstyle='round', facecolor=PROFESSIONAL_COLORS['surface_light'], 
                        edgecolor=GRAPH_COLORS['best_model'], linewidth=2))
        
        fig.patch.set_facecolor(PROFESSIONAL_COLORS['background'])
        canvas.draw()
        self.grid_layout.addWidget(widget, 2, 0, 1, 3)  # Span full width on row 2
        self.graphs['benchmark'] = (fig, canvas)
        
    def export_all(self):
        """Export all graphs to files."""
        from PyQt6.QtWidgets import QFileDialog
        import os
        
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Graphs")
        if directory:
            for name, (fig, canvas) in self.graphs.items():
                filename = os.path.join(directory, f"{name}_graph.png")
                fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor=theme.DARK_THEME['background'])
            
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export Complete", f"All graphs exported to {directory}")


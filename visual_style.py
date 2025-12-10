"""
Professional visual styling for graphs and UI elements.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np

# Professional color palette with high contrast
PROFESSIONAL_COLORS = {
    'primary': '#0078D4',        # Microsoft blue
    'primary_dark': '#005A9E',   # Darker blue
    'primary_light': '#40A6FF',  # Light blue
    'secondary': '#00BCF2',      # Cyan
    'accent': '#FF6B35',         # Coral orange
    'success': '#107C10',        # Green
    'success_light': '#16C60C',  # Bright green
    'warning': '#FFB900',        # Amber
    'error': '#E81123',          # Red
    'text_primary': '#FFFFFF',   # White
    'text_secondary': '#D2D2D2', # Light gray
    'background': '#0D1117',     # GitHub dark
    'surface': '#161B22',        # Dark surface
    'surface_light': '#21262D',  # Lighter surface
    'border': '#30363D',         # Border gray
    'grid': '#21262D',           # Grid color
}

# Graph color schemes
GRAPH_COLORS = {
    'line_primary': '#0078D4',
    'line_secondary': '#00BCF2',
    'line_accent': '#FF6B35',
    'scatter_primary': '#0078D4',
    'scatter_secondary': '#00BCF2',
    'scatter_accent': '#FF6B35',
    'bar_primary': '#0078D4',
    'bar_secondary': '#00BCF2',
    'bar_accent': '#FF6B35',
    'highlight': '#FFB900',
    'best_model': '#16C60C',
}

def apply_professional_style():
    """Apply professional matplotlib styling."""
    # Set style parameters (only valid rcParams)
    rcParams.update({
        'figure.facecolor': PROFESSIONAL_COLORS['background'],
        'axes.facecolor': PROFESSIONAL_COLORS['surface'],
        'axes.edgecolor': PROFESSIONAL_COLORS['border'],
        'axes.labelcolor': PROFESSIONAL_COLORS['text_primary'],
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlepad': 10,
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': PROFESSIONAL_COLORS['grid'],
        'grid.linewidth': 1,
        'axes.axisbelow': True,
        'xtick.color': PROFESSIONAL_COLORS['text_secondary'],
        'ytick.color': PROFESSIONAL_COLORS['text_secondary'],
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.color': PROFESSIONAL_COLORS['text_primary'],
        'legend.facecolor': PROFESSIONAL_COLORS['surface_light'],
        'legend.edgecolor': PROFESSIONAL_COLORS['border'],
        'legend.framealpha': 0.9,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        'font.size': 11,
        'font.weight': 'normal',
    })

def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply professional styling to axes."""
    if title:
        ax.set_title(title, color=PROFESSIONAL_COLORS['text_primary'], 
                    fontsize=14, fontweight='bold', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=PROFESSIONAL_COLORS['text_primary'], fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=PROFESSIONAL_COLORS['text_primary'], fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PROFESSIONAL_COLORS['border'])
    ax.spines['bottom'].set_color(PROFESSIONAL_COLORS['border'])
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.tick_params(colors=PROFESSIONAL_COLORS['text_secondary'], 
                   labelsize=10, width=1.5, length=5)
    
    try:
        ax.set_facecolor(PROFESSIONAL_COLORS['surface'])
    except:
        pass

def create_gradient_fill(ax, x, y, color, alpha=0.3):
    """Create a gradient fill under a line."""
    ax.fill_between(x, y, alpha=alpha, color=color, interpolate=True)

def style_line_plot(ax, x, y, label=None, color=None, linewidth=2.5, 
                    marker=None, markersize=8, alpha=1.0, gradient=False):
    """Create a styled line plot."""
    if color is None:
        color = GRAPH_COLORS['line_primary']
    
    line = ax.plot(x, y, color=color, linewidth=linewidth, 
                   marker=marker, markersize=markersize, 
                   label=label, alpha=alpha, zorder=3)
    
    if gradient:
        create_gradient_fill(ax, x, y, color, alpha=0.2)
    
    return line

def style_scatter_plot(ax, x, y, color=None, s=50, alpha=0.6, edgecolors=None):
    """Create a styled scatter plot."""
    if color is None:
        color = GRAPH_COLORS['scatter_primary']
    if edgecolors is None:
        edgecolors = 'white'
    
    scatter = ax.scatter(x, y, c=color, s=s, alpha=alpha, 
                        edgecolors=edgecolors, linewidths=1.5, zorder=3)
    return scatter

def style_bar_plot(ax, x, heights, color=None, width=0.6, alpha=0.9, 
                   edgecolor=None, label=None):
    """Create a styled bar plot."""
    if color is None:
        color = GRAPH_COLORS['bar_primary']
    if edgecolor is None:
        edgecolor = PROFESSIONAL_COLORS['text_primary']
    
    # Handle single color or list of colors
    if isinstance(color, list) and len(color) == len(heights):
        bars = ax.bar(x, heights, width=width, color=color, alpha=alpha,
                      edgecolor=edgecolor, linewidth=2.5, label=label, zorder=3)
    else:
        bars = ax.bar(x, heights, width=width, color=color, alpha=alpha,
                      edgecolor=edgecolor, linewidth=2.5, label=label, zorder=3)
    return bars

def add_value_labels(ax, bars, values=None, format_str='{:.2f}'):
    """Add value labels on top of bars."""
    if values is None:
        values = [bar.get_height() for bar in bars]
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               format_str.format(value),
               ha='center', va='bottom', 
               color=PROFESSIONAL_COLORS['text_primary'],
               fontsize=10, fontweight='bold')


"""
Modern, high-end neural network architecture visualization.
Uses a layered box diagram approach with professional styling.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from visual_style import PROFESSIONAL_COLORS, apply_professional_style


def visualize_architecture_modern(model, ax=None, figsize=(14, 8)):
    """
    Create a modern, professional architecture visualization using layered boxes.
    
    Design:
    - Each layer is a vertical box/column
    - Neurons are represented as small squares within the layer box
    - Connections shown as subtle lines between layers
    - Clean, minimal design with high contrast
    - Layer information displayed clearly
    """
    apply_professional_style()
    
    config_dict = model.config
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=PROFESSIONAL_COLORS['background'])
    else:
        fig = ax.figure
        fig.patch.set_facecolor(PROFESSIONAL_COLORS['background'])
    
    # Layer information
    layers = [config_dict['input_dim']] + [config_dict['hidden_units']] * config_dict['hidden_layers'] + [config_dict['output_dim']]
    layer_names = ['Input\nLayer'] + [f'Hidden\nLayer {i + 1}' for i in range(config_dict['hidden_layers'])] + ['Output\nLayer']
    
    # Modern color scheme
    layer_colors = {
        'input': '#0078D4',      # Microsoft blue
        'hidden': '#00BCF2',     # Cyan
        'output': '#FF6B35',     # Coral orange
    }
    
    num_layers = len(layers)
    max_neurons = max(layers)
    
    # Calculate layout
    layer_width = 0.12
    layer_spacing = (0.95 - layer_width * num_layers) / (num_layers + 1)
    start_x = 0.02
    
    # Vertical layout
    box_top = 0.85
    box_bottom = 0.15
    box_height = box_top - box_bottom
    
    # Draw layers from left to right
    layer_centers = []
    
    for i, (n_neurons, layer_name) in enumerate(zip(layers, layer_names)):
        x_center = start_x + layer_spacing + i * (layer_width + layer_spacing) + layer_width / 2
        layer_centers.append(x_center)
        
        # Determine layer color
        if i == 0:
            color = layer_colors['input']
            border_color = '#40A6FF'  # Lighter blue border
        elif i == num_layers - 1:
            color = layer_colors['output']
            border_color = '#FF8C5A'  # Lighter orange border
        else:
            color = layer_colors['hidden']
            border_color = '#40D4FF'  # Lighter cyan border
        
        # Draw main layer box with rounded corners
        box = FancyBboxPatch(
            (x_center - layer_width/2, box_bottom),
            layer_width,
            box_height,
            boxstyle="round,pad=0.01",
            linewidth=3,
            edgecolor=border_color,
            facecolor=color,
            alpha=0.15,
            zorder=1
        )
        ax.add_patch(box)
        
        # Draw inner box (more opaque)
        inner_box = FancyBboxPatch(
            (x_center - layer_width/2 + 0.005, box_bottom + 0.01),
            layer_width - 0.01,
            box_height - 0.02,
            boxstyle="round,pad=0.005",
            linewidth=2,
            edgecolor=border_color,
            facecolor=color,
            alpha=0.25,
            zorder=2
        )
        ax.add_patch(inner_box)
        
        # Draw neurons as small squares within the layer
        neuron_size = min(0.015, box_height / (n_neurons + 2))
        neuron_spacing = box_height / (n_neurons + 1)
        
        neuron_positions = []
        for j in range(n_neurons):
            y_pos = box_bottom + (j + 1) * neuron_spacing
            
            # Neuron square with glow
            glow = mpatches.Rectangle(
                (x_center - neuron_size/2 - 0.002, y_pos - neuron_size/2 - 0.002),
                neuron_size + 0.004,
                neuron_size + 0.004,
                color=color,
                alpha=0.4,
                zorder=3
            )
            ax.add_patch(glow)
            
            # Main neuron square
            neuron = mpatches.Rectangle(
                (x_center - neuron_size/2, y_pos - neuron_size/2),
                neuron_size,
                neuron_size,
                color=color,
                edgecolor=PROFESSIONAL_COLORS['text_primary'],
                linewidth=1.5,
                zorder=4
            )
            ax.add_patch(neuron)
            
            # Inner highlight
            highlight = mpatches.Rectangle(
                (x_center - neuron_size/3, y_pos - neuron_size/3),
                neuron_size * 2/3,
                neuron_size * 2/3,
                color=PROFESSIONAL_COLORS['text_primary'],
                alpha=0.3,
                zorder=5
            )
            ax.add_patch(highlight)
            
            neuron_positions.append((x_center, y_pos))
        
        # Layer label above the box
        ax.text(x_center, box_top + 0.03, layer_name, 
               ha='center', va='bottom',
               fontsize=11, weight='bold',
               color=PROFESSIONAL_COLORS['text_primary'],
               family='sans-serif')
        
        # Neuron count below the box
        ax.text(x_center, box_bottom - 0.02, f'{n_neurons}',
               ha='center', va='top',
               fontsize=10,
               color=PROFESSIONAL_COLORS['text_secondary'],
               family='sans-serif')
    
    # Draw connections between layers (subtle)
    for i in range(num_layers - 1):
        x1 = layer_centers[i]
        x2 = layer_centers[i + 1]
        
        # Sample connections to avoid clutter
        n_connections = min(30, layers[i] * layers[i + 1])
        for _ in range(n_connections):
            src_idx = np.random.randint(0, layers[i])
            tgt_idx = np.random.randint(0, layers[i + 1])
            
            # Get neuron positions (approximate)
            y1 = box_bottom + (src_idx + 1) * (box_height / (layers[i] + 1))
            y2 = box_bottom + (tgt_idx + 1) * (box_height / (layers[i + 1] + 1))
            
            # Draw subtle connection line
            ax.plot([x1, x2], [y1, y2], 
                   color=PROFESSIONAL_COLORS['text_secondary'],
                   alpha=0.15, linewidth=0.8, zorder=0)
    
    # Title and info
    title_y = 0.95
    ax.text(0.5, title_y, 'Neural Network Architecture',
           ha='center', va='top',
           fontsize=18, weight='bold',
           color=PROFESSIONAL_COLORS['text_primary'],
           family='sans-serif')

    info_text = f"{config_dict['hidden_layers']} Hidden Layers × {config_dict['hidden_units']} Units | {model.count_parameters():,} Parameters"
    ax.text(0.5, title_y - 0.04, info_text,
           ha='center', va='top',
           fontsize=12,
           color=PROFESSIONAL_COLORS['text_secondary'],
           family='sans-serif')
    
    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Set background
    try:
        ax.set_facecolor(PROFESSIONAL_COLORS['background'])
    except:
        pass
    
    return fig


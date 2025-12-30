import matplotlib.pyplot as plt
import os

# Configuration for the visualization
OUTPUT_DIR = "./model_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_neural_net(ax, layer_sizes, layer_names, title):
    """
    Draws a schematic of a neural network on a given matplotlib axis.
    
    Args:
        ax: The matplotlib axis to draw on.
        layer_sizes: List of integers representing the number of neurons in each layer.
        layer_names: List of strings describing each layer.
        title: Title of the plot.
    """
    n_layers = len(layer_sizes)
    v_spacing = 1.0 / float(max(layer_sizes) if max(layer_sizes) < 10 else 10)
    h_spacing = 1.0 / float(n_layers)

    for i, n in enumerate(layer_sizes):
        layer_top_x = i * h_spacing + h_spacing / 2
        n_draw = n if n <= 8 else 8
        layer_top_y = 0.5 + (n_draw * v_spacing) / 2
        node_coords = []
        
        for m in range(n_draw):
            circle_x = layer_top_x
            circle_y = layer_top_y - m * v_spacing
            node_coords.append((circle_x, circle_y))
            
            color = 'skyblue' if i == 0 else ('lightgreen' if i == n_layers - 1 else 'white')
            if "Quantum" in layer_names[i]:
                color = 'violet'
            
            circle = plt.Circle((circle_x, circle_y), v_spacing/4,
                                color=color, ec='k', zorder=4)
            ax.add_artist(circle)
            
        if n > 8:
            ax.text(layer_top_x, layer_top_y - (n_draw/2)*v_spacing, "...", 
                    ha='center', va='center', fontsize=20, zorder=5)

        ax.text(layer_top_x, 1.05, layer_names[i], ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(layer_top_x, -0.05, f"{n} Units", ha='center', va='top', fontsize=10)

        if i > 0:
            prev_layer_x = (i - 1) * h_spacing + h_spacing / 2
            prev_n_draw = layer_sizes[i-1] if layer_sizes[i-1] <= 8 else 8
            prev_top_y = 0.5 + (prev_n_draw * v_spacing) / 2
            
            for m_curr in range(n_draw):
                curr_y = layer_top_y - m_curr * v_spacing
                for m_prev in range(prev_n_draw):
                    prev_y = prev_top_y - m_prev * v_spacing
                    line = plt.Line2D([prev_layer_x, layer_top_x], 
                                      [prev_y, curr_y], 
                                      c='gray', alpha=0.3, zorder=1)
                    ax.add_artist(line)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=40)

def visualize():
    n_qubits = 4
    layer_sizes = [32, 128, 128, n_qubits, n_qubits, n_qubits, n_qubits, 1]
    layer_names = ["Input (PCA)", "RELU", "RELU", "RELU", "RELU", "RELU", "RELU", "Output"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_neural_net(ax, layer_sizes, layer_names, "Classical Neural Network Architecture with PCA")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/classical_pca_architecture.png")
    print(f"Saved {OUTPUT_DIR}/classical_pca_architecture.png")
    plt.close()

if __name__ == "__main__":
    visualize()
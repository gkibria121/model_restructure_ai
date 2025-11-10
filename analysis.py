import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load model
model_path = 'best_efficientnet_b0_ai_vs_real.pt'
checkpoint = torch.load(model_path, map_location='cpu')

# Extract state dict
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint

# Set style
sns.set_style("whitegrid")

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# 1. Parameter distribution by layer
ax1 = plt.subplot(2, 3, 1)
layer_params = {}
for name, param in state_dict.items():
    layer_name = name.split('.')[0]
    if layer_name not in layer_params:
        layer_params[layer_name] = 0
    layer_params[layer_name] += param.numel()

sorted_layers = sorted(layer_params.items(), key=lambda x: x[1], reverse=True)[:10]
layers, counts = zip(*sorted_layers)
ax1.barh(layers, counts, color='steelblue')
ax1.set_xlabel('Number of Parameters')
ax1.set_title('Top 10 Layers by Parameter Count')
ax1.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))

# 2. Parameter type distribution
ax2 = plt.subplot(2, 3, 2)
param_types = {'weights': 0, 'biases': 0, 'bn_weight': 0, 'bn_bias': 0, 'other': 0}
for name, param in state_dict.items():
    if 'weight' in name and 'bn' not in name and 'norm' not in name:
        param_types['weights'] += param.numel()
    elif 'bias' in name and 'bn' not in name and 'norm' not in name:
        param_types['biases'] += param.numel()
    elif ('bn' in name or 'norm' in name) and 'weight' in name:
        param_types['bn_weight'] += param.numel()
    elif ('bn' in name or 'norm' in name) and 'bias' in name:
        param_types['bn_bias'] += param.numel()
    else:
        param_types['other'] += param.numel()

param_types = {k: v for k, v in param_types.items() if v > 0}
ax2.pie(param_types.values(), labels=param_types.keys(), autopct='%1.1f%%', startangle=90)
ax2.set_title('Parameter Distribution by Type')

# 3. Weight distribution histogram
ax3 = plt.subplot(2, 3, 3)
all_weights = []
for name, param in state_dict.items():
    if 'weight' in name and len(param.shape) > 1:  # Convolutional or linear weights
        all_weights.extend(param.flatten().cpu().numpy()[:10000])  # Sample for speed

ax3.hist(all_weights, bins=50, color='coral', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Weight Value')
ax3.set_ylabel('Frequency')
ax3.set_title('Weight Value Distribution')
ax3.set_yscale('log')

# 4. Layer depth vs parameter count
ax4 = plt.subplot(2, 3, 4)
layer_depths = []
layer_counts = []
for name, param in state_dict.items():
    depth = name.count('.')
    layer_depths.append(depth)
    layer_counts.append(param.numel())

ax4.scatter(layer_depths, layer_counts, alpha=0.6, c='green', s=50)
ax4.set_xlabel('Layer Depth (nesting level)')
ax4.set_ylabel('Parameter Count')
ax4.set_title('Layer Depth vs Parameter Count')
ax4.set_yscale('log')

# 5. Cumulative parameter distribution
ax5 = plt.subplot(2, 3, 5)
sorted_params = sorted([p.numel() for p in state_dict.values()], reverse=True)
cumsum = np.cumsum(sorted_params)
ax5.plot(range(len(cumsum)), cumsum / cumsum[-1] * 100, linewidth=2, color='purple')
ax5.set_xlabel('Number of Layers')
ax5.set_ylabel('Cumulative Percentage of Parameters (%)')
ax5.set_title('Cumulative Parameter Distribution')
ax5.grid(True, alpha=0.3)

# 6. Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

total_params = sum(p.numel() for p in state_dict.values())
total_layers = len(state_dict)
avg_params = total_params / total_layers
model_size_mb = total_params * 4 / (1024**2)

summary_text = f"""
MODEL SUMMARY
{'='*30}

Total Parameters: {total_params:,}
Total Layers: {total_layers}
Avg Params/Layer: {avg_params:,.0f}
Model Size: {model_size_mb:.2f} MB

Largest Layer: {max(state_dict.items(), key=lambda x: x[1].numel())[0].split('.')[0]}
Largest Param Count: {max(p.numel() for p in state_dict.values()):,}

Smallest Layer: {min(state_dict.items(), key=lambda x: x[1].numel())[0].split('.')[0]}
Smallest Param Count: {min(p.numel() for p in state_dict.values()):,}
"""

ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', 
         verticalalignment='center')

plt.tight_layout()
plt.savefig('densenet_parameter_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'densenet_parameter_analysis.png'")
plt.show()

# Print detailed statistics
print("\n" + "="*70)
print("DETAILED STATISTICS")
print("="*70)
print(f"Total Parameters: {total_params:,}")
print(f"Model Size: {model_size_mb:.2f} MB")
print(f"Number of Layers: {total_layers}")
print(f"Average Parameters per Layer: {avg_params:,.0f}")
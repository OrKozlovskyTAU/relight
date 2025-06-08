import os
import pickle
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from collections import defaultdict

pkl_path = "/home/dcor/orkozlovsky/repos/relight/models/controlnet/plotly_debug_latest.pkl"  # <-- Change this to your actual file
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

steps = data['steps_tracked']
grad_norms_per_layer = data['grad_norms_per_layer']
weight_norms_per_layer = data['weight_norms_per_layer']
layer_names = list(grad_norms_per_layer.keys())
print(layer_names)

def get_group_name(layer_name):
    parts = layer_name.split('.')
    if parts[0] == 'module':
        if parts[1].startswith('controlnet'):
            return 'controlnet'
        elif parts[1] in ['conv_in', 'down_blocks', 'mid_block']:
            return 'main_model'
        else:
            return None  # Exclude other modules
    return None

def plot_norms_by_group(norms_per_layer, steps, layer_names, title, html_path):
    # Filter out bias layers
    filtered_layer_names = [name for name in layer_names if not name.endswith('.bias')]
    groups = {'main_model': [], 'controlnet': []}
    for name in filtered_layer_names:
        group = get_group_name(name)
        if group in groups:
            groups[group].append(name)
    # Only keep non-empty groups
    groups = {k: v for k, v in groups.items() if v}
    rows = 1
    cols = 2
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'surface'}]*cols],
        subplot_titles=[k.replace('_', ' ').title() for k in groups.keys()]
    )
    for i, (group, names) in enumerate(groups.items()):
        row = 1
        col = i + 1
        norm_matrix = [norms_per_layer[n] for n in names]
        fig.add_trace(
            go.Surface(z=norm_matrix, x=steps, y=names, showscale=False),
            row=row, col=col
        )
        # Set axis titles for each subplot
        fig.update_scenes(
            dict(
                xaxis_title='Step',
                zaxis_title='Norm',
            ),
            row=row, col=col
        )
    fig.update_layout(
        title=title,
    )
    pio.write_html(fig, html_path)
    print(f"Saved plot to {html_path}")

# Gradients norm plot by group
print("Creating gradients norm plots by group...")
grad_html_path = os.path.splitext(pkl_path)[0] + '_gradients_norm_by_group.html'
plot_norms_by_group(
    grad_norms_per_layer, steps, layer_names,
    'Gradients Norm per Layer per Step (by Group)', grad_html_path
)

# Weights norm plot by group
print("Creating weights norm plots by group...")
weight_html_path = os.path.splitext(pkl_path)[0] + '_weights_norm_by_group.html'
plot_norms_by_group(
    weight_norms_per_layer, steps, layer_names,
    'Weights Norm per Layer per Step (by Group)', weight_html_path
) 
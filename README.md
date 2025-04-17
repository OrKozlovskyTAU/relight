# Relight

A Blender-based relighting tool that supports two main use cases:
1. Calculating and using transport matrices
2. Generating datasets of random light images

## Project Structure

```
relight/
├── core/               # Core functionality
│   ├── blender_utils.py    # Blender utility functions
│   ├── transport_matrix.py # Transport matrix functionality
│   └── random_light_dataset.py # Random light dataset generation
├── cli/                # Command-line interfaces
│   ├── transport_matrix.py  # CLI for transport matrix operations
│   └── random_light_dataset.py # CLI for random light dataset generation
└── main.py             # Main entry point
```

## Requirements

- Blender 3.0 or later
- Python packages (see requirements.txt):
  - numpy
  - OpenEXR
  - tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/relight.git
cd relight
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

### Transport Matrix Operations

#### Generate Transport Matrix
```bash
python run_relight.py transport generate --proj-resx 64 --proj-resy 64 --batch-size 100 --use-gpu
```

#### Calculate Transport Matrix
```bash
python run_relight.py transport calculate --proj-resx 64 --proj-resy 64 --batch-size 100 --use-multiprocessing
```

#### Load Transport Matrix
```bash
python run_relight.py transport load --input transport_matrix.npy
```

### Random Light Dataset Generation

```bash
python run_relight.py dataset generate --start-index 0 --n-images 100 --random-sphere --use-gpu
```

### Example Scripts

The project includes example scripts to demonstrate how to use the transport matrix and generate random light datasets:

#### Using the Transport Matrix
```bash
python examples/use_transport_matrix.py --input transport_matrix.npy --target-image target.png --output relit.png --proj-resx 64 --proj-resy 64 --use-gpu
```

#### Generating Random Light Dataset
```bash
python examples/generate_random_lights.py --start-index 0 --n-images 10 --random-sphere --use-gpu
```

## GPU Rendering

The project supports GPU rendering in Blender if available. To use GPU rendering, make sure your Blender installation has GPU support enabled and the appropriate drivers are installed.

Example commands with GPU rendering:
```bash
# Generate transport matrix with GPU rendering
python run_relight.py transport generate --proj-resx 64 --proj-resy 64 --batch-size 100 --use-gpu

# Generate random light images with GPU rendering
python run_relight.py dataset generate --start-index 0 --n-images 100 --random-sphere --use-gpu
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# MLP Benchmarking Framework

A flexible and comprehensive framework for benchmarking Multi-Layer Perceptron (MLP) models for digit recognition. This framework allows you to declaratively define experiments, run them in batches, and automatically generate detailed reports and visualizations.

## Features

- **Declarative Experiment Definition**: Define experiments using YAML configuration files
- **Batch Processing**: Run multiple experiments in a batch with a single command
- **Comprehensive Metrics**: Track and visualize training and test performance metrics
- **Detailed Reporting**: Generate individual and comparative reports automatically
- **Flexible Architecture**: Easily modify network architecture, data processing, and training parameters
- **Extensive Augmentation Options**: Apply various data augmentation techniques
- **Layer-Specific Activations**: Configure different activation functions for each layer

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- torchvision
- matplotlib
- numpy
- pandas
- seaborn
- PyYAML

### Install Dependencies

```bash
pip install torch torchvision matplotlib numpy pandas seaborn pyyaml
```

## Directory Structure

Recommended project structure:

```
mlp-benchmarking/
├── benchmark.py                # Main script
├── experiment_runner.py        # Experiment execution logic
├── batch_processor.py          # Batch processing logic
├── model_builder.py            # Model definition and construction
├── metrics_collector.py        # Metrics collection and storage
├── visualization.py            # Visualization utilities
├── report_generator.py         # Report generation utilities
├── experiments/                # Individual experiment YAML files
│   ├── baseline.yaml
│   ├── deeper_network.yaml
│   └── ...
├── batches/                    # Batch definition YAML files
│   ├── initial_tests.yaml
│   ├── learning_rate_sweep.yaml
│   └── ...
├── data-2/                     # Dataset directory
│   ├── train/
│   │   ├── 0/
│   │   ├── 1/
│   │   └── ...
│   └── test/
│       ├── 0/
│       ├── 1/
│       └── ...
└── results/                    # Results directory (will be created)
    ├── experiment_001/
    ├── batch_001/
    └── ...
```

## Usage

### Basic Commands

```bash
# Run a single experiment
python benchmark.py --config experiments/baseline.yaml --data-dir ./data-2

# Run a batch of experiments
python benchmark.py --batch batches/initial_tests.yaml --data-dir ./data-2

# Process all configurations in a directory
python benchmark.py --config-dir experiments/ --data-dir ./data-2

# Process all configurations recursively
python benchmark.py --config-dir experiments/ --recursive --data-dir ./data-2

# Generate reports from existing results
python benchmark.py --report-only --results-dir results/batch_001
```

### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to a single experiment configuration YAML file |
| `--batch` | Path to a batch configuration YAML file |
| `--config-dir` | Directory containing experiment configuration YAML files |
| `--report-only` | Generate reports from existing results without running experiments |
| `--output-dir` | Output directory for results (default: `./results`) |
| `--data-dir` | Directory containing the dataset (default: `./data-2`) |
| `--recursive` | Recursively process directories when using `--config-dir` |
| `--results-dir` | Directory containing results for report generation (required with `--report-only`) |

## Experiment Configuration Guide

Experiments are defined using YAML configuration files with the following structure:

```yaml
test_id: "experiment_001"
description: "Baseline MLP configuration"

# Network architecture parameters
architecture:
  hidden_layers: [512, 256]
  activations: ["relu", "relu"]  # Can be a string or a list
  dropout_rates: [0.5, 0.3]

# Data processing parameters
data:
  color_mode: "grayscale"  # "grayscale" or "rgb"
  batch_size: 128
  augmentation:
    random_affine: 
      enabled: true
      degrees: 10
      translate: [0, 0.1]
    random_rotation:
      enabled: false
      degrees: 15
    random_horizontal_flip:
      enabled: false
    color_jitter:  # RGB mode only
      enabled: false
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    random_erasing:
      enabled: false
      probability: 0.5
      scale: [0.02, 0.33]
      ratio: [0.3, 3.3]
    gaussian_blur:
      enabled: false
      kernel_size: 5
      sigma: [0.1, 2.0]
    random_crop:
      enabled: false
      size: [32, 32]
      padding: 4
    random_perspective:
      enabled: false
      distortion_scale: 0.5
      probability: 0.5
  normalization: [0.5, 0.5]  # [mean, std]

# Training configuration
training:
  optimizer:
    name: "adam"  # "adam", "sgd", or "rmsprop"
    learning_rate: 0.001
    weight_decay: 0.0
    # For SGD only
    # momentum: 0.9
  scheduler:
    name: "step_lr"  # "step_lr", "cosine", or "plateau"
    # For StepLR
    step_size: 10
    gamma: 0.7
    # For CosineAnnealingLR
    # t_max: 50
    # For ReduceLROnPlateau
    # patience: 3
    # factor: 0.1
  epochs: 15
  early_stopping:
    enabled: false
    patience: 5
```

### Required Fields

At minimum, a configuration file must have:
- `test_id`: A unique identifier for the experiment

All other parameters will default to reasonable values if not specified.

### Network Architecture Options

The `architecture` section supports:

- `hidden_layers`: List of integers for hidden layer sizes
- `activations`: String or list of strings for activation functions
  - Supported values: `"relu"`, `"leaky_relu"`, `"selu"`, `"tanh"`, `"sigmoid"`, `"elu"`, `"gelu"`
  - If a single string is provided, the same activation is used for all layers
  - If a list is provided, each element corresponds to a specific layer
- `dropout_rates`: List of dropout rates (0.0 means no dropout)

### Data Processing Options

The `data` section supports:

- `color_mode`: `"grayscale"` (1 channel) or `"rgb"` (3 channels)
- `batch_size`: Number of samples per batch
- Extensive augmentation options (see YAML example above)
- `normalization`: Mean and standard deviation for normalization

### Training Configuration Options

The `training` section supports:

- Various optimizer options with specific parameters
- Different learning rate schedulers
- Number of training epochs
- Optional early stopping

## Batch Configuration Guide

Batches of experiments are defined using YAML files with the following structure:

```yaml
batch_id: "learning_rate_sweep"
description: "Comparing different learning rates"
experiments:
  - "experiments/lr_0.0001.yaml"
  - "experiments/lr_0.001.yaml"
  - "experiments/lr_0.01.yaml"
  - "experiments/lr_0.1.yaml"
```

### Path Resolution

Paths in the `experiments` list can be:
- **Absolute paths**: Starting with `/` (Unix) or a drive letter (Windows)
- **Relative paths**: Resolved relative to the batch YAML file's location

For example, if your batch file is at `/home/user/mlp-benchmarking/batches/learning_rate_sweep.yaml`, a relative path like `../experiments/lr_0.001.yaml` will resolve to `/home/user/mlp-benchmarking/experiments/lr_0.001.yaml`.

## Example Configurations

### Basic MLP

```yaml
test_id: "basic_mlp"
description: "Basic MLP with default settings"

architecture:
  hidden_layers: [512, 256]
  activations: "relu"
  dropout_rates: [0.5, 0.3]

data:
  color_mode: "grayscale"
  batch_size: 128

training:
  optimizer:
    name: "adam"
    learning_rate: 0.001
  scheduler:
    name: "step_lr"
    step_size: 10
    gamma: 0.7
  epochs: 15
```

### Deep MLP with Mixed Activations

```yaml
test_id: "deep_mixed_mlp"
description: "Deep MLP with different activation functions"

architecture:
  hidden_layers: [1024, 512, 256, 128]
  activations: ["relu", "leaky_relu", "selu", "gelu"]
  dropout_rates: [0.5, 0.4, 0.3, 0.2]

data:
  color_mode: "rgb"
  batch_size: 64
  augmentation:
    random_affine: 
      enabled: true
      degrees: 15
      translate: [0.1, 0.1]
    random_rotation:
      enabled: true
      degrees: 10
    random_horizontal_flip:
      enabled: true
    color_jitter:
      enabled: true
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1

training:
  optimizer:
    name: "sgd"
    learning_rate: 0.01
    momentum: 0.9
    weight_decay: 0.0001
  scheduler:
    name: "cosine"
    t_max: 50
  epochs: 50
```

## Output Structure

The framework generates the following output structure:

```
results/
├── experiment_001/                # Individual experiment
│   ├── model.pt                   # Trained model weights
│   ├── training_metrics.csv       # Per-epoch training metrics
│   ├── test_metrics.csv           # Overall test metrics
│   ├── class_metrics.csv          # Per-class metrics
│   ├── confusion_matrix.npy       # Confusion matrix data
│   ├── experiment_001_report.md   # Experiment report
│   ├── training_curves.png        # Training visualization
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   └── class_accuracy.png         # Per-class accuracy visualization
│
└── batch_001/                     # Batch of experiments
    ├── individual/                # Individual experiment results
    │   ├── experiment_001/
    │   ├── experiment_002/
    │   └── ...
    ├── comparison/                # Comparative visualizations
    │   ├── batch_accuracy_comparison.png
    │   ├── batch_training_comparison.png
    │   ├── efficiency_comparison.png
    │   ├── class_accuracy_comparison.png
    │   └── param_effect_*.png     # Hyperparameter effect plots
    ├── all_experiments.csv        # Compiled results
    └── batch_comparison_report.md # Batch report
```

## Best Practices

1. **Organize related experiments**: Group related experiment files in subdirectories
2. **Use meaningful IDs**: Name your experiments to reflect their configuration
3. **Systematic parameter exploration**: When exploring a specific parameter, keep other parameters constant
4. **Start small**: Begin with small batches and shorter training times
5. **Use relative paths**: Use relative paths in batch files for better portability

## Example Workflow

1. Create baseline experiment:
   ```bash
   # Create baseline configuration
   nano experiments/baseline.yaml
   
   # Run baseline experiment
   python benchmark.py --config experiments/baseline.yaml
   ```

2. Create variations for a parameter sweep:
   ```bash
   # Create learning rate variations
   nano experiments/lr_sweep/lr_0.0001.yaml
   nano experiments/lr_sweep/lr_0.001.yaml
   nano experiments/lr_sweep/lr_0.01.yaml
   ```

3. Create a batch configuration:
   ```bash
   # Create batch file
   nano batches/lr_sweep.yaml
   ```

4. Run the batch:
   ```bash
   # Run batch of experiments
   python benchmark.py --batch batches/lr_sweep.yaml
   ```

5. Analyze the results:
   ```bash
   # Open the batch report
   open results/lr_sweep/batch_comparison_report.md
   ```

## License

This project is licensed under the MIT License.

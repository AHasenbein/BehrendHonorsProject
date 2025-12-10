# Quadratic Equation Predictor - Hyperoptimization GUI

A comprehensive PyQt6 application for hyperoptimizing AI-based quadratic equation predictors. This project explores how to optimize neural network architectures to achieve faster and more stable solutions to nonlinear mathematical problems.

## Features

- **Interactive GUI**: Full-featured PyQt6 interface for training and visualization
- **Real-time Training**: Watch training progress with live updating graphs
- **Equation Input**: Enter quadratic equations and get instant predictions
- **Hyperparameter Optimization**: Systematic tuning of learning rates, batch sizes, dropout rates, and network topology
- **Success Metrics**: Clear display of best configurations and success rates
- **Historical Analysis**: Load and compare previous training runs
- **Benchmarking**: Compare AI inference speed vs traditional calculation methods

## Installation

1. Install Python 3.8 or higher

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
python main.py
```

### GUI Components

#### Left Panel
- **Equation Input**: Enter coefficients (a, b, c) for quadratic equations
  - Click "Predict" to get instant root predictions
  - Click "Add to Dataset" to expand training data
  
- **Training Settings**: Configure hyperparameters
  - Learning Rate: Controls training speed
  - Batch Size: Number of samples per batch
  - Hidden Layers: Network depth
  - Hidden Units: Network width
  - Dropout Rate: Regularization parameter
  - Number of Trials: Hyperoptimization trials
  - Epochs per Trial: Training epochs per trial
  - Dataset Samples: Size of training dataset
  - Coefficient Range: Range for generating synthetic data

- **Success Metrics**: View best model performance
  - Best validation loss
  - Best configuration achieved
  - Trial-by-trial comparison table

#### Center Panel
- **Visualization Tabs**:
  - Loss Curves: Real-time training/validation loss
  - Architecture: Neural network structure visualization
  - Search Trajectory: Hyperoptimization progress
  - Prediction Accuracy: Scatter plots of predictions vs ground truth
  - Topology Comparison: Model size vs performance

- **Training Controls**:
  - Start Training: Begin hyperoptimization
  - Stop Training: Interrupt current training
  - Fine-tune Best Model: Further optimize the best model
  - Progress Bar: Training progress indicator

#### Right Panel
- **Historical Graphs**: Load and save previous training runs
- **Benchmark Results**: Performance comparison with traditional methods

### Menu Bar

- **File**:
  - Save Model: Save trained model to disk
  - Load Model: Load previously saved model
  - Save Results: Export training history as JSON
  - Load Results: Import previous training results
  - Exit: Close application

- **Help**:
  - About: Application information

## Project Structure

```
BehrendHonorsProject/
├── main.py              # Entry point
├── gui.py               # PyQt6 GUI application
├── model.py             # Neural network architecture
├── data.py              # Dataset generation
├── training.py          # Training and hyperoptimization
├── utils.py             # Visualization and metrics
├── config.py            # Configuration constants
└── requirements.txt     # Python dependencies
```

## Technical Details

### Neural Network Architecture
- Dynamic MLP (Multi-Layer Perceptron)
- Configurable depth (hidden layers) and width (hidden units)
- Dropout regularization for overfitting prevention
- ReLU activation functions

### Hyperoptimization
- Random search algorithm
- Searches across:
  - Learning rates: [0.01, 0.001, 0.0005, 0.0001]
  - Batch sizes: [32, 64, 128]
  - Hidden layers: [2, 3, 4, 5]
  - Hidden units: [32, 64, 128, 256]
  - Dropout rates: [0.0, 0.1, 0.2, 0.3]

### Dataset
- Synthetic quadratic equations: ax² + bx + c = 0
- Handles both real and complex roots
- Standard scaling for faster convergence
- Configurable size and coefficient range

## Performance

The application automatically detects and uses:
- CUDA GPU (if available)
- Apple Metal Performance Shaders (MPS) on macOS
- CPU (fallback)

## License

This project is part of the Behrend Honors Research Project.

## Author

Behrend Honors Research Project - Hyperoptimization of AI-Based Quadratic Equation Predictors


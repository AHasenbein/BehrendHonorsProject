"""
Configuration constants and default settings for the Quadratic Equation Predictor.
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Default dataset parameters
DEFAULT_SAMPLES = 20000
DEFAULT_RANGE_VAL = 15

# Default model parameters
DEFAULT_INPUT_DIM = 3
DEFAULT_OUTPUT_DIM = 4
DEFAULT_HIDDEN_LAYERS = 3
DEFAULT_HIDDEN_UNITS = 64
DEFAULT_DROPOUT_RATE = 0.1

# Default training parameters
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS_PER_TRIAL = 15
DEFAULT_N_TRIALS = 12
DEFAULT_FINE_TUNE_EPOCHS = 30

# Hyperparameter search space
PARAM_GRID = {
    'lr': [0.01, 0.001, 0.0005, 0.0001],
    'batch_size': [32, 64, 128],
    'hidden_layers': [2, 3, 4, 5],
    'hidden_units': [32, 64, 128, 256],
    'dropout': [0.0, 0.1, 0.2, 0.3]
}

# Benchmark parameters
BENCHMARK_N_RUNS = 10
BENCHMARK_SAMPLE_SIZE = 1000


"""
Utility functions for visualization, metrics, and benchmarking.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import config


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def calculate_metrics(model, dataset, device=None):
    """Calculate comprehensive error metrics."""
    if device is None:
        device = get_device()
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        X = dataset.X.to(device)
        preds = model(X).cpu().numpy()
        actuals = dataset.y.numpy()

    metrics = {}
    labels = ['Root1_Real', 'Root1_Imag', 'Root2_Real', 'Root2_Imag']

    for i, label in enumerate(labels):
        mae = mean_absolute_error(actuals[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(actuals[:, i], preds[:, i]))
        r2 = r2_score(actuals[:, i], preds[:, i])
        metrics[label] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    return metrics, preds, actuals


def run_benchmark(model, dataset, n_runs=10, device=None):
    """Compares AI Inference speed vs Traditional Calculation with statistical rigor."""
    if device is None:
        device = get_device()
    
    model.eval()
    model = model.to(device)
    
    input_sample = dataset.X[:config.BENCHMARK_SAMPLE_SIZE].to(device)
    coeffs = dataset.inputs[:config.BENCHMARK_SAMPLE_SIZE]
    a, b, c = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    ai_times = []
    trad_times = []

    for _ in range(n_runs):
        # AI Timing
        start_ai = time.time()
        with torch.no_grad():
            _ = model(input_sample)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        end_ai = time.time()
        ai_times.append((end_ai - start_ai) * 1000)

        # Traditional Timing (FULL calculation, not just sqrt)
        start_trad = time.time()
        delta = b ** 2 - 4 * a * c

        # Real roots
        mask_real = delta >= 0
        r1_real = np.zeros_like(delta)
        r2_real = np.zeros_like(delta)
        r1_imag = np.zeros_like(delta)
        r2_imag = np.zeros_like(delta)

        if np.any(mask_real):
            sqrt_delta = np.sqrt(delta[mask_real])
            r1_real[mask_real] = (-b[mask_real] + sqrt_delta) / (2 * a[mask_real])
            r2_real[mask_real] = (-b[mask_real] - sqrt_delta) / (2 * a[mask_real])

        # Complex roots
        mask_imag = delta < 0
        if np.any(mask_imag):
            sqrt_delta = np.sqrt(-delta[mask_imag])
            real_part = -b[mask_imag] / (2 * a[mask_imag])
            imag_part = sqrt_delta / (2 * a[mask_imag])
            r1_real[mask_imag] = real_part
            r2_real[mask_imag] = real_part
            r1_imag[mask_imag] = imag_part
            r2_imag[mask_imag] = -imag_part

        end_trad = time.time()
        trad_times.append((end_trad - start_trad) * 1000)

    return {
        'ai_mean': np.mean(ai_times),
        'ai_std': np.std(ai_times),
        'trad_mean': np.mean(trad_times),
        'trad_std': np.std(trad_times),
        'speedup': np.mean(trad_times) / np.mean(ai_times) if np.mean(ai_times) > 0 else 0
    }


def visualize_architecture(model, ax=None):
    """Create modern architecture visualization using new design."""
    from architecture_viz import visualize_architecture_modern
    return visualize_architecture_modern(model, ax)


def plot_results(history, model, dataset, optimizer, device=None):
    """Generate comprehensive research visualizations."""
    if device is None:
        device = get_device()

    # Calculate metrics
    metrics, preds, actuals = calculate_metrics(model, dataset, device)
    benchmark = run_benchmark(model, dataset, device=device)

    # Create figure with 6 subplots
    fig = plt.figure(figsize=(20, 12))

    # 1. Network Architecture
    ax1 = plt.subplot(2, 3, 1)
    visualize_architecture(model, ax=ax1)

    # 2. Hyperoptimization Search Trajectory
    ax2 = plt.subplot(2, 3, 2)
    losses = [h[1] for h in history]
    trials = range(1, len(losses) + 1)
    best_trial = np.argmin(losses) + 1
    ax2.plot(trials, losses, marker='o', linestyle='--', color='b', linewidth=2, markersize=8)
    ax2.scatter([best_trial], [min(losses)], color='red', s=200, marker='*', zorder=5, label='Best Model')
    ax2.set_title("Hyperoptimization Search Trajectory", fontsize=14, weight='bold')
    ax2.set_xlabel("Trial ID", fontsize=12)
    ax2.set_ylabel("Validation MSE Loss", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Convergence Analysis - Best Model
    ax3 = plt.subplot(2, 3, 3)
    best_idx = np.argmin([h[1] for h in history])
    train_losses = history[best_idx][2]
    val_losses = history[best_idx][3]
    epochs = range(1, len(train_losses) + 1)
    ax3.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    ax3.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    ax3.set_title("Convergence Rate (Best Model)", fontsize=14, weight='bold')
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("MSE Loss", fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 4. Topology Comparison
    ax4 = plt.subplot(2, 3, 4)
    topology_data = optimizer.topology_results
    params = [t['params'] for t in topology_data]
    val_losses_topo = [t['val_loss'] for t in topology_data]
    colors = ['red' if i == best_idx else 'blue' for i in range(len(params))]
    ax4.scatter(params, val_losses_topo, c=colors, s=100, alpha=0.6)
    ax4.set_title("Model Topology vs Performance", fontsize=14, weight='bold')
    ax4.set_xlabel("Total Parameters", fontsize=12)
    ax4.set_ylabel("Validation Loss", fontsize=12)
    ax4.grid(True, alpha=0.3)

    # 5. Prediction Accuracy Scatter
    ax5 = plt.subplot(2, 3, 5)
    subset_idx = np.random.choice(len(actuals), min(500, len(actuals)), replace=False)
    ax5.scatter(actuals[subset_idx, 0], preds[subset_idx, 0], alpha=0.5, color='purple', s=30)
    min_val, max_val = actuals[:, 0].min(), actuals[:, 0].max()
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    r2 = metrics['Root1_Real']['R2']
    ax5.set_title(f"Prediction Accuracy (Root 1 Real)\nR² = {r2:.4f}", fontsize=14, weight='bold')
    ax5.set_xlabel("Ground Truth", fontsize=12)
    ax5.set_ylabel("AI Prediction", fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Inference Benchmark with Error Bars
    ax6 = plt.subplot(2, 3, 6)
    methods = ['Traditional\nCalculation', 'AI\nInference']
    means = [benchmark['trad_mean'], benchmark['ai_mean']]
    stds = [benchmark['trad_std'], benchmark['ai_std']]
    colors_bar = ['#FF6B6B', '#4ECDC4']
    bars = ax6.bar(methods, means, yerr=stds, color=colors_bar, capsize=10, alpha=0.7, edgecolor='black', linewidth=2)
    ax6.set_title(f"Inference Latency (1000 samples)\nSpeedup: {benchmark['speedup']:.2f}×", fontsize=14, weight='bold')
    ax6.set_ylabel("Time (ms)", fontsize=12)
    for idx, (bar, mean) in enumerate(zip(bars, means)):
        ax6.text(bar.get_x() + bar.get_width() / 2, mean + stds[idx],
                 f'{mean:.2f}ms', ha='center', va='bottom', fontsize=11, weight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('research_results_comprehensive.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to 'research_results_comprehensive.png'")
    plt.show()

    # Print detailed metrics
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS REPORT")
    print("=" * 60)
    for label, vals in metrics.items():
        print(f"\n{label}:")
        print(f"  MAE:  {vals['MAE']:.6f}")
        print(f"  RMSE: {vals['RMSE']:.6f}")
        print(f"  R²:   {vals['R2']:.6f}")

    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Traditional Method: {benchmark['trad_mean']:.3f} ± {benchmark['trad_std']:.3f} ms")
    print(f"AI Inference:       {benchmark['ai_mean']:.3f} ± {benchmark['ai_std']:.3f} ms")
    print(f"Speedup Factor:     {benchmark['speedup']:.2f}×")

    return fig, metrics, benchmark


"""
Training logic and hyperoptimization engine.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import copy
import config
from model import DynamicQuadraticNet
from utils import get_device


class HyperOptimizer:
    """
    Custom Random Search algorithm to find the optimal Neural Topology.
    """

    def __init__(self, dataset, n_trials=10, epochs_per_trial=20, device=None, callback=None, fixed_config=None):
        self.dataset = dataset
        self.n_trials = n_trials
        self.epochs = epochs_per_trial
        self.device = device if device is not None else get_device()
        self.callback = callback  # Callback function for GUI updates
        self.fixed_config = fixed_config or {}
        
        self.best_model = None
        self.best_loss = float('inf')
        self.best_config = {}
        self.history = []
        self.topology_results = []

        # Search Space
        self.param_grid = config.PARAM_GRID.copy()
        
        # Override with fixed config if provided
        if self.fixed_config.get('use_fixed_config', False):
            # Use fixed configuration instead of random search
            if 'fixed_lr' in self.fixed_config:
                self.param_grid['lr'] = [self.fixed_config['fixed_lr']]
            if 'fixed_batch_size' in self.fixed_config:
                self.param_grid['batch_size'] = [self.fixed_config['fixed_batch_size']]
            if 'fixed_hidden_layers' in self.fixed_config:
                self.param_grid['hidden_layers'] = [self.fixed_config['fixed_hidden_layers']]
            if 'fixed_hidden_units' in self.fixed_config:
                self.param_grid['hidden_units'] = [self.fixed_config['fixed_hidden_units']]
            if 'fixed_dropout' in self.fixed_config:
                self.param_grid['dropout'] = [self.fixed_config['fixed_dropout']]

    def run_search(self):
        """Run hyperparameter search."""
        print(f"--- Starting Hyperoptimization ({self.n_trials} trials) ---")
        print(f"Using device: {self.device}")

        for i in range(self.n_trials):
            # Check if interrupted
            if hasattr(self, '_interrupted') and self._interrupted:
                print("Search interrupted by user.")
                break
                
            # 1. Sample Hyperparameters
            config_dict = {k: random.choice(v) for k, v in self.param_grid.items()}

            # 2. Setup Data Loaders
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            train_ds, val_ds = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            train_loader = DataLoader(train_ds, batch_size=config_dict['batch_size'], shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=config_dict['batch_size'])

            # 3. Initialize Model
            model = DynamicQuadraticNet(
                hidden_layers=config_dict['hidden_layers'],
                hidden_units=config_dict['hidden_units'],
                dropout_rate=config_dict['dropout']
            ).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=config_dict['lr'])
            criterion = nn.MSELoss()

            # 4. Training Loop with convergence tracking
            train_losses = []
            val_losses = []

            for epoch in range(self.epochs):
                # Check if interrupted
                if hasattr(self, '_interrupted') and self._interrupted:
                    break
                    
                model.train()
                epoch_train_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()

                avg_train_loss = epoch_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # Validation
                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        pred = model(X_batch)
                        epoch_val_loss += criterion(pred, y_batch).item()

                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Callback for GUI updates
                if self.callback:
                    self.callback({
                        'trial': i + 1,
                        'epoch': epoch + 1,
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'config': config_dict
                    })

            final_val_loss = val_losses[-1] if val_losses else float('inf')

            print(
                f"Trial {i + 1}/{self.n_trials} | Config: {config_dict} | Val Loss: {final_val_loss:.5f} | Params: {model.count_parameters()}")

            # Store topology information
            topology_info = {
                'trial': i + 1,
                'layers': config_dict['hidden_layers'],
                'units': config_dict['hidden_units'],
                'params': model.count_parameters(),
                'val_loss': final_val_loss,
                'convergence': train_losses
            }
            self.topology_results.append(topology_info)

            self.history.append((config_dict, final_val_loss, train_losses, val_losses))

            # Check if best
            if final_val_loss < self.best_loss:
                self.best_loss = final_val_loss
                self.best_model = copy.deepcopy(model)
                self.best_config = config_dict

        print(f"\n✓ Optimization Complete. Best Loss: {self.best_loss:.5f}")
        print(f"✓ Best Configuration: {self.best_config}")
        if self.best_model:
            print(f"✓ Best Model Parameters: {self.best_model.count_parameters():,}")
        return self.best_model, self.best_config, self.history

    def interrupt(self):
        """Interrupt the search process."""
        self._interrupted = True


def fine_tune_model(model, dataset, config_dict, epochs=30, device=None):
    """Fine-tune a model with validation split."""
    if device is None:
        device = get_device()
    
    model = model.to(device)
    
    # Create train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config_dict['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config_dict['batch_size'])
    
    optimizer = optim.Adam(model.parameters(), lr=config_dict['lr'] * 0.5)  # Lower LR for fine-tuning
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    model.train()
    for epoch in range(epochs):
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch)
                epoch_val_loss += criterion(pred, y_batch).item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    return model, train_losses, val_losses


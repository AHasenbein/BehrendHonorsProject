"""
Wizard step widgets for the main application.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWizard, QWizardPage, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np
import torch
import theme


class BaseWizardPage(QWizardPage):
    """Base class for wizard pages with common functionality."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setTitle(title)
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(25, 25, 25, 25)
        
        # Make page resizable
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        self.setLayout(self.main_layout)
        
    def add_section(self, title: str):
        """Add a section header."""
        header = QLabel(title)
        header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {theme.DARK_THEME['primary']}; margin-top: 10px;")
        self.main_layout.addWidget(header)
        return header


class Step1EquationPage(BaseWizardPage):
    """Step 1: Enter Equation with visual builder."""
    
    def __init__(self, parent=None):
        super().__init__("Step 1: Enter Your Quadratic Equation", parent)
        self.equation_data = {'a': 1.0, 'b': 0.0, 'c': 0.0}
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        # Import here to avoid circular imports
        from equation_builder import EquationBuilder
        
        # Title
        title_label = QLabel("Enter Your Quadratic Equation")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {theme.DARK_THEME['text_bright']}; margin-bottom: 20px;")
        self.main_layout.addWidget(title_label)
        
        # Equation builder
        self.equation_builder = EquationBuilder()
        self.main_layout.addWidget(self.equation_builder, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Examples section
        self.add_section("Quick Examples")
        examples_layout = QHBoxLayout()
        examples = [
            ("x² - 1 = 0", 1, 0, -1),
            ("x² + 2x + 1 = 0", 1, 2, 1),
            ("x² - 5x + 6 = 0", 1, -5, 6),
            ("2x² + 3x - 1 = 0", 2, 3, -1),
        ]
        
        for eq_text, a, b, c in examples:
            btn = QPushButton(eq_text)
            btn.clicked.connect(lambda checked, a=a, b=b, c=c: self.load_example(a, b, c))
            examples_layout.addWidget(btn)
        
        self.main_layout.addLayout(examples_layout)
        self.main_layout.addStretch()
        
    def load_example(self, a, b, c):
        """Load an example equation."""
        self.equation_builder.set_coefficients(a, b, c)
        self.equation_data = {'a': a, 'b': b, 'c': c}
        
    def get_equation(self):
        """Get the current equation coefficients."""
        return self.equation_builder.get_coefficients()
        
    def validatePage(self):
        """Validate the page before proceeding."""
        a, b, c = self.equation_builder.get_coefficients()
        if abs(a) < 0.1:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Equation", "Coefficient 'a' cannot be zero or near-zero.")
            return False
        self.equation_data = {'a': a, 'b': b, 'c': c}
        return True


class Step2ConfigPage(BaseWizardPage):
    """Step 2: Configure Training."""
    
    def __init__(self, parent=None):
        super().__init__("Step 2: Configure Training", parent)
        self.config_data = {}
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        # Two-column layout
        columns_layout = QHBoxLayout()
        
        # Left column: Data Generation
        left_column = QVBoxLayout()
        left_column.setSpacing(15)
        
        data_header = QLabel("Data Generation")
        data_header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        data_header.setStyleSheet(f"color: {theme.DARK_THEME['primary']};")
        left_column.addWidget(data_header)
        
        # Generate from equation toggle
        from PyQt6.QtWidgets import QCheckBox
        self.generate_from_eq = QCheckBox("Generate training data around entered equation")
        self.generate_from_eq.setChecked(True)
        left_column.addWidget(self.generate_from_eq)
        
        # Variation
        variation_layout = QVBoxLayout()
        variation_label = QLabel("Variation (±%):")
        variation_layout.addWidget(variation_label)
        
        from PyQt6.QtWidgets import QSlider, QHBoxLayout as QH
        self.variation_slider = QSlider(Qt.Orientation.Horizontal)
        self.variation_slider.setRange(10, 200)
        self.variation_slider.setValue(50)
        self.variation_label = QLabel("50%")
        self.variation_slider.valueChanged.connect(
            lambda v: self.variation_label.setText(f"{v}%")
        )
        variation_slider_layout = QH()
        variation_slider_layout.addWidget(self.variation_slider)
        variation_slider_layout.addWidget(self.variation_label)
        variation_layout.addLayout(variation_slider_layout)
        left_column.addLayout(variation_layout)
        
        # Number of samples
        samples_layout = QVBoxLayout()
        samples_label = QLabel("Number of Training Samples:")
        samples_layout.addWidget(samples_label)
        
        from PyQt6.QtWidgets import QSpinBox
        self.samples_spinbox = QSpinBox()
        self.samples_spinbox.setRange(1000, 100000)
        self.samples_spinbox.setValue(20000)
        self.samples_spinbox.setSingleStep(1000)
        samples_layout.addWidget(self.samples_spinbox)
        left_column.addLayout(samples_layout)
        
        left_column.addStretch()
        
        # Right column: Model Configuration
        right_column = QVBoxLayout()
        right_column.setSpacing(15)
        
        model_header = QLabel("Model Configuration")
        model_header.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        model_header.setStyleSheet(f"color: {theme.DARK_THEME['primary']};")
        right_column.addWidget(model_header)
        
        # Presets
        presets_label = QLabel("Quick Presets:")
        right_column.addWidget(presets_label)
        presets_layout = QH()
        for preset_name, preset_data in [("Fast", {'lr': 0.01, 'layers': 2, 'units': 32}),
                                          ("Balanced", {'lr': 0.001, 'layers': 3, 'units': 64}),
                                          ("Accurate", {'lr': 0.0005, 'layers': 4, 'units': 128})]:
            btn = QPushButton(preset_name)
            btn.clicked.connect(lambda checked, d=preset_data: self.apply_preset(d))
            presets_layout.addWidget(btn)
        right_column.addLayout(presets_layout)
        
        # Learning rate
        lr_layout = QVBoxLayout()
        lr_label = QLabel("Learning Rate:")
        lr_layout.addWidget(lr_label)
        from PyQt6.QtWidgets import QDoubleSpinBox
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.0001, 0.1)
        self.lr_spinbox.setSingleStep(0.0001)
        self.lr_spinbox.setValue(0.001)
        self.lr_spinbox.setDecimals(4)
        lr_layout.addWidget(self.lr_spinbox)
        right_column.addLayout(lr_layout)
        
        # Batch size
        batch_layout = QVBoxLayout()
        batch_label = QLabel("Batch Size:")
        batch_layout.addWidget(batch_label)
        from PyQt6.QtWidgets import QComboBox
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(["32", "64", "128"])
        self.batch_combo.setCurrentText("64")
        batch_layout.addWidget(self.batch_combo)
        right_column.addLayout(batch_layout)
        
        # Hidden layers
        layers_layout = QVBoxLayout()
        layers_label = QLabel("Hidden Layers:")
        layers_layout.addWidget(layers_label)
        self.layers_spinbox = QSpinBox()
        self.layers_spinbox.setRange(2, 8)
        self.layers_spinbox.setValue(3)
        layers_layout.addWidget(self.layers_spinbox)
        right_column.addLayout(layers_layout)
        
        # Hidden units
        units_layout = QVBoxLayout()
        units_label = QLabel("Hidden Units:")
        units_layout.addWidget(units_label)
        self.units_spinbox = QSpinBox()
        self.units_spinbox.setRange(16, 512)
        self.units_spinbox.setValue(64)
        self.units_spinbox.setSingleStep(32)
        units_layout.addWidget(self.units_spinbox)
        right_column.addLayout(units_layout)
        
        # Dropout
        dropout_layout = QVBoxLayout()
        dropout_label = QLabel("Dropout Rate:")
        dropout_layout.addWidget(dropout_label)
        self.dropout_slider = QSlider(Qt.Orientation.Horizontal)
        self.dropout_slider.setRange(0, 30)
        self.dropout_slider.setValue(10)
        self.dropout_label = QLabel("0.10")
        self.dropout_slider.valueChanged.connect(
            lambda v: self.dropout_label.setText(f"{v/100:.2f}")
        )
        dropout_slider_layout = QH()
        dropout_slider_layout.addWidget(self.dropout_slider)
        dropout_slider_layout.addWidget(self.dropout_label)
        dropout_layout.addLayout(dropout_slider_layout)
        right_column.addLayout(dropout_layout)
        
        # Number of trials
        trials_layout = QVBoxLayout()
        trials_label = QLabel("Number of Trials:")
        trials_layout.addWidget(trials_label)
        self.trials_spinbox = QSpinBox()
        self.trials_spinbox.setRange(1, 50)
        self.trials_spinbox.setValue(12)
        trials_layout.addWidget(self.trials_spinbox)
        right_column.addLayout(trials_layout)
        
        # Epochs per trial
        epochs_layout = QVBoxLayout()
        epochs_label = QLabel("Epochs per Trial:")
        epochs_layout.addWidget(epochs_label)
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 100)
        self.epochs_spinbox.setValue(15)
        epochs_layout.addWidget(self.epochs_spinbox)
        right_column.addLayout(epochs_layout)
        
        right_column.addStretch()
        
        # Add columns
        columns_layout.addLayout(left_column, 1)
        columns_layout.addLayout(right_column, 1)
        self.main_layout.addLayout(columns_layout)
        
    def apply_preset(self, preset_data):
        """Apply a preset configuration."""
        if 'lr' in preset_data:
            self.lr_spinbox.setValue(preset_data['lr'])
        if 'layers' in preset_data:
            self.layers_spinbox.setValue(preset_data['layers'])
        if 'units' in preset_data:
            self.units_spinbox.setValue(preset_data['units'])
            
    def get_config(self):
        """Get the current configuration."""
        return {
            'generate_from_equation': self.generate_from_eq.isChecked(),
            'variation': self.variation_slider.value() / 100.0,
            'samples': self.samples_spinbox.value(),
            'lr': self.lr_spinbox.value(),
            'batch_size': int(self.batch_combo.currentText()),
            'hidden_layers': self.layers_spinbox.value(),
            'hidden_units': self.units_spinbox.value(),
            'dropout': self.dropout_slider.value() / 100.0,
            'n_trials': self.trials_spinbox.value(),
            'epochs_per_trial': self.epochs_spinbox.value(),
        }


class Step3TrainingPage(BaseWizardPage):
    """Step 3: Train Model with progress display."""
    
    def __init__(self, parent=None):
        super().__init__("Step 3: Training Model", parent)
        self.training_thread = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        # Title
        title_label = QLabel("Training in Progress")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {theme.DARK_THEME['text_bright']}; margin-bottom: 20px;")
        self.main_layout.addWidget(title_label)
        
        # Progress section
        progress_layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        # Progress bar
        from PyQt6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        # Time info
        self.time_label = QLabel("Time elapsed: 0s")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.time_label)
        
        self.main_layout.addLayout(progress_layout)
        
        # Metrics section (two columns)
        metrics_layout = QHBoxLayout()
        
        # Left: Current metrics
        metrics_left = QVBoxLayout()
        metrics_left_header = QLabel("Current Metrics")
        metrics_left_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        metrics_left.addWidget(metrics_left_header)
        
        self.best_loss_label = QLabel("Best Loss: N/A")
        self.best_loss_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.best_loss_label.setStyleSheet(f"color: {theme.DARK_THEME['success']};")
        metrics_left.addWidget(self.best_loss_label)
        
        self.current_trial_label = QLabel("Trial: 0/0")
        metrics_left.addWidget(self.current_trial_label)
        
        self.current_epoch_label = QLabel("Epoch: 0/0")
        metrics_left.addWidget(self.current_epoch_label)
        
        metrics_left.addStretch()
        
        # Right: Architecture preview (will be updated)
        metrics_right = QVBoxLayout()
        metrics_right_header = QLabel("Best Model Architecture")
        metrics_right_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        metrics_right.addWidget(metrics_right_header)
        
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        self.arch_figure = Figure(figsize=(6, 4))
        self.arch_canvas = FigureCanvas(self.arch_figure)
        self.arch_canvas.setMinimumHeight(300)
        self.arch_canvas.setMinimumWidth(400)
        metrics_right.addWidget(self.arch_canvas)
        
        metrics_right.addStretch()
        
        metrics_layout.addLayout(metrics_left, 1)
        metrics_layout.addLayout(metrics_right, 1)
        self.main_layout.addLayout(metrics_layout)
        
        self.main_layout.addStretch()
        
    def update_progress(self, trial, total_trials, epoch, total_epochs, train_loss, val_loss, best_loss):
        """Update training progress."""
        total_steps = total_trials * total_epochs
        current_step = (trial - 1) * total_epochs + epoch
        progress = int((current_step / total_steps) * 100) if total_steps > 0 else 0
        
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Training trial {trial} of {total_trials}...")
        self.current_trial_label.setText(f"Trial: {trial}/{total_trials}")
        self.current_epoch_label.setText(f"Epoch: {epoch}/{total_epochs}")
        
        if best_loss < float('inf'):
            self.best_loss_label.setText(f"Best Loss: {best_loss:.6f}")
        
    def update_architecture(self, model):
        """Update architecture visualization."""
        if model is None:
            return
        self.arch_figure.clear()
        ax = self.arch_figure.add_subplot(111)
        from architecture_viz import visualize_architecture_modern
        visualize_architecture_modern(model, ax=ax, figsize=(6, 4))
        self.arch_canvas.draw()


class Step4ResultsPage(BaseWizardPage):
    """Step 4: View Results - Full page with comprehensive information."""
    
    def __init__(self, parent=None):
        super().__init__("Step 4: Training Results", parent)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        # Title
        title_label = QLabel("Training Results Summary")
        title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {theme.DARK_THEME['text_bright']}; margin-bottom: 15px; padding: 10px;")
        self.main_layout.addWidget(title_label)
        
        # Top metrics bar
        metrics_bar = QHBoxLayout()
        metrics_bar.setSpacing(20)
        
        self.final_loss_label = QLabel("Best Loss: N/A")
        self.final_loss_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.final_loss_label.setStyleSheet(f"color: {theme.DARK_THEME['success']}; padding: 10px; background-color: {theme.DARK_THEME['surface']}; border-radius: 5px;")
        metrics_bar.addWidget(self.final_loss_label)
        
        self.config_label = QLabel("Best Config: N/A")
        self.config_label.setFont(QFont("Arial", 12))
        self.config_label.setWordWrap(True)
        self.config_label.setStyleSheet(f"padding: 10px; background-color: {theme.DARK_THEME['surface']}; border-radius: 5px;")
        metrics_bar.addWidget(self.config_label, 2)
        
        self.params_label = QLabel("Parameters: N/A")
        self.params_label.setFont(QFont("Arial", 12))
        self.params_label.setStyleSheet(f"padding: 10px; background-color: {theme.DARK_THEME['surface']}; border-radius: 5px;")
        metrics_bar.addWidget(self.params_label)
        
        self.main_layout.addLayout(metrics_bar)
        
        # Graph grid - takes most of the space
        from graph_grid import GraphGridWidget
        self.graph_grid = GraphGridWidget()
        self.graph_grid.setMinimumHeight(600)
        self.main_layout.addWidget(self.graph_grid, 1)  # Stretch factor
        
    def update_results(self, model, history, best_config, best_loss, dataset, device):
        """Update results display."""
        if best_loss < float('inf'):
            self.final_loss_label.setText(f"Best Validation Loss: {best_loss:.6f}")
            
        if best_config:
            config_text = f"LR: {best_config.get('lr', 'N/A'):.4f} | "
            config_text += f"Batch: {best_config.get('batch_size', 'N/A')} | "
            config_text += f"Layers: {best_config.get('hidden_layers', 'N/A')} | "
            config_text += f"Units: {best_config.get('hidden_units', 'N/A')} | "
            config_text += f"Dropout: {best_config.get('dropout', 'N/A'):.2f}"
            self.config_label.setText(f"Best Config: {config_text}")
        
        if model:
            self.params_label.setText(f"Parameters: {model.count_parameters():,}")
        
        # Update graph grid
        self.graph_grid.update_graphs(model, history, dataset, device)
        self.current_model = model
        self.current_dataset = dataset
        self.current_device = device


class Step5NewEquationPage(BaseWizardPage):
    """Step 5: Test New Equation."""
    
    def __init__(self, parent=None):
        super().__init__("Step 5: Test New Equation", parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        # Title
        title_label = QLabel("Test New Equation")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {theme.DARK_THEME['text_bright']}; margin-bottom: 20px;")
        self.main_layout.addWidget(title_label)
        
        # Equation builder
        from equation_builder import EquationBuilder
        self.prediction_builder = EquationBuilder()
        self.main_layout.addWidget(self.prediction_builder, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Predict button
        predict_btn = QPushButton("Predict Roots")
        predict_btn.setMinimumHeight(40)
        predict_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        predict_btn.clicked.connect(self.predict_equation)
        self.main_layout.addWidget(predict_btn)
        
        # Results display
        self.prediction_result = QLabel("Enter equation coefficients and click 'Predict Roots'")
        self.prediction_result.setWordWrap(True)
        self.prediction_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_result.setStyleSheet(
            f"padding: 20px; background-color: {theme.DARK_THEME['surface']}; "
            f"border-radius: 5px; font-size: 14px; min-height: 150px;"
        )
        self.main_layout.addWidget(self.prediction_result, 1)
        
    def set_model_data(self, model, dataset, device):
        """Set the model and dataset for prediction."""
        self.current_model = model
        self.current_dataset = dataset
        self.current_device = device
        
    def predict_equation(self):
        """Predict roots for the equation in the prediction builder."""
        if not hasattr(self, 'current_model') or self.current_model is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Model", "Please complete training first.")
            return
            
        try:
            a, b, c = self.prediction_builder.get_coefficients()
            
            if abs(a) < 0.1:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Invalid Input", "Coefficient 'a' cannot be zero.")
                return
            
            # Calculate actual roots
            delta = b ** 2 - 4 * a * c
            if delta >= 0:
                sqrt_delta = np.sqrt(delta)
                actual_r1 = (-b + sqrt_delta) / (2 * a)
                actual_r2 = (-b - sqrt_delta) / (2 * a)
                actual_text = f"Actual Roots:\n  Root 1 = {actual_r1:.8f}\n  Root 2 = {actual_r2:.8f}"
            else:
                sqrt_delta = np.sqrt(-delta)
                real_part = -b / (2 * a)
                imag_part = sqrt_delta / (2 * a)
                actual_text = f"Actual Roots (Complex):\n  Root 1 = {real_part:.8f} + {imag_part:.8f}i\n  Root 2 = {real_part:.8f} - {imag_part:.8f}i"
            
            # Predict with model
            inputs = np.array([[a, b, c]])
            inputs_scaled = self.current_dataset.scaler_in.transform(inputs)
            input_tensor = torch.FloatTensor(inputs_scaled).to(self.current_device)
            
            self.current_model.eval()
            with torch.no_grad():
                pred = self.current_model(input_tensor).cpu().numpy()[0]
            
            r1_real, r1_imag, r2_real, r2_imag = pred
            
            if abs(r1_imag) < 1e-6:
                pred_r1 = f"{r1_real:.8f}"
            else:
                pred_r1 = f"{r1_real:.8f} + {r1_imag:.8f}i"
                
            if abs(r2_imag) < 1e-6:
                pred_r2 = f"{r2_real:.8f}"
            else:
                pred_r2 = f"{r2_real:.8f} + {r2_imag:.8f}i"
            
            # Calculate errors
            if delta >= 0:
                error_r1 = abs(r1_real - actual_r1)
                error_r2 = abs(r2_real - actual_r2)
                error_text = f"\n\nPrediction Errors:\n  Root 1 Error: {error_r1:.8f}\n  Root 2 Error: {error_r2:.8f}"
            else:
                error_r1 = abs(r1_real - real_part) + abs(r1_imag - imag_part)
                error_r2 = abs(r2_real - real_part) + abs(r2_imag - imag_part)
                error_text = f"\n\nPrediction Errors:\n  Root 1 Error: {error_r1:.8f}\n  Root 2 Error: {error_r2:.8f}"
            
            result_text = f"AI Prediction:\n  Root 1 = {pred_r1}\n  Root 2 = {pred_r2}\n\n{actual_text}{error_text}"
            self.prediction_result.setText(result_text)
            
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")


"""
New main GUI application with wizard interface.
"""
import sys
import time
from PyQt6.QtWidgets import QApplication, QWizard, QMessageBox, QSizePolicy
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import numpy as np
import torch
import config
from theme import apply_dark_theme, DARK_THEME
from wizard_steps import Step1EquationPage, Step2ConfigPage, Step3TrainingPage, Step4ResultsPage, Step5NewEquationPage
from data import QuadraticDataset
from training import HyperOptimizer
from utils import get_device


class TrainingThread(QThread):
    """Thread for running training without blocking GUI."""
    progress = pyqtSignal(int, int, int, int, float, float, float)  # trial, total_trials, epoch, total_epochs, train_loss, val_loss, best_loss
    finished = pyqtSignal(object, object, object)  # best_model, best_config, history
    error = pyqtSignal(str)
    architecture_update = pyqtSignal(object)  # model

    def __init__(self, dataset, config_dict, device):
        super().__init__()
        self.dataset = dataset
        self.config_dict = config_dict
        self.device = device
        self.optimizer = None
        self._interrupted = False
        self.start_time = None

    def run(self):
        try:
            self.start_time = time.time()
            
            def callback(data):
                if self._interrupted:
                    return
                    
                trial = data['trial']
                epoch = data['epoch']
                train_loss = data['train_loss']
                val_loss = data['val_loss']
                
                # Get best loss from optimizer
                best_loss = self.optimizer.best_loss if self.optimizer else val_loss
                
                self.progress.emit(
                    trial,
                    self.config_dict['n_trials'],
                    epoch,
                    self.config_dict['epochs_per_trial'],
                    train_loss,
                    val_loss,
                    best_loss
                )
                
                # Emit architecture update when best model changes
                if self.optimizer and self.optimizer.best_model:
                    self.architecture_update.emit(self.optimizer.best_model)

            fixed_config = {
                'fixed_lr': self.config_dict.get('fixed_lr'),
                'fixed_batch_size': self.config_dict.get('fixed_batch_size'),
                'fixed_hidden_layers': self.config_dict.get('fixed_hidden_layers'),
                'fixed_hidden_units': self.config_dict.get('fixed_hidden_units'),
                'fixed_dropout': self.config_dict.get('fixed_dropout'),
                'use_fixed_config': self.config_dict.get('use_fixed_config', False),
            }
            
            self.optimizer = HyperOptimizer(
                self.dataset,
                n_trials=self.config_dict['n_trials'],
                epochs_per_trial=self.config_dict['epochs_per_trial'],
                device=self.device,
                callback=callback,
                fixed_config=fixed_config
            )
            
            best_model, best_config, history = self.optimizer.run_search()
            if not self._interrupted:
                self.finished.emit(best_model, best_config, history)
        except Exception as e:
            self.error.emit(str(e))

    def interrupt(self):
        """Interrupt training."""
        self._interrupted = True
        if self.optimizer:
            self.optimizer.interrupt()


class MainWizard(QWizard):
    """Main wizard application."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quadratic Equation Predictor - Hyperoptimization")
        
        # Make fully resizable with better defaults
        self.setMinimumSize(1200, 700)
        self.resize(1800, 1100)
        
        # Enable resizing
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        # Set wizard style
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.HaveHelpButton, False)
        self.setOption(QWizard.WizardOption.IndependentPages, False)
        
        # Data storage
        self.equation_data = {}
        self.config_data = {}
        self.dataset = None
        self.training_thread = None
        self.device = get_device()
        
        # Add pages
        self.step1 = Step1EquationPage()
        self.step2 = Step2ConfigPage()
        self.step3 = Step3TrainingPage()
        self.step4 = Step4ResultsPage()
        self.step5 = Step5NewEquationPage()
        
        self.addPage(self.step1)
        self.addPage(self.step2)
        self.addPage(self.step3)
        self.addPage(self.step4)
        self.addPage(self.step5)
        
        # Connect signals
        self.currentIdChanged.connect(self.on_page_changed)
        
        # Set button text
        self.setButtonText(QWizard.WizardButton.NextButton, "Next")
        self.setButtonText(QWizard.WizardButton.BackButton, "Back")
        self.setButtonText(QWizard.WizardButton.FinishButton, "New Equation")
        self.setButtonText(QWizard.WizardButton.CancelButton, "Cancel")
        
    def on_page_changed(self, page_id):
        """Handle page changes."""
        if page_id == 2:  # Step 3 - Training page
            # Only start training if not already started
            if self.training_thread is None or not self.training_thread.isRunning():
                self.start_training()
        elif page_id == 3:  # Step 4 - Results page
            # Training should be complete
            pass
        elif page_id == 4:  # Step 5 - New Equation page
            # Set model data for prediction
            if hasattr(self, 'current_model') and self.current_model:
                self.step5.set_model_data(self.current_model, self.dataset, self.device)
            
    def start_training(self):
        """Start the training process."""
        # Get equation data
        a, b, c = self.step1.get_equation()
        self.equation_data = {'a': a, 'b': b, 'c': c}
        
        # Get config data
        self.config_data = self.step2.get_config()
        
        # Generate dataset
        try:
            if self.config_data['generate_from_equation']:
                # Generate from equation
                self.dataset = QuadraticDataset.from_equation(
                    a, b, c,
                    samples=self.config_data['samples'],
                    variation=self.config_data['variation'],
                    seed=config.RANDOM_SEED
                )
            else:
                # Generate random dataset
                self.dataset = QuadraticDataset(
                    samples=self.config_data['samples'],
                    range_val=15,
                    seed=config.RANDOM_SEED
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate dataset: {str(e)}")
            return
        
        # Update config with model parameters
        # For now, we'll do hyperoptimization but the user's settings will influence the search
        training_config = {
            'n_trials': self.config_data['n_trials'],
            'epochs_per_trial': self.config_data['epochs_per_trial'],
            'fixed_lr': self.config_data['lr'],
            'fixed_batch_size': self.config_data['batch_size'],
            'fixed_hidden_layers': self.config_data['hidden_layers'],
            'fixed_hidden_units': self.config_data['hidden_units'],
            'fixed_dropout': self.config_data['dropout'],
            'use_fixed_config': False,  # Keep hyperoptimization enabled
        }
        
        # Start training thread (pass fixed_config to optimizer)
        self.training_thread = TrainingThread(self.dataset, training_config, self.device)
        self.training_thread.progress.connect(self.step3.update_progress)
        self.training_thread.architecture_update.connect(self.step3.update_architecture)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.error.connect(self.training_error)
        
        # Update time label
        self.step3.time_label.setText("Time elapsed: 0s")
        self.training_start_time = time.time()
        
        # Timer for updating time
        from PyQt6.QtCore import QTimer
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)  # Update every second
        
        self.training_thread.start()
        
        # Disable next button during training
        self.button(QWizard.WizardButton.NextButton).setEnabled(False)
        
    def update_time(self):
        """Update elapsed time display."""
        if hasattr(self, 'training_start_time') and self.training_thread and self.training_thread.isRunning():
            elapsed = int(time.time() - self.training_start_time)
            self.step3.time_label.setText(f"Time elapsed: {elapsed}s")
        else:
            if hasattr(self, 'time_timer'):
                self.time_timer.stop()
        
    def training_finished(self, best_model, best_config, history):
        """Handle training completion."""
        # Stop time timer
        if hasattr(self, 'time_timer'):
            self.time_timer.stop()
            
        # Update final time
        if hasattr(self, 'training_start_time'):
            elapsed = int(time.time() - self.training_start_time)
            self.step3.time_label.setText(f"Time elapsed: {elapsed}s (Complete)")
        
        # Enable next button
        self.button(QWizard.WizardButton.NextButton).setEnabled(True)
        
        # Store model data
        self.current_model = best_model
        self.best_config = best_config
        
        # Update results page
        best_loss = min([h[1] for h in history]) if history else float('inf')
        self.step4.update_results(
            best_model,
            history,
            best_config,
            best_loss,
            self.dataset,
            self.device
        )
        
        # Auto-advance to results page
        self.next()
        
    def training_error(self, error_msg):
        """Handle training error."""
        QMessageBox.critical(self, "Training Error", f"Training failed: {error_msg}")
        self.button(QWizard.WizardButton.NextButton).setEnabled(True)
        
    def closeEvent(self, event):
        """Handle window close event."""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Training in Progress",
                "Training is still running. Do you want to cancel and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                if self.training_thread:
                    self.training_thread.interrupt()
                    self.training_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set random seeds
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    import random
    random.seed(config.RANDOM_SEED)
    
    # Apply professional visual style
    from visual_style import apply_professional_style
    apply_professional_style()
    
    # Apply dark theme
    apply_dark_theme(app)
    
    # Create and show wizard
    wizard = MainWizard()
    wizard.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


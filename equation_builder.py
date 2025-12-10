"""
Visual equation builder with LaTeX rendering.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import theme


class EquationBuilder(QWidget):
    """Visual equation builder with LaTeX rendering."""
    
    equation_changed = pyqtSignal(float, float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.coefficients = {'a': 1.0, 'b': 0.0, 'c': 0.0}
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # LaTeX equation display
        self.equation_figure = Figure(figsize=(8, 2))
        self.equation_canvas = FigureCanvas(self.equation_figure)
        self.equation_canvas.setMinimumHeight(100)
        self.equation_canvas.setMaximumHeight(150)
        layout.addWidget(self.equation_canvas)
        
        # Coefficient inputs
        coeff_layout = QHBoxLayout()
        coeff_layout.setSpacing(20)
        
        # Coefficient a
        a_layout = QVBoxLayout()
        a_label = QLabel("Coefficient a:")
        a_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        a_layout.addWidget(a_label)
        self.a_input = QLineEdit("1.0")
        self.a_input.setMaximumWidth(120)
        self.a_input.textChanged.connect(self.update_equation)
        a_layout.addWidget(self.a_input)
        coeff_layout.addLayout(a_layout)
        
        # Coefficient b
        b_layout = QVBoxLayout()
        b_label = QLabel("Coefficient b:")
        b_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b_layout.addWidget(b_label)
        self.b_input = QLineEdit("0.0")
        self.b_input.setMaximumWidth(120)
        self.b_input.textChanged.connect(self.update_equation)
        b_layout.addWidget(self.b_input)
        coeff_layout.addLayout(b_layout)
        
        # Coefficient c
        c_layout = QVBoxLayout()
        c_label = QLabel("Coefficient c:")
        c_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c_layout.addWidget(c_label)
        self.c_input = QLineEdit("0.0")
        self.c_input.setMaximumWidth(120)
        self.c_input.textChanged.connect(self.update_equation)
        c_layout.addWidget(self.c_input)
        coeff_layout.addLayout(c_layout)
        
        coeff_layout.addStretch()
        layout.addLayout(coeff_layout)
        
        # Validation indicator
        self.validation_label = QLabel("")
        self.validation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.validation_label.setStyleSheet(f"color: {theme.DARK_THEME['success']};")
        layout.addWidget(self.validation_label)
        
        self.setLayout(layout)
        
        # Initial render
        self.update_equation()
        
    def update_equation(self):
        """Update the LaTeX equation display."""
        try:
            a = float(self.a_input.text())
            b = float(self.b_input.text())
            c = float(self.c_input.text())
            
            self.coefficients = {'a': a, 'b': b, 'c': c}
            
            # Validate
            if abs(a) < 0.1:
                self.validation_label.setText("⚠ Warning: Coefficient 'a' is near zero")
                self.validation_label.setStyleSheet(f"color: {theme.DARK_THEME['warning']};")
            else:
                self.validation_label.setText("✓ Valid equation")
                self.validation_label.setStyleSheet(f"color: {theme.DARK_THEME['success']};")
            
            # Build LaTeX string
            latex_parts = []
            
            # ax² term
            if abs(a) < 1e-6:
                pass  # Skip if zero
            elif abs(a - 1.0) < 1e-6:
                latex_parts.append("x^2")
            elif abs(a + 1.0) < 1e-6:
                latex_parts.append("-x^2")
            else:
                latex_parts.append(f"{a:.2f}x^2")
            
            # bx term
            if abs(b) > 1e-6:
                sign = "+" if b > 0 else ""
                if abs(b - 1.0) < 1e-6:
                    latex_parts.append(f"{sign}x")
                elif abs(b + 1.0) < 1e-6:
                    latex_parts.append("-x")
                else:
                    latex_parts.append(f"{sign}{b:.2f}x")
            
            # c term
            if abs(c) > 1e-6:
                sign = "+" if c > 0 else ""
                latex_parts.append(f"{sign}{c:.2f}")
            
            # Combine
            if not latex_parts:
                equation_str = "0"
            else:
                equation_str = "".join(latex_parts)
            
            equation_str = f"${equation_str} = 0$"
            
            # Render
            self.equation_figure.clear()
            ax = self.equation_figure.add_subplot(111)
            ax.text(0.5, 0.5, equation_str, fontsize=24, ha='center', va='center',
                   transform=ax.transAxes)
            ax.axis('off')
            self.equation_canvas.draw()
            
            # Emit signal
            self.equation_changed.emit(a, b, c)
            
        except ValueError:
            self.validation_label.setText("⚠ Invalid input")
            self.validation_label.setStyleSheet(f"color: {theme.DARK_THEME['error']};")
            
    def set_coefficients(self, a, b, c):
        """Set the coefficients programmatically."""
        self.a_input.setText(str(a))
        self.b_input.setText(str(b))
        self.c_input.setText(str(c))
        self.update_equation()
        
    def get_coefficients(self):
        """Get the current coefficients."""
        try:
            a = float(self.a_input.text())
            b = float(self.b_input.text())
            c = float(self.c_input.text())
            return a, b, c
        except ValueError:
            return 1.0, 0.0, 0.0


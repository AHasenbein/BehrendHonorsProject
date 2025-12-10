"""
Dark theme styling and colors for the GUI.
"""
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

DARK_THEME = {
    'background': '#0D1117',      # GitHub dark - deeper, more professional
    'surface': '#161B22',         # Dark surface
    'surface_light': '#21262D',   # Lighter surface
    'primary': '#0078D4',         # Microsoft blue - high contrast
    'primary_hover': '#40A6FF',   # Light blue hover
    'primary_dark': '#005A9E',    # Darker blue
    'secondary': '#00BCF2',       # Cyan accent
    'text': '#D2D2D2',            # Light gray text
    'text_bright': '#FFFFFF',     # Pure white
    'text_dim': '#8B949E',        # Dimmed text
    'success': '#16C60C',         # Bright green
    'success_dark': '#107C10',    # Dark green
    'warning': '#FFB900',         # Amber
    'error': '#E81123',           # Red
    'accent': '#FF6B35',          # Coral orange
    'border': '#30363D',          # Border gray
    'border_light': '#484F58',    # Light border
    'highlight': '#FFB900',       # Highlight color
    'button': '#0078D4',          # Button blue
    'button_hover': '#40A6FF',    # Button hover
    'button_pressed': '#005A9E',  # Button pressed
    'grid': '#21262D',            # Grid color
}

def apply_dark_theme(app: QApplication):
    """Apply dark theme to the application."""
    palette = QPalette()
    
    # Window colors
    palette.setColor(QPalette.ColorRole.Window, QColor(DARK_THEME['background']))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(DARK_THEME['text']))
    
    # Base colors
    palette.setColor(QPalette.ColorRole.Base, QColor(DARK_THEME['surface']))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(DARK_THEME['surface_light']))
    
    # Text colors
    palette.setColor(QPalette.ColorRole.Text, QColor(DARK_THEME['text']))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(DARK_THEME['text_bright']))
    
    # Button colors
    palette.setColor(QPalette.ColorRole.Button, QColor(DARK_THEME['surface']))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(DARK_THEME['text']))
    
    # Highlight colors
    palette.setColor(QPalette.ColorRole.Highlight, QColor(DARK_THEME['primary']))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(DARK_THEME['text_bright']))
    
    # Link colors
    palette.setColor(QPalette.ColorRole.Link, QColor(DARK_THEME['primary']))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(DARK_THEME['primary']))
    
    app.setPalette(palette)
    
    # Set stylesheet for additional styling
    app.setStyleSheet(f"""
        QMainWindow {{
            background-color: {DARK_THEME['background']};
            color: {DARK_THEME['text']};
        }}
        QWidget {{
            background-color: {DARK_THEME['background']};
            color: {DARK_THEME['text']};
        }}
        QPushButton {{
            background-color: {DARK_THEME['button']};
            color: {DARK_THEME['text_bright']};
            border: 2px solid {DARK_THEME['button']};
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 13px;
            font-weight: bold;
            min-height: 35px;
        }}
        QPushButton:hover {{
            background-color: {DARK_THEME['button_hover']};
            border-color: {DARK_THEME['button_hover']};
        }}
        QPushButton:pressed {{
            background-color: {DARK_THEME['button_pressed']};
            border-color: {DARK_THEME['button_pressed']};
        }}
        QPushButton:disabled {{
            background-color: {DARK_THEME['surface_light']};
            color: {DARK_THEME['text_dim']};
            border-color: {DARK_THEME['border']};
        }}
        QLineEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {DARK_THEME['surface']};
            color: {DARK_THEME['text']};
            border: 1px solid {DARK_THEME['border']};
            border-radius: 4px;
            padding: 6px;
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid {DARK_THEME['primary']};
            background-color: {DARK_THEME['surface_light']};
        }}
        QComboBox {{
            background-color: {DARK_THEME['surface']};
            color: {DARK_THEME['text']};
            border: 1px solid {DARK_THEME['border']};
            border-radius: 4px;
            padding: 6px;
        }}
        QComboBox:focus {{
            border: 2px solid {DARK_THEME['primary']};
        }}
        QSlider::groove:horizontal {{
            background: {DARK_THEME['secondary']};
            height: 6px;
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {DARK_THEME['primary']};
            width: 18px;
            height: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {DARK_THEME['primary_hover']};
        }}
        QProgressBar {{
            background-color: {DARK_THEME['secondary']};
            border: 1px solid {DARK_THEME['border']};
            border-radius: 4px;
            text-align: center;
            color: {DARK_THEME['text']};
        }}
        QProgressBar::chunk {{
            background-color: {DARK_THEME['primary']};
            border-radius: 3px;
        }}
        QGroupBox {{
            border: 2px solid {DARK_THEME['border']};
            border-radius: 6px;
            margin-top: 15px;
            padding-top: 15px;
            font-weight: bold;
            font-size: 13px;
            color: {DARK_THEME['text_bright']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }}
        QLabel {{
            color: {DARK_THEME['text']};
        }}
        QTextEdit {{
            background-color: {DARK_THEME['surface']};
            color: {DARK_THEME['text']};
            border: 1px solid {DARK_THEME['border']};
            border-radius: 4px;
        }}
        QTableWidget {{
            background-color: {DARK_THEME['surface']};
            color: {DARK_THEME['text']};
            border: 1px solid {DARK_THEME['border']};
            gridline-color: {DARK_THEME['border']};
        }}
        QTableWidget::item {{
            padding: 4px;
        }}
        QTableWidget::item:selected {{
            background-color: {DARK_THEME['highlight']};
        }}
        QTabWidget::pane {{
            border: 1px solid {DARK_THEME['border']};
            background-color: {DARK_THEME['background']};
        }}
        QTabBar::tab {{
            background-color: {DARK_THEME['surface']};
            color: {DARK_THEME['text']};
            padding: 8px 16px;
            border: 1px solid {DARK_THEME['border']};
            border-bottom: none;
        }}
        QTabBar::tab:selected {{
            background-color: {DARK_THEME['background']};
            color: {DARK_THEME['primary']};
        }}
        QScrollArea {{
            border: 1px solid {DARK_THEME['border']};
            background-color: {DARK_THEME['background']};
        }}
    """)


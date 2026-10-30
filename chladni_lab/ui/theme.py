"""Temas y estilos visuales para PyQt5 con estilo Fusion en Linux Mint XFCE."""
from PyQt5 import QtCore, QtGui, QtWidgets


def apply_theme(app: QtWidgets.QApplication, theme: str = "dark") -> None:
    """Aplica el tema visual (dark o light) con el motor Fusion de Qt."""
    app.setStyle("Fusion")
    palette = QtGui.QPalette()

    if theme == "light":
        # Paleta Clara
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(245, 245, 247))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(235, 235, 238))
        palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(235, 235, 238))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
        palette.setColor(QtGui.QPalette.Link, QtGui.QColor(33, 115, 204))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(33, 115, 204))
        palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
    else:
        # Paleta Oscura moderna (estándar para laboratorios de audio y visualización de nodos)
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(34, 36, 42))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(225, 228, 235))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(22, 24, 28))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(30, 32, 38))
        palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(225, 228, 235))
        palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(34, 36, 42))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(225, 228, 235))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 48, 56))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(225, 228, 235))
        palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
        palette.setColor(QtGui.QPalette.Link, QtGui.QColor(74, 144, 226))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(74, 144, 226))
        palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)

    app.setPalette(palette)


def get_stylesheet(theme: str = "dark") -> str:
    """Devuelve hojas de estilo CSS para controles clave (bordes, botones destacados, pestañas)."""
    if theme == "light":
        return """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d0d5;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 10px;
                background-color: #fafafc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #2b5797;
            }
            QPushButton {
                border-radius: 4px;
                padding: 4px 10px;
                background-color: #e4e4e8;
                border: 1px solid #c0c0c5;
            }
            QPushButton:hover {
                background-color: #d8d8de;
            }
            QPushButton:pressed {
                background-color: #c8c8ce;
            }
            QPushButton:checked {
                background-color: #2b5797;
                color: white;
            }
            QTableWidget {
                gridline-color: #e0e0e5;
                border: 1px solid #d0d0d5;
                background-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #e8e8ed;
                padding: 4px;
                font-weight: bold;
                border: 1px solid #d0d0d5;
            }
        """
    else:
        return """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3d414d;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 10px;
                background-color: #262930;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #5da0e4;
            }
            QPushButton {
                border-radius: 4px;
                padding: 4px 10px;
                background-color: #353842;
                border: 1px solid #484c5a;
                color: #e1e4eb;
            }
            QPushButton:hover {
                background-color: #404450;
            }
            QPushButton:pressed {
                background-color: #2c2f38;
            }
            QPushButton:checked {
                background-color: #3b7bc4;
                color: white;
            }
            QTableWidget {
                gridline-color: #333640;
                border: 1px solid #3d414d;
                background-color: #191b20;
            }
            QHeaderView::section {
                background-color: #282b33;
                padding: 4px;
                font-weight: bold;
                border: 1px solid #3d414d;
                color: #d0d4e0;
            }
        """


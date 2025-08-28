import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QListWidget, QFileDialog,
                             QLabel, QGroupBox, QMessageBox, QAction)
from PyQt5.QtCore import Qt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import numpy as np

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.plotted_vectors = []

    def plot_data(self, data, title, clear_first=False):
        if clear_first:
            self.fig.clear()
            self.plotted_vectors = []

        if not self.fig.axes:
            ax = self.fig.add_subplot(111)
        else:
            ax = self.fig.axes[0]

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        color_idx = len(self.plotted_vectors) % len(colors)

        if len(data.shape) == 1:
            ax.plot(data, label=title, color=colors[color_idx])
        else:
            line_styles = ['-', '--', '-.', ':']
            for i, col in enumerate(data.columns):
                style = line_styles[i % len(line_styles)]
                ax.plot(data[col],
                        label=f"{title}.{col}",
                        color=colors[color_idx],
                        linestyle=style)

        self.plotted_vectors.append(title)

        ax.legend()
        ax.grid(True)
        ax.set_title(f"Vectors: {', '.join(self.plotted_vectors)}")

        self.draw()

    def clear_plot(self):
        self.fig.clear()
        self.plotted_vectors = []
        self.draw()

class PlotWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Plot View')
        self.setGeometry(300, 300, 900, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Plot controls
        plot_controls = QHBoxLayout()

        self.plot_selected_btn = QPushButton('Plot Selected (New)')
        plot_controls.addWidget(self.plot_selected_btn)

        self.add_selected_btn = QPushButton('Add Selected to Plot')
        plot_controls.addWidget(self.add_selected_btn)

        self.clear_plot_btn = QPushButton('Clear Plot')
        plot_controls.addWidget(self.clear_plot_btn)

        layout.addLayout(plot_controls)

        # Currently plotted vectors info
        self.plotted_info_label = QLabel("No vectors plotted")
        layout.addWidget(self.plotted_info_label)

        # Plot canvas
        self.plot_canvas = PlotCanvas(self, width=8, height=6)
        layout.addWidget(self.plot_canvas)

        # Connect signals
        self.clear_plot_btn.clicked.connect(self.clear_plot)

    def clear_plot(self):
        self.plot_canvas.clear_plot()
        self.update_plotted_info()

    def update_plotted_info(self):
        if self.plot_canvas.plotted_vectors:
            info_text = f"Currently plotted: {', '.join(self.plot_canvas.plotted_vectors)}"
        else:
            info_text = "No vectors plotted"
        self.plotted_info_label.setText(info_text)

class CSVManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.vectors = {}
        self.plot_window = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CSV Vector Manager')
        self.setGeometry(100, 100, 400, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Import section
        import_group = QGroupBox("Import CSV Files")
        import_layout = QVBoxLayout()
        import_group.setLayout(import_layout)

        import_btn = QPushButton('Import CSV Files')
        import_btn.clicked.connect(self.import_csvs)
        import_layout.addWidget(import_btn)

        layout.addWidget(import_group)

        # File list section
        list_group = QGroupBox("Loaded Vectors")
        list_layout = QVBoxLayout()
        list_group.setLayout(list_layout)

        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_vector_selected)
        list_layout.addWidget(self.list_widget)

        # Vector info label
        self.vector_info_label = QLabel("Select a vector to see details")
        list_layout.addWidget(self.vector_info_label)

        # Action buttons
        btn_layout = QVBoxLayout()

        plot_btn = QPushButton('Plot Vector (New)')
        plot_btn.clicked.connect(self.plot_vector_new)
        btn_layout.addWidget(plot_btn)

        add_plot_btn = QPushButton('Add to Plot')
        add_plot_btn.clicked.connect(self.add_to_plot)
        btn_layout.addWidget(add_plot_btn)

        remove_btn = QPushButton('Remove Vector')
        remove_btn.clicked.connect(self.remove_vector)
        btn_layout.addWidget(remove_btn)

        list_layout.addLayout(btn_layout)
        layout.addWidget(list_group)

        # Create menu bar
        self.create_menu_bar()

        # Create plot window initially
        self.create_plot_window()

    def create_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        import_action = QAction('Import CSV', self)
        import_action.triggered.connect(self.import_csvs)
        file_menu.addAction(import_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('View')

        show_plot_action = QAction('Show Plot View', self)
        show_plot_action.triggered.connect(self.show_plot_view)
        view_menu.addAction(show_plot_action)

        hide_plot_action = QAction('Hide Plot View', self)
        hide_plot_action.triggered.connect(self.hide_plot_view)
        view_menu.addAction(hide_plot_action)

        # Plot menu
        plot_menu = menubar.addMenu('Plot')

        new_plot_action = QAction('New Plot', self)
        new_plot_action.triggered.connect(self.plot_vector_new)
        plot_menu.addAction(new_plot_action)

        add_plot_action = QAction('Add to Plot', self)
        add_plot_action.triggered.connect(self.add_to_plot)
        plot_menu.addAction(add_plot_action)

        clear_plot_action = QAction('Clear Plot', self)
        clear_plot_action.triggered.connect(self.clear_plot)
        plot_menu.addAction(clear_plot_action)

    def create_plot_window(self):
        if self.plot_window is None:
            self.plot_window = PlotWindow(self)
            # Connect plot window buttons to main window methods
            self.plot_window.plot_selected_btn.clicked.connect(self.plot_vector_new)
            self.plot_window.add_selected_btn.clicked.connect(self.add_to_plot)
        self.plot_window.show()
        return self.plot_window

    def show_plot_view(self):
        if self.plot_window is None:
            self.create_plot_window()
        else:
            self.plot_window.show()
            self.plot_window.raise_()
            self.plot_window.activateWindow()

    def hide_plot_view(self):
        if self.plot_window:
            self.plot_window.hide()

    def import_csvs(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 'Select CSV Files', '', 'CSV files (*.csv)'
        )

        for file in files:
            try:
                df = pd.read_csv(file)
                name = os.path.basename(file)
                self.vectors[name] = df
                self.list_widget.addItem(f"{name} ({df.shape[0]}x{df.shape[1]})")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading {file}:\n{str(e)}")

    def on_vector_selected(self, item):
        # Update vector info when selected
        vector_name = item.text().split(' (')[0]
        if vector_name in self.vectors:
            df = self.vectors[vector_name]
            info_text = f"Selected: {vector_name} | Shape: {df.shape} | Columns: {list(df.columns)}"
            self.vector_info_label.setText(info_text)

    def plot_vector_new(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a vector to plot")
            return

        # Ensure plot window exists and is visible
        if self.plot_window is None:
            self.create_plot_window()
        else:
            self.plot_window.show()
            self.plot_window.raise_()
            self.plot_window.activateWindow()

        vector_name = current_item.text().split(' (')[0]
        df = self.vectors[vector_name]

        try:
            self.plot_window.plot_canvas.plot_data(df, title=vector_name, clear_first=True)
            self.plot_window.update_plotted_info()
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Error plotting {vector_name}:\n{str(e)}")

    def add_to_plot(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a vector to add to plot")
            return

        # Ensure plot window exists and is visible
        if self.plot_window is None:
            self.create_plot_window()
        else:
            self.plot_window.show()
            self.plot_window.raise_()
            self.plot_window.activateWindow()

        vector_name = current_item.text().split(' (')[0]

        if vector_name in self.plot_window.plot_canvas.plotted_vectors:
            QMessageBox.information(self, "Info", f"{vector_name} is already plotted")
            return

        df = self.vectors[vector_name]

        try:
            self.plot_window.plot_canvas.plot_data(df, title=vector_name, clear_first=False)
            self.plot_window.update_plotted_info()
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Error adding {vector_name} to plot:\n{str(e)}")

    def clear_plot(self):
        if self.plot_window:
            self.plot_window.clear_plot()

    def remove_vector(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a vector to remove")
            return

        vector_name = current_item.text().split(' (')[0]

        reply = QMessageBox.question(self, 'Remove Vector',
                                     f'Remove {vector_name}?',
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            del self.vectors[vector_name]

            row = self.list_widget.row(current_item)
            self.list_widget.takeItem(row)

            # Clear info label
            self.vector_info_label.setText("Select a vector to see details")

            # Remove from plot if it's plotted
            if (self.plot_window and
                    vector_name in self.plot_window.plot_canvas.plotted_vectors):
                self.clear_plot()
                QMessageBox.information(self, "Info", "Plot cleared because plotted vector was removed")

    def closeEvent(self, event):
        """Handle main window close event"""
        reply = QMessageBox.question(self, 'Exit',
                                     'Are you sure you want to exit?',
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Close plot window
            if self.plot_window:
                self.plot_window.close()
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CSVManager()
    window.show()
    sys.exit(app.exec_())
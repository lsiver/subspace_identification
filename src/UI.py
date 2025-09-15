import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QListWidget, QFileDialog,
                             QLabel, QGroupBox, QMessageBox, QAction, QInputDialog,
                             QDialog, QDialogButtonBox, QListWidgetItem, QCheckBox,
                             QScrollArea, QFrame, QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import numpy as np
import json
#from CaseRunClass import CaseRun
from src.CasesClass import Case


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

        # central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # plot controls
        plot_controls = QHBoxLayout()

        self.plot_selected_btn = QPushButton('Plot Selected (New)')
        plot_controls.addWidget(self.plot_selected_btn)

        self.add_selected_btn = QPushButton('Add Selected to Plot')
        plot_controls.addWidget(self.add_selected_btn)

        self.clear_plot_btn = QPushButton('Clear Plot')
        plot_controls.addWidget(self.clear_plot_btn)

        layout.addLayout(plot_controls)

        self.plotted_info_label = QLabel("No vectors plotted")
        layout.addWidget(self.plotted_info_label)

        # plot canvas
        self.plot_canvas = PlotCanvas(self, width=8, height=6)
        layout.addWidget(self.plot_canvas)

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

class CaseDialog(QDialog):
    def __init__(self, parent=None, available_vectors=None, case_data=None):
        super().__init__(parent)
        self.available_vectors = available_vectors or []
        self.case_data = case_data or {"name": "", "inputs": [], "outputs": [], "ttss": None}
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Create/Edit Case')
        self.setGeometry(200, 200, 500, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # case name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Case Name:"))
        self.name_input = QInputDialog()
        # simple input for the name
        name, ok = QInputDialog.getText(self, 'Case Name', 'Enter case name:', text=self.case_data["name"])
        if not ok or not name.strip():
            self.reject()
            return
        self.case_name = name.strip()

        # main layout after getting name
        self.setWindowTitle(f'Configure Case: {self.case_name}')

        layout.addWidget(QLabel("Select vectors for inputs and outputs:"))

        #TTSS fields
        ttss_group = QGroupBox("TTSS preset")
        ttss_row = QHBoxLayout(ttss_group)

        raw_ttss = self.case_data.get("ttss",[45,90,120])
        if isinstance(raw_ttss,(int,float)):
            defaults = [int(raw_ttss),90,120]
        else:
            defaults = list(raw_ttss) if isinstance(raw_ttss, list) else [45,90,120]
        defaults = (defaults + [45,90,120])[:3]

        self.ttss_spins = []
        for k, val in enumerate(defaults, 1):
            ttss_row.addWidget(QLabel(f"TTSS{k}:"))
            spin = QSpinBox()
            spin.setRange(0, 6000)
            spin.setSingleStep(5)
            spin.setValue(int(val))
            self.ttss_spins.append(spin)
            ttss_row.addWidget(spin)

        layout.addWidget(ttss_group)


        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)

        inputs_group = QGroupBox("Input Vectors")
        inputs_layout = QVBoxLayout()
        inputs_group.setLayout(inputs_layout)

        self.input_checkboxes = {}
        for vector in self.available_vectors:
            checkbox = QCheckBox(vector)
            checkbox.setChecked(vector in self.case_data["inputs"])
            self.input_checkboxes[vector] = checkbox
            inputs_layout.addWidget(checkbox)

        scroll_layout.addWidget(inputs_group)

        # output vectors section
        outputs_group = QGroupBox("Output Vectors")
        outputs_layout = QVBoxLayout()
        outputs_group.setLayout(outputs_layout)

        self.output_checkboxes = {}
        for vector in self.available_vectors:
            checkbox = QCheckBox(vector)
            checkbox.setChecked(vector in self.case_data["outputs"])
            self.output_checkboxes[vector] = checkbox
            outputs_layout.addWidget(checkbox)

        scroll_layout.addWidget(outputs_group)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_case_data(self):
        inputs = [vector for vector, checkbox in self.input_checkboxes.items() if checkbox.isChecked()]
        outputs = [vector for vector, checkbox in self.output_checkboxes.items() if checkbox.isChecked()]
        ttss_list = [spin.value() for spin in self.ttss_spins]

        return {
            "name": self.case_name,
            "inputs": inputs,
            "outputs": outputs,
            "ttss": ttss_list,
        }

class InputScalingDialog(QDialog):
    def __init__(self, parent=None, vectors=None, defaults=None,title="Input Scaling"):
        super().__init__(parent)
        self.vectors = list(vectors or [])
        self.defaults = dict(defaults or {})
        self.setWindowTitle(title)
        self.resize(480,420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Set scaling for the inputs:", self))

        self.table = QTableWidget(len(self.vectors), 2, self)
        self.table.setHorizontalHeaderLabels(["Input", "Scale"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnWidth(0, 280)
        self.table.setColumnWidth(1, 120)

        for r, name in enumerate(self.vectors):
        # left column: name
            item = QTableWidgetItem(name)
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(r, 0, item)

            #right column for scaling
            spin = QSpinBox(self.table)
            spin.setRange(0, 1_000_000)
            spin.setSingleStep(1)
            spin.setValue(int(self.defaults.get(name, 1)))
            self.table.setCellWidget(r, 1, spin)

        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        # ok/cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self):
        out = {}
        for r in range(self.table.rowCount()):
            name = self.table.item(r, 0).text()
            spin = self.table.cellWidget(r, 1)
            out[name] = spin.value()
        return out

class CSVManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.vectors = {}
        self.cases = {}
        self.plot_window = None
        self.cases_file = "cases.json"
        self.initUI()
        self.load_cases()
        self.caserun_list = {} #not used yet
        self.input_scaling = {}

    def initUI(self):
        self.setWindowTitle('MIMO Dynamics')
        self.setGeometry(100, 100, 600, 700)

        # central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # left side, vectors
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # import section
        import_group = QGroupBox("Import CSV Files")
        import_layout = QVBoxLayout()
        import_group.setLayout(import_layout)

        import_btn = QPushButton('Import CSV Files')
        import_btn.clicked.connect(self.import_csvs)
        import_layout.addWidget(import_btn)

        left_layout.addWidget(import_group)

        # file list section
        list_group = QGroupBox("Loaded Vectors")
        list_layout = QVBoxLayout()
        list_group.setLayout(list_layout)

        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_vector_selected)
        list_layout.addWidget(self.list_widget)

        # vector info label
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
        left_layout.addWidget(list_group)

        main_layout.addWidget(left_panel)

        # right side, cases
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # cases section
        cases_group = QGroupBox("Cases")
        cases_layout = QVBoxLayout()
        cases_group.setLayout(cases_layout)

        # case controls
        case_controls = QHBoxLayout()

        create_case_btn = QPushButton('Create Case')
        create_case_btn.clicked.connect(self.create_case)
        case_controls.addWidget(create_case_btn)

        edit_case_btn = QPushButton('Edit Case')
        edit_case_btn.clicked.connect(self.edit_case)
        case_controls.addWidget(edit_case_btn)

        run_case_btn = QPushButton('Run Case')
        run_case_btn.clicked.connect(self.run_case)
        case_controls.addWidget(run_case_btn)

        run_preds_btn = QPushButton('Run Pred')
        run_preds_btn.clicked.connect(self.run_pred)
        case_controls.addWidget(run_preds_btn)

        cases_layout.addLayout(case_controls)

        # cases list
        self.cases_list = QListWidget()
        cases_layout.addWidget(self.cases_list)

        # case info
        self.case_info_label = QLabel("Select a case to see details")
        cases_layout.addWidget(self.case_info_label)

        # case actions
        case_actions = QVBoxLayout()

        plot_case_inputs_btn = QPushButton('Plot Case Inputs')
        plot_case_inputs_btn.clicked.connect(self.plot_case_inputs)
        case_actions.addWidget(plot_case_inputs_btn)

        plot_case_outputs_btn = QPushButton('Plot Case Outputs')
        plot_case_outputs_btn.clicked.connect(self.plot_case_outputs)
        case_actions.addWidget(plot_case_outputs_btn)

        delete_case_btn = QPushButton('Delete Case')
        delete_case_btn.clicked.connect(self.delete_case)
        case_actions.addWidget(delete_case_btn)

        cases_layout.addLayout(case_actions)
        right_layout.addWidget(cases_group)

        main_layout.addWidget(right_panel)

        # set equal widths for both panels
        main_layout.setStretchFactor(left_panel, 1)
        main_layout.setStretchFactor(right_panel, 1)

        # menu bar
        self.create_menu_bar()

    def create_menu_bar(self):
        menubar = self.menuBar()

        # file menu
        file_menu = menubar.addMenu('File')

        import_action = QAction('Import CSV', self)
        import_action.triggered.connect(self.import_csvs)
        file_menu.addAction(import_action)

        file_menu.addSeparator()

        save_cases_action = QAction('Save Cases', self)
        save_cases_action.triggered.connect(self.save_cases)
        file_menu.addAction(save_cases_action)

        load_cases_action = QAction('Load Cases', self)
        load_cases_action.triggered.connect(self.load_cases)
        file_menu.addAction(load_cases_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # view menu
        view_menu = menubar.addMenu('View')

        show_plot_action = QAction('Show Plot View', self)
        show_plot_action.triggered.connect(self.show_plot_view)
        view_menu.addAction(show_plot_action)

        hide_plot_action = QAction('Hide Plot View', self)
        hide_plot_action.triggered.connect(self.hide_plot_view)
        view_menu.addAction(hide_plot_action)

        # plot menu
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

        # cases menu
        cases_menu = menubar.addMenu('Cases')

        create_case_action = QAction('Create Case', self)
        create_case_action.triggered.connect(self.create_case)
        cases_menu.addAction(create_case_action)

        edit_case_action = QAction('Edit Case', self)
        edit_case_action.triggered.connect(self.edit_case)
        cases_menu.addAction(edit_case_action)

        run_case_action = QAction('Run Case',self)
        run_case_action.triggered.connect(self.run_case)
        cases_menu.addAction(run_case_action)

        # options menu
        options_menu = menubar.addMenu('Options')

        set_input_scaling_action = QAction('Input Scaling',self)
        set_input_scaling_action.triggered.connect(self.set_scaling)
        options_menu.addAction(set_input_scaling_action)


    def create_plot_window(self):
        if self.plot_window is None:
            self.plot_window = PlotWindow(self)
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
                name = os.path.splitext(os.path.basename(file))[0]
                self.vectors[name] = df

                item = QListWidgetItem(f"{name}")
                item.setData(Qt.UserRole, name)
                self.list_widget.addItem(item)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading {file}:\n{str(e)}")

    def on_vector_selected(self, item):
        vector_name = item.data(Qt.UserRole)
        if vector_name in self.vectors:
            df = self.vectors[vector_name]
            info_text = f"{vector_name} | Points: {df.shape[0]}"
            self.vector_info_label.setText(info_text)

    def create_case(self):
        if not self.vectors:
            QMessageBox.warning(self, "Warning", "Please load some vectors first")
            return

        available_vectors = list(self.vectors.keys())
        dialog = CaseDialog(self, available_vectors)

        if dialog.exec_() == QDialog.Accepted:
            case_data = dialog.get_case_data()
            case_name = case_data["name"]

            if case_name in self.cases:
                reply = QMessageBox.question(self, "Case Exists",
                                             f"Case '{case_name}' already exists. Overwrite?",
                                             QMessageBox.Yes | QMessageBox.No)
                if reply != QMessageBox.Yes:
                    return

            self.cases[case_name] = case_data
            self.update_cases_list()
            self.save_cases()
            QMessageBox.information(self, "Success", f"Case '{case_name}' created successfully")

    def run_case(self):
        current_item = self.cases_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a case to run")
            return
        case_name = current_item.text()
        case_data = self.cases[case_name]

        input_stack = []
        for inputs in case_data['inputs']:
            if inputs in self.vectors:
                df = self.vectors[inputs].to_numpy().reshape(1, -1)
                input_stack.append(df)
        output_stack = []
        for outputs in case_data['outputs']:
            if outputs in self.vectors:
                df = self.vectors[outputs].to_numpy().reshape(1, -1)
                output_stack.append(df)
        ttss = case_data['ttss']
        final_input = np.vstack(input_stack)
        final_output = np.vstack(output_stack)
        input_tuple = (case_data['inputs'],final_input)
        output_tuple = (case_data['outputs'],final_output)

        newCase = Case(input_tuple,output_tuple,ttss,case_name)
        newCase.runcases()

        self.caserun_list[case_name] = newCase

        newCase.plot_overlaid()


    def run_pred(self):
        current_item = self.cases_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a case to predict")
            return
        case_name = current_item.text()

        self.caserun_list[case_name].caseruns[1].create_predictor()
        self.caserun_list[case_name].caseruns[1].create_predictions()
        #we'll predict the "middle" TTSS, eventually give the option to choose
        #the one to predict

    def set_scaling(self):

        #shouldn't do this
        #should add an MV each time a case is edited/added
        #will re-do it eventually

        inputs = []
        for case_name, case in self.cases.items():
            for mv in case.get("inputs",[]):
                if mv not in inputs:
                    inputs.append(mv)

        if not inputs:
            QMessageBox.information(self, "Input Scaling", "No inputs found")
            return

        # open dialog box with current defaults (self.input_scaling), if any
        dlg = InputScalingDialog(self, vectors=inputs, defaults=self.input_scaling, title="Input Scaling")
        if dlg.exec_() == QDialog.Accepted:
            self.input_scaling = dlg.values()
            QMessageBox.information(self, "Input Scaling", "Scaling updated.")

    def edit_case(self):
        current_item = self.cases_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a case to edit")
            return

        case_name = current_item.text()
        case_data = self.cases[case_name]
        available_vectors = list(self.vectors.keys())

        dialog = CaseDialog(self, available_vectors, case_data)

        if dialog.exec_() == QDialog.Accepted:
            new_case_data = dialog.get_case_data()
            new_name = new_case_data["name"]

            # if name changed, delete old case
            if new_name != case_name:
                del self.cases[case_name]

            self.cases[new_name] = new_case_data
            self.update_cases_list()
            self.save_cases()
            QMessageBox.information(self, "Success", f"Case updated successfully")

    def delete_case(self):
        current_item = self.cases_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a case to delete")
            return

        case_name = current_item.text()
        reply = QMessageBox.question(self, "Delete Case",
                                     f"Delete case '{case_name}'?",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            del self.cases[case_name]
            self.update_cases_list()
            self.save_cases()
            self.case_info_label.setText("Select a case to see details")

    def update_cases_list(self):
        self.cases_list.clear()
        for case_name in sorted(self.cases.keys()):
            self.cases_list.addItem(case_name)

    def plot_case_inputs(self):
        current_item = self.cases_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a case")
            return

        case_name = current_item.text()
        case_data = self.cases[case_name]
        inputs = case_data["inputs"]

        if not inputs:
            QMessageBox.warning(self, "Warning", f"Case '{case_name}' has no input vectors")
            return

        self.plot_vectors_list(inputs, f"Case '{case_name}' Inputs")

    def plot_case_outputs(self):
        current_item = self.cases_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a case")
            return

        case_name = current_item.text()
        case_data = self.cases[case_name]
        outputs = case_data["outputs"]

        if not outputs:
            QMessageBox.warning(self, "Warning", f"Case '{case_name}' has no output vectors")
            return

        self.plot_vectors_list(outputs, f"Case '{case_name}' Outputs")

    def plot_vectors_list(self, vector_names, title_prefix=""):
        # ensure plot window is visible
        if self.plot_window is None:
            self.create_plot_window()
        else:
            self.plot_window.show()
            self.plot_window.raise_()
            self.plot_window.activateWindow()

        # clear existing plot
        self.plot_window.plot_canvas.clear_plot()

        # plot each vector
        for vector_name in vector_names:
            if vector_name in self.vectors:
                df = self.vectors[vector_name]
                try:
                    self.plot_window.plot_canvas.plot_data(df, title=vector_name, clear_first=False)
                except Exception as e:
                    QMessageBox.warning(self, "Plot Warning", f"Could not plot {vector_name}: {str(e)}")

        # update plot info
        self.plot_window.update_plotted_info()

        # update title if provided
        if title_prefix and self.plot_window.plot_canvas.fig.axes:
            ax = self.plot_window.plot_canvas.fig.axes[0]
            current_title = ax.get_title()
            ax.set_title(f"{title_prefix} - {current_title}")
            self.plot_window.plot_canvas.draw()

    def plot_vector_new(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a vector to plot")
            return

        if self.plot_window is None:
            self.create_plot_window()
        else:
            self.plot_window.show()
            self.plot_window.raise_()
            self.plot_window.activateWindow()

        vector_name = current_item.data(Qt.UserRole)
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

        if self.plot_window is None:
            self.create_plot_window()
        else:
            self.plot_window.show()
            self.plot_window.raise_()
            self.plot_window.activateWindow()

        vector_name = current_item.data(Qt.UserRole)

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

        vector_name = current_item.data(Qt.UserRole)

        reply = QMessageBox.question(self, 'Remove Vector',
                                     f'Remove {vector_name}?',
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            del self.vectors[vector_name]

            row = self.list_widget.row(current_item)
            self.list_widget.takeItem(row)

            self.vector_info_label.setText("Select a vector to see details")

            if (self.plot_window and
                    vector_name in self.plot_window.plot_canvas.plotted_vectors):
                self.clear_plot()
                QMessageBox.information(self, "Info", "Plot cleared because plotted vector was removed")

    def save_cases(self):
        try:
            with open(self.cases_file, 'w') as f:
                json.dump(self.cases, f, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not save cases: {str(e)}")

    def load_cases(self):
        try:
            if os.path.exists(self.cases_file):
                with open(self.cases_file, 'r') as f:
                    self.cases = json.load(f)
                self.update_cases_list()
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not load cases: {str(e)}")
            self.cases = {}

    def closeEvent(self, event):
        self.save_cases()  # auto-save cases on exit

        reply = QMessageBox.question(self, 'Exit',
                                     'Are you sure you want to exit?',
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
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
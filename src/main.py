import sys

import numpy as np
from PyQt5.QtWidgets import QApplication

from identification import identification
import matplotlib.pyplot as plt
from UI import CSVManager

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVManager()
    window.show()
    sys.exit(app.exec_())

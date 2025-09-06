import sys

from PyQt5.QtWidgets import QApplication
#from UI import CSVManager
from src.UI import CSVManager

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVManager()
    window.show()
    sys.exit(app.exec_())

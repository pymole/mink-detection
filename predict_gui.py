from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QFileDialog, QFrame, QHBoxLayout
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtGui
import sys
import os

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QFrame, QHBoxLayout, QPushButton
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "PyQt5 Frame"
        self.top = 200
        self.left = 500
        self.width = 400
        self.height = 300
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet('background-color:brown')
        hbox = QHBoxLayout()
        btn1 = QPushButton( "Click Me")
        btn1.setStyleSheet("color:white")
        btn1.setStyleSheet("background-color:green")
        frame = QFrame(self)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setLineWidth(0.6)
        hbox.addWidget(frame)
        hbox.addWidget(btn1)
        self.setLayout(hbox)
        self.show()

    @pyqtSlot()
    def on_click(self):
        pred_dir = QFileDialog.getExistingDirectory(None, "Select directory",
                                                    os.getcwd(),
                                                    QFileDialog.ShowDirsOnly)
        print(pred_dir)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Window()
    sys.exit(app.exec_())

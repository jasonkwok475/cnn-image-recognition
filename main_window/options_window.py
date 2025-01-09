from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class NetworkOptions(QDialog):
  def __init__(self, parent=None):
    super().__init__(parent)

    self.parent = parent

    self.initUi()
    self.exec_()

  def initUi(self):       
    self.setWindowTitle('Network Parameters') 
    self.resize(300, 230)

    self.grid = QGridLayout()
    self.form = QFormLayout()

    self.trainButton = QPushButton("Start Training")
    self.trainButton.clicked.connect(self.train)
    
    self.randomize = QCheckBox()

    self.datasetNum = QLineEdit()
    self.datasetNum.setText('60000')
    self.datasetNum.setReadOnly(True)

    self.epochNum = QLineEdit()
    self.epochNum.setText('1')
    self.epochNum.setInputMask('00')

    self.rateLineEdit = QLineEdit()
    self.rateLineEdit.setText('0.001')

    self.form.addRow(self.tr("&Randomize Dataset:"), self.randomize)
    self.form.addRow(self.tr("&Datasets to Use:"), self.datasetNum)
    self.form.addRow(self.tr("&Epochs:"), self.epochNum)
    self.form.addRow(self.tr("&Learning Rate:"), self.rateLineEdit)

    self.grid.addLayout(self.form, 1, 1)
    self.grid.addWidget(self.trainButton, 2, 1)
    #self.setGeometry(300, 230)

    self.setLayout(self.grid)

  def train(self):
    self.parent.network.init(float(self.rateLineEdit.text()), 10, 20)
    self.parent.trainNetwork(int(self.datasetNum.text()), None, int(self.epochNum.text()))
    self.close()



  

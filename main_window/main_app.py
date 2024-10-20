import asyncio
import logging
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
from structures.network import Network
from keras.datasets import mnist #!Temp

graphPen = pg.mkPen(color=(2, 135, 195), width=2)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

network = Network()

logging.basicConfig(level=logging.INFO)

class MainWindow(QWidget):

  #!Change these names later 
  datasetsAcc = []
  datasetsErr = []
  acc = []
  loss = []

  def __init__(self, parent = None):
    super(MainWindow, self).__init__(parent)

    self.network = network
    self.network.log.connect(self.log)
    self.network.training.connect(self.updateAccPlot)
    self.network.training.connect(self.updateErrorPlot)

    self.resize(200,50)
    self.setWindowTitle("CNN Machine Learning - MNIST")
    self.initUi()

    self.showMaximized()

  def initStructures(self):
    self.network.init(0.001, 10, 20)

  def initUi(self):
    logTextBox = QTextEditLogger(self)
    logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s --- %(message)s'))
    logging.getLogger().addHandler(logTextBox)

    self.grid = QGridLayout()

    self.progress = QProgressBar()
    
    self.grid.setRowStretch(1,2) 
    self.grid.setRowStretch(2,2) 
    self.grid.setColumnStretch(1,1) 
    self.grid.setColumnStretch(2,1) 

    self.loggingGroupBox = QGroupBox("Network Logs")

    self.vboxLog = QVBoxLayout()
    self.vboxLog.addWidget(logTextBox.widget)

    self.loggingGroupBox.setLayout(self.vboxLog)

    self.grid.addWidget(self.generateErrorPlot(), 1, 1)
    self.grid.addWidget(self.generateAccuracyPlot(), 2, 1)
    self.grid.addWidget(QPushButton("A"), 1, 2)
    self.grid.addWidget(self.generateButtonList(), 2, 2)
    self.grid.addWidget(self.loggingGroupBox, 3, 1, 1, 2)

    self.setLayout(self.grid)

    # self.label = QLabel(self)
    # self.label.setText("Hello World")
    # font = QFont()
    # font.setFamily("Arial")
    # font.setPointSize(16)
    # self.label.setFont(font)
    # self.label.move(50,20)


  def generateButtonList(self):
    self.buttonList = QGroupBox("Network Actions")

    self.trainBtn = QPushButton("Train Network")
    self.trainBtn.clicked.connect(self.trainNetwork)

    self.vboxButtons = QVBoxLayout()
    self.vboxButtons.addWidget(self.trainBtn)
    self.vboxButtons.addWidget(self.progress)

    self.progress.hide()

    self.buttonList.setLayout(self.vboxButtons)

    return self.buttonList
    
  def generateErrorPlot(self):
    #Eventually have this plot cross entropy loss against epoch
    self.errorPlot = pg.PlotWidget()
    self.errorPlot.setBackground("w")
    self.errorPlot.setTitle("Model Cross Entropy Loss", color="k")
    self.errorPlot.setLabel("left", "Cross Entropy Loss")
    self.errorPlot.setLabel("bottom", "Datasets")

    self.errorLine = self.errorPlot.plot(self.datasetsErr, self.loss, pen=graphPen)

    return self.errorPlot
  
  def updateErrorPlot(self, datapoints, err, acc):
    self.datasetsErr = self.datasetsErr[0:]
    self.datasetsErr.append(datapoints)
    self.loss = self.loss[0:]
    self.loss.append(err)
    self.errorLine.setData(self.datasetsErr, self.loss)
  
  def generateAccuracyPlot(self):
    #Eventually have this plot accuracy against epoch
    self.accPlot = pg.PlotWidget()
    self.accPlot.setBackground("w")
    self.accPlot.setTitle("Model Accuracy", color="k")
    self.accPlot.setLabel("left", "Accuracy [%]")
    self.accPlot.setLabel("bottom", "Datasets")

    self.accLine = self.accPlot.plot(self.datasetsAcc, self.acc, pen=graphPen)

    return self.accPlot
  
  def updateAccPlot(self, datapoints, err, acc):
    self.datasetsAcc = self.datasetsAcc[0:]
    self.datasetsAcc.append(datapoints)
    self.acc = self.acc[0:]
    self.acc.append(acc)
    self.accLine.setData(self.datasetsAcc, self.acc)
  
  def trainNetwork(self):
    #https://stackoverflow.com/questions/65772946/pyqt5-qmainwindow-freezes-when-calling-a-long-running-function

    self.datasetsAcc = []
    self.datasetsErr = []
    self.acc = []
    self.loss = []

    self.update_progress(0)
    self.trainingThread = QThread()
    self.worker = Worker()
    self.worker.moveToThread(self.trainingThread)

    self.trainingThread.started.connect(self.worker.run)
    self.worker.finished.connect(self.trainingThread.quit)
    self.worker.finished.connect(self.worker.deleteLater)
    self.trainingThread.finished.connect(self.trainingThread.deleteLater)
    self.worker.progress.connect(self.update_progress)

    self.trainingThread.start()
    self.progress.show()
    self.trainBtn.setEnabled(False)

  def update_progress(self, progress):
    self.progress.setValue(progress)
    if (progress == 100):
      self.trainBtn.setEnabled(True)
      self.progress.hide()

  def log(self, str):
    logging.info(str)
     
    
class Worker(QObject):
  finished = pyqtSignal()
  progress = pyqtSignal(int)

  def run(self):
      # Here we pass the update_progress (uncalled!)
      # function to the long_running_function:
      network.train(self.update_progress, x_train, y_train)
      self.finished.emit()

  def update_progress(self, percent):
      self.progress.emit(percent)   


class QTextEditLogger(logging.Handler):
  def __init__(self, parent):
      super().__init__()
      self.widget = QPlainTextEdit(parent)
      self.widget.setReadOnly(True)

  def emit(self, record):
      msg = self.format(record)
      self.widget.appendPlainText(msg)

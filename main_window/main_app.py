import logging
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
from structures.network import Network
from main_window.options_window import NetworkOptions
from handlers.model_handler import ModelHandler
from keras.datasets import mnist #!Temp

graphPen = pg.mkPen(color=(2, 135, 195), width=2)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

network = Network()
trainingParams = dict(datasets=60000,randomize=False,epochs=1)

logging.basicConfig(level=logging.INFO)

#https://forum.qt.io/topic/80053/styling-qgroupbox-in-qt-design/9

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
    self.network.progress.connect(self.updateAccPlot)
    self.network.progress.connect(self.updateErrorPlot)
    self.network.progress.connect(self.updateParameters)

    self.resize(200,50)
    self.setWindowTitle("CNN Machine Learning - MNIST")
    self.initUi()

    self.showMaximized()

    self.models = ModelHandler()

  #def initStructures(self):
  #  self.network.init(0.001, 10, 20)

  def initUi(self):
    logTextBox = QTextEditLogger(self)
    logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s --- %(message)s'))
    logging.getLogger().addHandler(logTextBox)

    self.grid = QGridLayout()

    self.progress = QProgressBar()
    
    self.grid.setRowStretch(1,2) 
    self.grid.setRowStretch(2,2) 
    self.grid.setColumnStretch(1,2) 
    self.grid.setColumnStretch(2,1) 
    self.grid.setColumnStretch(3,1) 

    self.loggingGroupBox = QGroupBox("Network Logs")

    self.vboxLog = QVBoxLayout()
    self.vboxLog.addWidget(logTextBox.widget)

    self.loggingGroupBox.setLayout(self.vboxLog)

    self.grid.addWidget(self.generateErrorPlot(), 1, 1)
    self.grid.addWidget(self.generateAccuracyPlot(), 2, 1)
    self.grid.addWidget(self.generateButtonList(), 2, 3)
    self.grid.addWidget(self.generateNetworkData(), 2, 2)
    self.grid.addWidget(self.loggingGroupBox, 3, 1, 1, 3)

    self.setLayout(self.grid)

    # self.label = QLabel(self)
    # self.label.setText("Hello World")
    # font = QFont()
    # font.setFamily("Arial")
    # font.setPointSize(16)
    # self.label.setFont(font)
    # self.label.move(50,20)

  def generateNetworkData(self):
    self.dataList = QGroupBox("Network Parameters")

    self.dataFormList = QFormLayout()

    self.trainedDatasetsNum = 0
    self.trainedDatasets = QLabel(str(self.trainedDatasetsNum))
    self.epochsNum = 0
    self.epochs = QLabel(str(self.epochsNum))
    self.highestAccuracyNum = 0
    self.highestAccuracy = QLabel(str(self.highestAccuracyNum))
    self.avgAccuracyNum = 0
    self.avgAccuracy = QLabel(str(self.avgAccuracyNum))
    self.avgLossNum = 0
    self.avgLoss = QLabel(str(self.avgLossNum))

    self.dataFormList.addRow(self.tr("&Datasets Trained:"), self.trainedDatasets)
    self.dataFormList.addRow(self.tr("&Epochs:"), self.epochs)
    self.dataFormList.addRow(self.tr("&Highest Accuracy:"), self.highestAccuracy)
    self.dataFormList.addRow(self.tr("&Avg. Accuracy:"), self.avgAccuracy)
    self.dataFormList.addRow(self.tr("&Avg. Loss:"), self.avgLoss)
    self.dataFormList.addRow(self.tr("&Progress:"), self.progress)

    self.dataList.setLayout(self.dataFormList)

    return self.dataList

  def generateButtonList(self):
    self.buttonList = QGroupBox("Network Actions")

    self.trainBtn = QPushButton("Train Network")
    self.trainBtn.clicked.connect(self.newOptionsWindow)#self.trainNetwork)

    self.vboxButtons = QVBoxLayout()
    self.vboxButtons.addWidget(self.trainBtn)
    self.vboxButtons.addWidget(self.progress)

    self.progress.hide()

    self.buttonList.setLayout(self.vboxButtons)

    return self.buttonList
    
  def updateParameters(self, datapoints, err, acc):
    self.trainedDatasetsNum += datapoints
    self.trainedDatasets.setText(str(self.trainedDatasetsNum))
    #self.epochsNum = 0
    #self.epochs = QLabel(str(self.epochsNum))
    self.avgAccuracyNum = acc
    self.avgAccuracy.setText(str(self.avgAccuracyNum) + "%") #Add percent symbol
    self.avgLossNum = err
    self.avgLoss.setText(str(self.avgLossNum))  

    if acc > self.highestAccuracyNum: 
      self.highestAccuracyNum = acc 
      self.highestAccuracy.setText(str(self.highestAccuracyNum) + "%")

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
    self.datasetsErr.append(datapoints if len(self.datasetsErr) == 0 else self.datasetsErr[-1] + datapoints)

    self.loss = self.loss[0:]
    self.loss.append(err)
    self.errorLine.setData(self.datasetsErr, self.loss)
  
#https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python

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
    self.datasetsAcc.append(datapoints if len(self.datasetsAcc) == 0 else self.datasetsAcc[-1] + datapoints)

    self.acc = self.acc[0:]
    self.acc.append(acc)
    self.accLine.setData(self.datasetsAcc, self.acc)
  
  def trainNetwork(self, datasets, randomize, epochs):
    #https://stackoverflow.com/questions/65772946/pyqt5-qmainwindow-freezes-when-calling-a-long-running-function

    self.datasetsAcc = []
    self.datasetsErr = []
    self.acc = []
    self.loss = []

    self.update_progress(0)
    self.trainingThread = QThread()
    self.trainingWorker = TrainingWorker()
    self.trainingWorker.moveToThread(self.trainingThread)

    self.trainingThread.started.connect(self.trainingWorker.run)
    self.trainingWorker.finished.connect(self.trainingThread.quit)
    self.trainingWorker.finished.connect(self.trainingWorker.deleteLater)
    self.trainingThread.finished.connect(self.trainingThread.deleteLater)
    self.trainingWorker.progress.connect(self.update_progress)

    self.trainingThread.start()
    self.progress.show()
    self.trainBtn.setEnabled(False)

  def update_progress(self, progress):
    self.progress.setValue(progress)
    if (progress == 100):

      #! Have the UI open a dialog box asking whether or not to save the model data

      self.trainBtn.setEnabled(True)
      self.progress.hide()

  def log(self, str):
    logging.info(str)

  def newOptionsWindow(self):
    self.optionsWindow = NetworkOptions(self)
     
    
class TrainingWorker(QObject):
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

import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from xlwt import Workbook
import xlsxwriter
from xlrd import open_workbook
from RBF import *
from Segmentation import *
from FeatureExtraction import *


class RBF_inputs():
    def __init__(self):
        self.k = None
        self.mse = None
        self.eta = None
        self.epochs = None


def start_window():
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QWidget()
    generate_features_button = QtWidgets.QPushButton(window)
    MLP_button = QtWidgets.QPushButton(window)
    RBF_button = QtWidgets.QPushButton(window)
    window.setWindowTitle('Object Recognition')
    window.setGeometry(500, 100, 500, 360)
    generate_features_button.setText('Generate Features')
    generate_features_button.resize(400, 100)
    generate_features_button.move(50, 20)
    MLP_button.setText('MLP')
    MLP_button.resize(400, 100)
    MLP_button.move(50, 130)
    RBF_button.setText('RBF')
    RBF_button.resize(400, 100)
    RBF_button.move(50, 240)
    window.show()
    generate_features_button.clicked.connect(PCA_Window)
    RBF_button.clicked.connect(RBF_Window)
    app.exec()


def PCA_Window():
    window = QtWidgets.QDialog()
    PCA_Button = QtWidgets.QPushButton(window)
    GHA_Button = QtWidgets.QPushButton(window)
    window.setWindowTitle('PCA Features')
    window.setGeometry(600, 100, 300, 260)
    PCA_Button.setText('Generate PCA Features')
    PCA_Button.resize(200, 100)
    PCA_Button.move(50, 20)
    GHA_Button.setText('Generate PCA(GHA) Features')
    GHA_Button.resize(200, 100)
    GHA_Button.move(50, 130)
    window.show()
    window.exec()


def MLP_Window():
    window = QtWidgets.QDialog()
    PCA_Button = QtWidgets.QPushButton(window)
    GHA_Button = QtWidgets.QPushButton(window)
    window.setWindowTitle('PCA Features')
    window.setGeometry(500, 100, 500, 360)
    PCA_Button.setText('Generate PCA Features')
    PCA_Button.resize(400, 100)
    PCA_Button.move(50, 20)
    GHA_Button.setText('Generate PCA(GHA) Features')
    GHA_Button.resize(400, 100)
    GHA_Button.move(50, 130)
    window.show()
    window.exec()


class RBF_Window(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.title = 'RBF Classifier'
        self.left = 350
        self.top = 50
        self.width = 800
        self.height = 660
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        label1 = QtWidgets.QLabel(self)
        label1.setText('Number of hidden layers')
        label1.setFont(QtGui.QFont("Times", 20, QtGui.QFont.Bold))
        label1.move(20, 50)
        label2 = QtWidgets.QLabel(self)
        label2.setText('MSE threshold')
        label2.setFont(QtGui.QFont("Times", 20, QtGui.QFont.Bold))
        label2.move(20, 110)
        label3 = QtWidgets.QLabel(self)
        label3.setText('Learning rate')
        label3.setFont(QtGui.QFont("Times", 20, QtGui.QFont.Bold))
        label3.move(20, 170)
        label4 = QtWidgets.QLabel(self)
        label4.setText('Number of epochs')
        label4.setFont(QtGui.QFont("Times", 20, QtGui.QFont.Bold))
        label4.move(20, 230)
        self.txtbox1 = QtWidgets.QLineEdit(self)
        self.txtbox1.setFont(QtGui.QFont("Times", 15, QtGui.QFont.Bold))
        self.txtbox1.resize(300, 50)
        self.txtbox1.move(490, 40)
        self.txtbox2 = QtWidgets.QLineEdit(self)
        self.txtbox2.setFont(QtGui.QFont("Times", 15, QtGui.QFont.Bold))
        self.txtbox2.resize(300, 50)
        self.txtbox2.move(490, 100)
        self.txtbox3 = QtWidgets.QLineEdit(self)
        self.txtbox3.setFont(QtGui.QFont("Times", 15, QtGui.QFont.Bold))
        self.txtbox3.resize(300, 50)
        self.txtbox3.move(490, 160)
        self.txtbox4 = QtWidgets.QLineEdit(self)
        self.txtbox4.setFont(QtGui.QFont("Times", 15, QtGui.QFont.Bold))
        self.txtbox4.resize(300, 50)
        self.txtbox4.move(490, 220)
        '''rbf_inputs.k = int(txtbox1.text())
        rbf_inputs.mse = float(txtbox2.text())
        rbf_inputs.eta = float(txtbox3.text())
        rbf_inputs.epochs = int(txtbox4.text())'''
        button1 = QtWidgets.QPushButton(self)
        button1.setText('Train to get weigths')
        button1.setFont(QtGui.QFont("Times", 13, QtGui.QFont.Bold))
        button1.resize(250, 150)
        button1.move(20, 300)
        button1.clicked.connect(self.train)
        button2 = QtWidgets.QPushButton(self)
        button2.setText('Test to get confusion matrix')
        button2.setFont(QtGui.QFont("Times", 13, QtGui.QFont.Bold))
        button2.resize(250, 150)
        button2.move(530, 300)
        button2.clicked.connect(self.test)
        button3 = QtWidgets.QPushButton(self)
        button3.setText('Classify')
        button3.setFont(QtGui.QFont("Times", 13, QtGui.QFont.Bold))
        button3.resize(250, 150)
        button3.move(276, 480)
        button3.clicked.connect(Classify_window)
        self.show()
        self.exec()

    def train(self):
        rbf_inputs = RBF_inputs()
        rbf_inputs.k = int(self.txtbox1.text())
        rbf_inputs.mse = float(self.txtbox2.text())
        rbf_inputs.eta = float(self.txtbox3.text())
        rbf_inputs.epochs = int(self.txtbox4.text())
        file = open_workbook('ObjectRecognition.xls')
        testing_sheet = file.sheet_by_name("Testing")
        training_sheet = file.sheet_by_name("Training")
        x = rbf(rbf_inputs.k, training_sheet, testing_sheet)
        x.initial_values(rbf_inputs.mse, rbf_inputs.eta, rbf_inputs.epochs)
        x.train()
        print(x.weights)

    def test(self):
        rbf_inputs = RBF_inputs()
        rbf_inputs.k = int(self.txtbox1.text())
        rbf_inputs.mse = float(self.txtbox2.text())
        rbf_inputs.eta = float(self.txtbox3.text())
        rbf_inputs.epochs = int(self.txtbox4.text())
        file = open_workbook('ObjectRecognition.xls')
        testing_sheet = file.sheet_by_name("Testing")
        training_sheet = file.sheet_by_name("Training")
        x = rbf(rbf_inputs.k, training_sheet, testing_sheet)
        x.initial_values(rbf_inputs.mse, rbf_inputs.eta, rbf_inputs.epochs)
        x.train()
        a = x.test()
        print(a)


class Classify_window(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.title = 'Classify'
        self.left = 450
        self.top = 50
        self.width = 600
        self.height = 500
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.label1 = QtWidgets.QLabel(self)
        self.label1.setPixmap(QtGui.QPixmap('white.jpg'))
        self.label1.resize(275, 250)
        self.label1.move(20, 20)
        self.label2 = QtWidgets.QLabel(self)
        self.label2.setPixmap(QtGui.QPixmap('white.jpg'))
        self.label2.resize(275, 250)
        self.label2.move(310, 20)
        button1 = QtWidgets.QPushButton(self)
        button1.setText('Load original image')
        button1.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        button1.resize(200, 50)
        button1.move(50, 300)
        button1.clicked.connect(self.Browse_label1)
        button2 = QtWidgets.QPushButton(self)
        button2.setText('Load segmented image')
        button2.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        button2.resize(200, 50)
        button2.move(350, 300)
        button2.clicked.connect(self.Browse_label2)
        button3 = QtWidgets.QPushButton(self)
        button3.setText('Recognize')
        button3.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Bold))
        button3.resize(150, 100)
        button3.move(230, 400)
        button3.clicked.connect(self.Recognize)
        self.show()
        self.exec()

    def Browse_label1(self):
        window = QtWidgets.QDialog()
        self.original_path, _ = QtWidgets.QFileDialog.getOpenFileName(window, 'Single File', QtCore.QDir.rootPath(),
                                                                      '*')
        if len(self.original_path) > 0:
            self.label1.setPixmap(QtGui.QPixmap(self.original_path))
            self.label1.resize(275, 250)
            self.label1.move(20, 20)

    def Browse_label2(self):
        window = QtWidgets.QDialog()
        self.segmented_path, _ = QtWidgets.QFileDialog.getOpenFileName(window, 'Single File', QtCore.QDir.rootPath(),
                                                                       '*')
        if len(self.segmented_path) > 0:
            self.label2.setPixmap(QtGui.QPixmap(self.segmented_path))
            self.label2.resize(275, 250)
            self.label2.move(310, 20)

    def Recognize(self):
        pca = PCA(5)
        pca.ReadImages('Training')
        mean = pca.ImagesMean()
        pca.Training(mean)
        rbf_inputs = RBF_inputs()
        rbf_inputs.k = 11
        rbf_inputs.mse = .001
        rbf_inputs.eta = .9
        rbf_inputs.epochs = 16
        file = open_workbook('ObjectRecognition.xls')
        print(1)
        training_sheet = file.sheet_by_name("Training")
        testing_sheet = file.sheet_by_name("Testing")
        print(3)
        x = rbf(rbf_inputs.k, training_sheet, testing_sheet)
        x.initial_values(rbf_inputs.mse, rbf_inputs.eta, rbf_inputs.epochs)
        x.train()
        print(2)
        original_image = cv2.imread(self.original_path)
        output_image=original_image
        getimages = Segmentation()
        Coordinates_images = Segmentation.Segment(self.segmented_path)
        for coordinate in Coordinates_images:
            crop_img = original_image[coordinate.y:coordinate.y + coordinate.h,
                       coordinate.x:coordinate.x + coordinate.w]
            image = cv2.imread(crop_img)
            image = cv2.resize(crop_img, (50, 50))
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            gray = np.reshape(crop_img, 2500)
            feautures = pca.TestingOne(gray)
            type = x.classify(feautures)
            output_image=add_rectangle(output_image,coordinate.x,coordinate.y,coordinate.x+coordinate.w,coordinate.y+coordinate.h,type)
        cv2.imshow('IMAGE',output_image)

def type(k):
    if k == 0:
        return 'Cat'
    elif k == 1:
        return 'Laptop'
    elif k == 2:
        return 'Apple'
    elif k == 3:
        return 'Car'
    elif k == 4:
        return 'Helicopter'


def add_rectangle(img,minx,miny,maxx,maxy,k):
    a=img
    cv2.rectangle(a, (minx, miny), (maxx, maxy), (0, 255, 0), 2)  # (startX , startY) (endX , endY) (R,G,B) (rectangle thickness)
    font = cv2.FONT_HERSHEY_PLAIN
#    text posX,posY,  , size, color,thick, lineType
    cv2.putText(a, type(k), (minx, miny), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
    return a
start_window()

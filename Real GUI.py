import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from xlwt import Workbook
import xlsxwriter
from xlrd import open_workbook
from RBF import *
from Segmentation import *
from FeatureExtraction import *
from MLP import *


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
    generate_features_button.clicked.connect(Do_graph)
    MLP_button.clicked.connect(MLP_Window)
    RBF_button.clicked.connect(RBF_Window)
    app.exec()

def Do_graph():
    pca = PCA(5)
    pca.ReadImages('Training')
    mean = pca.ImagesMean()
    pca.Training(mean)
    pca.ReadImages('Testing')
    pca.Testing()


'''class PCA_Window(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.title = 'Generate Features'
        self.left = 600
        self.top = 100
        self.width = 300
        self.height = 260
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setGeometry(600, 100, 300, 260)
        PCA_Button = QtWidgets.QPushButton(self)
        GHA_Button = QtWidgets.QPushButton(self)
        PCA_Button.setText('Generate PCA Features')
        PCA_Button.resize(200, 100)
        PCA_Button.move(50, 20)
        PCA_Button.clicked.connect(self.Do_graph)
        GHA_Button.setText('Generate PCA(GHA) Features')
        GHA_Button.resize(200, 100)
        GHA_Button.move(50, 130)
        self.show()
        self.exec()'''


class MLP_Window(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.title = 'MLP Classifier'
        self.left = 500
        self.top = 200
        self.width = 500
        self.height = 230
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        button2 = QtWidgets.QPushButton(self)
        button2.setText('Test With Best')
        button2.setFont(QtGui.QFont("Times", 13, QtGui.QFont.Bold))
        button2.resize(200, 100)
        button2.move(280, 70)
        button2.clicked.connect(self.test)
        button3 = QtWidgets.QPushButton(self)
        button3.setText('Classify')
        button3.setFont(QtGui.QFont("Times", 13, QtGui.QFont.Bold))
        button3.resize(200, 100)
        button3.move(30, 70)
        button3.clicked.connect(Classify_window_mlp)
        self.show()
        self.exec()

    def test(self):
        multi = MLP()
        multi.read_excel('ObjectRecognition2.xls')
        multi.read_xml("MLP.xml")
        acc, conf = multi.test()
        print(acc)
        print(conf)
        self.do_message()

    def do_message(self):
        Mbox = QtWidgets.QMessageBox(self)
        Mbox.setText("DONE")
        Mbox.resize(400, 300)
        Mbox.move(700, 300)
        Mbox.show()
        Mbox.exec()


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
        button1.setText('Train to get weights')
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
        button3.clicked.connect(Classify_window_rbf)
        self.show()
        self.exec()

    def train(self):
        rbf_inputs = RBF_inputs()
        rbf_inputs.k = int(self.txtbox1.text())
        rbf_inputs.mse = float(self.txtbox2.text())
        rbf_inputs.eta = float(self.txtbox3.text())
        rbf_inputs.epochs = int(self.txtbox4.text())
        file = open_workbook('ObjectRecognition2.xls')
        testing_sheet = file.sheet_by_name("Testing")
        training_sheet = file.sheet_by_name("Training")
        x = rbf(rbf_inputs.k, training_sheet, testing_sheet)
        x.initial_values(rbf_inputs.mse, rbf_inputs.eta, rbf_inputs.epochs)
        x.train()
        print(x.weights)
        self.do_message()

    def test(self):
        rbf_inputs = RBF_inputs()
        rbf_inputs.k = int(self.txtbox1.text())
        rbf_inputs.mse = float(self.txtbox2.text())
        rbf_inputs.eta = float(self.txtbox3.text())
        rbf_inputs.epochs = int(self.txtbox4.text())
        file = open_workbook('ObjectRecognition2.xls')
        testing_sheet = file.sheet_by_name("Testing")
        training_sheet = file.sheet_by_name("Training")
        x = rbf(rbf_inputs.k, training_sheet, testing_sheet)
        x.initial_values(rbf_inputs.mse, rbf_inputs.eta, rbf_inputs.epochs)
        x.train()
        a = x.test()
        print(a)
        self.do_message()

    def do_message(self):
        Mbox = QtWidgets.QMessageBox(self)
        Mbox.setText("DONE")
        Mbox.resize(400, 300)
        Mbox.move(700, 300)
        Mbox.show()
        Mbox.exec()


class Classify_window_rbf(QtWidgets.QDialog):
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
        original_image = cv2.imread(self.original_path)
        output_image = cv2.imread(self.original_path)
        getimages = Segmentation()
        Coordinates_images = getimages.Segment(self.segmented_path)
        r = read_rbf_data()
        for coordinate in Coordinates_images:
            crop_img = original_image[coordinate.y:coordinate.y + coordinate.h,
                       coordinate.x:coordinate.x + coordinate.w]
            image = cv2.resize(crop_img, (50, 50))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = np.reshape(gray, 2500)
            feautures = generante_features_sample(gray, pca_data.totalimages, pca_data.finalEigenVectors)
            type = classify_from_file(feautures, r.k, r.num_classes, r.avg_list, r.mx_list, r.mn_list, r.centers,
                                      r.weights)
            output_image = add_rectangle(output_image, coordinate.x, coordinate.y, coordinate.x + coordinate.w,
                                         coordinate.y + coordinate.h, type)
        cv2.imshow('IMAGE', output_image)
        cv2.waitKey(400000)


class Classify_window_mlp(QtWidgets.QDialog):
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
        multi = MLP()
        multi.read_xml("MLP.xml")
        original_image = cv2.imread(self.original_path)
        output_image = cv2.imread(self.original_path)
        getimages = Segmentation()
        Coordinates_images = getimages.Segment(self.segmented_path)
        for coordinate in Coordinates_images:
            crop_img = original_image[coordinate.y:coordinate.y + coordinate.h,
                       coordinate.x:coordinate.x + coordinate.w]
            image = cv2.resize(crop_img, (50, 50))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = np.reshape(gray, 2500)
            feautures = generante_features_sample(gray, pca_data.totalimages, pca_data.finalEigenVectors)
            type = multi.determine_class(np.asarray(feautures)) - 1
            output_image = add_rectangle(output_image, coordinate.x, coordinate.y, coordinate.x + coordinate.w,
                                         coordinate.y + coordinate.h, type)
        cv2.imshow('IMAGE', output_image)
        cv2.waitKey(400000)


class read_rbf_data():
    def __init__(self):
        self.k = 13
        self.num_classes = 5
        self.avg_list = []
        self.mx_list = []
        self.mn_list = []
        self.centers = []
        self.weights = []
        center = []
        weight = []
        file = open("DataforRBF.txt", "r")
        text = file.read()
        i = 0
        j = 0
        tmp = ""
        while (i < len(text)):
            if text[i] != ' ' and text[i] != '\n':
                tmp += text[i]
            if (text[i] == '\n'):
                if j >= 3 and j <= 15:
                    self.centers.append(center)
                    center = []
                if j >= 16 and j <= 20:
                    self.weights.append(weight)
                    weight = []
                j += 1

            elif j == 0 and text[i] == ' ':
                if tmp != ' ':
                    self.avg_list.append(float(tmp))
                tmp = ""
            elif j == 1 and text[i] == ' ':
                if tmp != ' ':
                    self.mx_list.append(float(tmp))
                tmp = ""
            elif j == 2 and text[i] == ' ':
                if tmp != ' ':
                    self.mn_list.append(float(tmp))
                tmp = ""
            elif j >= 3 and j <= 15 and text[i] == ' ':
                if tmp != ' ':
                    center.append(float(tmp))
                tmp = ""
            elif j >= 16 and j <= 20 and text[i] == ' ':
                if tmp != ' ':
                    weight.append(float(tmp))
                tmp = ""
            i += 1


class read_pca_data():
    def __init__(self):
        self.totalimages = []
        self.finalEigenVectors = []
        Vector = []
        file = open("DataPCA.txt", "r")
        text = file.read()
        i = 0
        j = 0
        tmp = ""
        while (i < len(text)):
            if text[i] != ' ' and text[i] != '\n':
                tmp += text[i]
            if (text[i] == '\n'):
                if j >= 1:
                    self.finalEigenVectors.append(Vector)
                    Vector = []
                j += 1
            elif j == 0 and text[i] == ' ':
                if tmp != ' ':
                    self.totalimages.append(float(tmp))
                tmp = ""
            elif j >= 1 and text[i] == ' ':
                if tmp != ' ':
                    Vector.append(float(tmp))
                tmp = ""
            i += 1


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


def add_rectangle(img, minx, miny, maxx, maxy, k):
    a = img
    cv2.rectangle(a, (minx, miny), (maxx, maxy), (0, 255, 0),
                  2)  # (startX , startY) (endX , endY) (R,G,B) (rectangle thickness)
    font = cv2.FONT_HERSHEY_PLAIN
    #    text posX,posY,  , size, color,thick, lineType
    cv2.putText(a, type(k), (minx, miny + 20), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
    return a


pca_data = read_pca_data()
start_window()

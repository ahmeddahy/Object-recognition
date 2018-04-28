import numpy as np
import glob
import cv2
import random
import xlwt
import matplotlib.pyplot as plt
import math


def generante_features_sample(image,totalimages,finalEigenVectors):
    mean = image - totalimages
    FinalData = np.dot(np.array(finalEigenVectors).T, mean)
    FinalData = FinalData.T
    return FinalData


class PCA:
    def __init__(self, NumberOfClasses):
        self.NumberOfClasses = NumberOfClasses
        self.Images = []
        self.totalimages = [0.0 for i in range(0, 50, 1) for j in range(0, 50, 1)]
        self.wbk = xlwt.Workbook()

    def ReadImages(self, operation):
        self.Images = []
        for Class in range(0, self.NumberOfClasses, 1):
            if (operation == "Training"):
                path = operation + "/" + str(Class + 1) + "/*.jpg"
            else:
                path = operation + "/" + str(Class + 1) + "/*.bmp"
            for Image in glob.glob(path):
                image = cv2.imread(Image)
                image = cv2.resize(image, (50, 50))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = np.reshape(gray, 2500)
                if (operation == 'Training'):
                    self.totalimages = self.totalimages + gray
                self.Images.append([Class + 1, np.double(gray)])
        # self.Images=np.array(self.Images).T
        self.totalimages = np.double(self.totalimages)

    def ImagesMean(self):
        # self.Images=[[image[np.double(image[1]) for image in self.Images]
        mean = [0 for i in range(0, len(self.Images[0][1]), 1)]
        mx = [0 for i in range(0, len(self.Images[0][1]), 1)]
        out = [[0.0 for x in range(0, len(self.Images), 1)] for y in range(0, len(self.Images[0][1]), 1)]
        for i in range(0, len(self.Images[0][1]), 1):
            q = 0.0;
            for j in range(0, len(self.Images), 1):
                q = q + self.Images[j][1][i]
            mean[i] = (q / len(self.Images))
        for i in range(0, len(self.Images[0][1]), 1):
            tmp = -1e9
            for j in range(0, len(self.Images), 1):
                out[i][j] = self.Images[j][1][i] - mean[i]
                if (out[i][j] > tmp):
                    tmp = out[i][j]
            mx[i] = tmp
            """""
            % for i = 1:h % Normalization
            % for j = 1:w
            % out(i, j) = out(i, j) / mx(i);
            % end
            % end
            """""
        return out

    def Training(self, meanAdjustedDataset):
        CovarianceMatrix = np.cov(meanAdjustedDataset)
        meanAdjustedDataset = np.array(meanAdjustedDataset).T
        # CovarianceMatrix=np.dot(np.array(meanAdjustedDataset).T*meanAdjustedDataset)
        [eigenValue, eigenVector] = np.linalg.eig(CovarianceMatrix)
        # eigenvaluediagonal=np.diag(eigenValue)
        eigenvaluediagonal = eigenValue
        # sortedeigenvaluediagonal=sorted(eigenvaluediagonal,reverse=True)
        eigens = []
        eigenVector = np.real(eigenVector)
        # print(eigenValue)
        # print(eigenVector)
        print("----------------------------")
        idx = eigenvaluediagonal.argsort()[::-1]
        sortedeigenvaluediagonal = eigenvaluediagonal[idx]
        sortedeigenvector = eigenVector[:, idx]
        print(sortedeigenvector)
        ''''
        for i in range(0,len(eigenValue),1):
         eigens.append(tuple([eigenvaluediagonal[i],eigenVector[i]]))
        #print(eigens)
        sortedeigens=sorted(eigens, key=lambda eigens:(eigens[0]),reverse=True)
        sortedeigenvector=[i[1] for i in sortedeigens]
        '''
        self.finalEigenVectors = self.getPCs(sortedeigenvaluediagonal, sortedeigenvector)
        sheet = self.wbk.add_sheet('EigenVector')
        print(self.finalEigenVectors)
        for i in range(0, len(self.finalEigenVectors), 1):
            for j in range(0, len(self.finalEigenVectors[0]), 1):
                sheet.write(i, j, self.finalEigenVectors[i][j])
        sheet = self.wbk.add_sheet('TrainingImagesMean')
        self.totalimages = self.totalimages / len(self.Images)
        print(len(self.totalimages))
        for i in range(0, len(self.totalimages), 1):
            sheet.write(i, 0, self.totalimages[i])
        FinalData = np.dot(np.array(self.finalEigenVectors).T, np.array(meanAdjustedDataset).T)
        FinalData = FinalData.T
        sheet = self.wbk.add_sheet('Training')
        for i in range(0, len(FinalData), 1):
            sheet.write(i, 0, self.Images[i][0])
            for j in range(0, len(FinalData[0]), 1):
                sheet.write(i, j + 1, FinalData[i][j])
        self.wbk.save("ObjectRecognition.xls")
        FinalData = []
        file = open("DataPCA.txt", "w")
        for i in self.totalimages:
            a = str(i)
            file.write(a)
            file.write(" ")
        file.write("\n")
        for i in self.finalEigenVectors:
            for j in i:
                a = str(j)
                file.write(a)
                file.write(" ")
            file.write("\n")
        file.close()
        return FinalData

    def getPCs(self, ediagonal, orderedVectors):
        ediagonal = np.double(ediagonal);
        # print(ediagonal)
        print(np.sum(ediagonal))
        CumProb = np.cumsum(ediagonal / np.sum(ediagonal))
        out = [[] for j in range(0, len(orderedVectors), 1)]
        j = 0
        y = []
        x = [i + 1 for i in range(len(CumProb))]
        while (j < len(CumProb)):
            if (CumProb[j] > 0.97):
                break
            for i in range(0, len(orderedVectors), 1):
                out[i].append(orderedVectors[i][j])
            j = j + 1
        plt.plot(x, CumProb)
        plt.xlabel("PCs")
        plt.ylabel("CumProb")
        plt.show()
        return out

    def Testing(self):
        TestingFeatures = []
        for image in self.Images:
            TestingFeatures.append(self.TestingOne(image[1]))
        sheet = self.wbk.add_sheet('Testing')
        TestingFeatures = np.array(TestingFeatures).tolist()
        for i in range(0, len(TestingFeatures), 1):
            sheet.write(i, 0, self.Images[i][0])
            for j in range(0, len(TestingFeatures[0]), 1):
                sheet.write(i, j + 1, TestingFeatures[i][j])
        self.wbk.save("ObjectRecognition.xls")
        return TestingFeatures

    def TestingOne(self, image):
        mean = image - self.totalimages
        FinalData = np.dot(np.array(self.finalEigenVectors).T, mean)
        FinalData = FinalData.T
        return FinalData

    def SetData(self, EigenVectors, ImagesMean):
        self.finalEigenVectors = EigenVectors.copy()
        self.totalimages = ImagesMean.copy()


class Node:
    def __init__(self, input):
        self.input = input.copy()
        self.output = [0 for i in range(20)]


class GeneralHebbianAlgorithm:
    def __init__(self):
        self.OldWeight = [[random.random() for i in range(2500)] for j in range(20)]
        self.weights = [[0 for i in range(2500)] for j in range(20)]
        self.weightperepoch = [[0 for i in range(2500)] for j in range(20)]

    def calculate(self, i, j, PCs):
        sum = 0.0
        for k in range(i + 1):
            sum = sum + self.OldWeight[k][j] * PCs[k]
        # print(sum)
        # print("--------------")
        return sum

    def Stopping(self):
        for i in range(len(self.OldWeight)):
            for j in range(len(self.OldWeight[0])):
                if ((self.weightperepoch[i][j] - self.weights[i][j]) != 0):
                    return False
        return True

    def Normalize(self, weights):
        for i in range(len(weights)):
            avg = 0
            mx = 0
            mn = 1e9
            for j in range(len(weights[0])):
                avg = avg + weights[i][j]
                if (weights[i][j] > mx):
                    mx = weights[i][j]
                if (weights[i][j] < mn):
                    mn = weights[i][j]
            avg = avg / len(weights[0])
            for j in range(len(weights[0])):
                weights[i][j] = (weights[i][j] - avg) / (mx - mn)
        return weights

    def Network(self, network, learningrate):
        self.OldWeight = self.Normalize(self.OldWeight)
        epoch = 100
        while (epoch > 0):
            print(epoch)
            for n in range(0, len(network), 1):
                for i in range(0, len(network[0].output), 1):
                    network[n].output[i] = 0
                    for j in range(0, len(network[0].input), 1):
                        network[n].output[i] = (network[n].output[i] + self.OldWeight[i][j] * network[n].input[
                            j])  # %255 # calc output
                    for j in range(0, len(network[0].input), 1):
                        self.weights[i][j] = network[n].output[i] * network[n].input[j] - network[n].output[i] * \
                                             self.weights[i][j]
                        if (i != 0):
                            self.weights[i][j] = self.weights[i][j] - (network[n].output[i] * self.weights[i - 1][j])
                        self.weights[i][j] = self.weights[i][j] * learningrate
                self.OldWeight = self.Normalize(self.weights).copy()
            if (self.Stopping()):
                break
            self.weightperepoch = self.weights.copy()
            # print(self.weightperepoch)
            epoch = epoch - 1
        return self.OldWeight

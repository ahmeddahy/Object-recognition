from random import *
import math
import numpy as np
from xlwt import Workbook
import xlsxwriter
from xlrd import open_workbook


def euclidean_distance(point1, point2):
    dist = 0
    for i in range(0, len(point1)):
        dist += ((point1[i] - point2[i]) * (point1[i] - point2[i]))
    return dist


def check_stopping(current_centers, previous_centers):
    for i in range(0, len(current_centers)):
        if (euclidean_distance(current_centers[i], previous_centers[i]) > .0001):
            return 0
    return 1


def calculate_new_centers(cluster_data):
    centers = []
    for data in cluster_data:
        center = []
        for j in range(0, len(data[0])):
            tmp = 0
            for i in range(0, len(data)):
                tmp += data[i][j]
            tmp /= len(data)
            center.append(tmp)
        centers.append(center)
    return centers


def multipky_two_vectors(vec1, vec2):
    res = 0
    for i in range(0, len(vec1)):
        res += (vec1[i] * vec2[i])
    return res


def add_two_vectors(vec1, vec2):
    for i in range(0, len(vec1)):
        vec1[i] += vec2[i]
    return vec1


def multiply_const_vector(const, vec):
    for i in range(0, len(vec)):
        vec[i] *= const
    return vec


def get_average(vec):
    res = 0
    for i in vec:
        res += i
    res /= (len(vec))
    return res


def classify_from_file(sample,k,num_classes,avg_list,mx_list,mn_list,centers,weights):

    for j in range(0, len(sample)):
        sample[j] = ((sample[j] - avg_list[j]) / (mx_list[j] - mn_list[j]))
    maxdistance = 0
    for i in centers:
        for j in centers:
            maxdistance = max(euclidean_distance(i, j), maxdistance)
    spread = maxdistance / math.sqrt(2 * k)
    hidden_layer = []
    for j in range(0, k):
        dist_x_c = euclidean_distance(sample, centers[j])
        hidden_layer.append(math.exp((-1 * dist_x_c * dist_x_c) / (2 * spread * spread)))
    output_layer = []
    for i in range(0, num_classes):
        output_layer.append(multipky_two_vectors(hidden_layer, weights[i]))
    clas = -1
    mx = -1e25
    for i in range(0, num_classes):
        if (output_layer[i] > mx):
            mx = output_layer[i]
            clas = i
    return clas


class rbf:
    def __init__(self, ck, training_sheet, testing_sheet):
        self.k = ck
        #################load training data and do K-mean
        self.training_data = []
        self.train_data_kmean = []
        rows = training_sheet.nrows
        cols = training_sheet.ncols
        self.avg_list = []
        self.mn_list = []
        self.mx_list = []
        for i in range(0, rows):
            sample_features = []
            sample_features_with_class = []
            for j in range(0, cols):
                if j > 0:
                    sample_features.append(training_sheet.cell_value(i, j))
                sample_features_with_class.append(training_sheet.cell_value(i, j))
            self.train_data_kmean.append(sample_features)
            self.training_data.append(sample_features_with_class)
        ################load testing data
        self.testing_data = []
        rows = testing_sheet.nrows
        cols = testing_sheet.ncols
        for i in range(0, rows):
            sample_features = []
            for j in range(0, cols):
                sample_features.append(testing_sheet.cell_value(i, j))
            self.testing_data.append(sample_features)
        ################
        self.normalize()
        self.centers = self.kmeans(self.train_data_kmean, self.k)

    def initial_values(self, cmse_th, ceta, cepochs):
        self.mse_th = cmse_th
        self.eta = ceta
        self.epochs = cepochs
        self.num_classes = 5
        self.weights = self.initial_weights(self.k)

    def normalize(self):
        for j in range(1, len(self.training_data[0])):
            avg = 0
            mx = -1e18
            mn = 1e18
            num = 0
            for i in range(0, len(self.training_data)):
                avg += self.training_data[i][j]
                num += 1
                mx = max(mx, self.training_data[i][j])
                mn = min(mn, self.training_data[i][j])
            avg = avg / num
            self.avg_list.append(avg)
            self.mn_list.append(mn)
            self.mx_list.append(mx)
            for i in range(0, len(self.training_data)):
                self.train_data_kmean[i][j - 1] = (self.train_data_kmean[i][j - 1] - avg) / (mx - mn)
                self.training_data[i][j] = (self.training_data[i][j] - avg) / (mx - mn)
            for i in range(0, len(self.testing_data)):
                self.testing_data[i][j] = (self.testing_data[i][j] - avg) / (mx - mn)

    def normalize_sample(self, sample):
        for j in range(0, len(sample)):
            sample[j] = ((sample[j] - self.avg_list[j]) / (self.mx_list[j] - self.mn_list[j]))
        return sample

    def kmeans(self, training_data, k):
        current_centers = []
        previous_centers = []
        cluster_data = []
        for i in range(0, k):
            current_centers.append(training_data[i])
            cluster_data.append([])
            tmp_list = []
            for j in range(0, len(training_data[i])):
                tmp_list.append(1e18)
            previous_centers.append(tmp_list)
        while (check_stopping(current_centers, previous_centers) == 0):
            for i in range(0, len(training_data)):
                cluster = 0
                mn = 1e18
                for j in range(0, len(current_centers)):
                    euc_dist = euclidean_distance(training_data[i], current_centers[j])
                    if euc_dist < mn:
                        mn = euc_dist
                        cluster = j
                cluster_data[cluster].append(training_data[i])
            previous_centers = current_centers
            current_centers = calculate_new_centers(cluster_data)
        return current_centers

    def initial_weights(self, k):
        weights = []
        for i in range(0, self.num_classes):
            weight_vector = []
            for j in range(0, k):
                weight_vector.append(0)  # uniform(0, 1))
            weights.append(weight_vector)
        return weights

    def train(self):
        maxdistance = 0
        for i in self.centers:
            for j in self.centers:
                maxdistance = max(euclidean_distance(i, j), maxdistance)
        spread = maxdistance / math.sqrt(2 * self.k)
        for i in range(0, self.epochs):
            self.mse_classes = []
            for j in range(0, self.num_classes):
                self.mse_classes.append(0)
            for x in self.training_data:
                clas = x[0]
                features = x[1:len(x)]
                output_layer = []
                for j in range(0, self.num_classes):
                    output_layer.append(0)
                output_layer[np.int(clas - 1)] = 1
                hidden_layer = []
                for j in range(0, self.k):
                    dist_x_c = euclidean_distance(features, self.centers[j])
                    hidden_layer.append(math.exp((-1 * dist_x_c * dist_x_c) / (2 * spread * spread)))
                for j in range(0, self.num_classes):
                    d = output_layer[j]
                    y = multipky_two_vectors(self.weights[j], hidden_layer)
                    error = d - y
                    self.weights[j] = add_two_vectors(self.weights[j],
                                                      multiply_const_vector(error * self.eta, hidden_layer))
                    # if j==0:
                    # print(self.weights[0])
            for x in self.training_data:
                features = x[1:len(x)]
                out = self.mse_sample(features)
                clas = x[0]
                output_layer = []
                for j in range(0, self.num_classes):
                    output_layer.append(0)
                output_layer[np.int(clas - 1)] = 1
                for m in range(0, self.num_classes):
                    self.mse_classes[m] += (output_layer[m] - out[m])
            for m in range(0, self.num_classes):
                self.mse_classes[m] /= len(self.training_data)
            mse = .5 * get_average(self.mse_classes)
            if (mse < self.mse_th):
                break

    def mse_sample(self, sample):
        maxdistance = 0
        for i in self.centers:
            for j in self.centers:
                maxdistance = max(euclidean_distance(i, j), maxdistance)
        spread = maxdistance / math.sqrt(2 * self.k)
        hidden_layer = []
        for j in range(0, self.k):
            dist_x_c = euclidean_distance(sample, self.centers[j])
            hidden_layer.append(math.exp((-1 * dist_x_c * dist_x_c) / (2 * spread * spread)))
        output_layer = []
        for i in range(0, self.num_classes):
            output_layer.append(multipky_two_vectors(hidden_layer, self.weights[i]))
        return output_layer

    def classify(self, sample):
        maxdistance = 0
        for i in self.centers:
            for j in self.centers:
                maxdistance = max(euclidean_distance(i, j), maxdistance)
        spread = maxdistance / math.sqrt(2 * self.k)
        hidden_layer = []
        for j in range(0, self.k):
            dist_x_c = euclidean_distance(sample, self.centers[j])
            hidden_layer.append(math.exp((-1 * dist_x_c * dist_x_c) / (2 * spread * spread)))
        output_layer = []
        for i in range(0, self.num_classes):
            output_layer.append(multipky_two_vectors(hidden_layer, self.weights[i]))
        clas = -1
        mx = -1e25
        for i in range(0, self.num_classes):
            if (output_layer[i] > mx):
                mx = output_layer[i]
                clas = i
        return clas

    def test(self):
        confusion_matrix = np.zeros((5, 5))
        for x in self.testing_data:
            features = x[1:len(x)]
            desired_class = x[0] - 1
            predicted_class = self.classify(features)
            confusion_matrix[np.int(desired_class)][np.int(predicted_class)] += 1
        return confusion_matrix
        # return np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix) * 100

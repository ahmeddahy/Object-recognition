import numpy as np
from enum import Enum
from typing import List
import random
import pandas as pd
import xml.etree.cElementTree as ET
import xml.etree.ElementTree as ER


class Activation(Enum):
    SIGMOID = 1
    TANH = 2


class StoppingCriterion(Enum):
    EPOCHS = 1
    MSE = 2
    CROSS_VALIDATION = 3


class MLP:
    class __Layer:

        class _Node:
            def __init__(self, node_input: np.ndarray, af: Activation, bias: bool):
                self.__node_input = node_input
                self.__is_biased = bias
                self.__weight = np.empty(node_input.__len__())
                self.__weight.fill(0)
                self.__bias = 0
                self.af = af
                self.__output = 0
                self.local_gradient = 0

            @property
            def weight(self):
                return self.__weight.copy()

            @weight.setter
            def weight(self, value: np.ndarray):
                if value.__len__() == self.__weight.__len__():
                    self.__weight = value

            @property
            def bias(self):
                return self.__bias

            @bias.setter
            def bias(self, value: float):
                self.__bias = value

            @property
            def node_input(self):
                return self.__node_input.copy()

            @node_input.setter
            def node_input(self, value: np.ndarray):
                self.__node_input = value

            @property
            def output(self):
                return self.__output

            def __activate(self, net):
                if self.af == Activation.SIGMOID:
                    return 1 / (1 + np.exp(-net))
                elif self.af == Activation.TANH:
                    return np.tanh(net)

            def calculate_output(self):
                self.__output = self.__activate((self.__weight @ self.__node_input) + self.__bias)

            def update_weights(self, eta: float):
                self.__weight = self.__weight + eta * self.local_gradient * self.__node_input
                if self.__is_biased:
                    self.__bias = self.__bias + eta * self.local_gradient

        def __init__(self, node_count: int, layer_input: np.ndarray, layer_af: Activation, bias: bool):
            self._nodes: List[self._Node] = []
            self._bias = bias
            self._layer_output = np.empty(node_count)
            self._layer_output.fill(0)
            for i in range(0, node_count):
                node = self._Node(layer_input, layer_af, bias)
                self._nodes.append(node)

        @property
        def nodes(self):
            return self._nodes.copy()

        @property
        def modify_nodes(self):
            return self._nodes

        @property
        def layer_output(self):
            return self._layer_output.copy()

        def initialize_weights(self):
            n = self._nodes[0].weight.__len__()
            all_nodes = self._nodes.__len__()
            for i in range(0, all_nodes):
                some_weight = np.empty(n)
                for j in range(0, n):
                    some_weight[j] = (2 * random.random()) - 1
                self._nodes[i].weight = some_weight

        def supply_input(self, node_input: np.ndarray):
            n = self._nodes.__len__()
            for i in range(0, n):
                self._nodes[i].node_input = node_input

        def forward_pass(self):
            n = self._nodes.__len__()
            for i in range(0, n):
                self._nodes[i].calculate_output()
                self._layer_output[i] = self._nodes[i].output

        def update_weights(self, eta: float):
            n = self._nodes.__len__()
            for i in range(0, n):
                self._nodes[i].update_weights(eta)

    class __HiddenLayer(__Layer):

        def __init__(self, node_count: int, layer_input: np.ndarray, layer_af: Activation, bias: bool):
            super().__init__(node_count, layer_input, layer_af, bias)

        def backward_pass(self, layer):
            n = self._nodes.__len__()
            other_nodes = layer.nodes
            other_n = other_nodes.__len__()
            for i in range(0, n):
                summation = 0
                for j in range(0, other_n):
                    summation = summation + other_nodes[j].local_gradient * other_nodes[j].weight[i]
                a = self._nodes[i].output
                act = self._nodes[i].af
                if act == Activation.SIGMOID:
                    self._nodes[i].local_gradient = a * (1 - a) * summation
                elif act == Activation.TANH:
                    self._nodes[i].local_gradient = (1 + a) * (1 - a) * summation

    class __OutputLayer(__Layer):

        def __init__(self, node_count: int, layer_input: np.ndarray, layer_af: Activation, bias: bool):
            super().__init__(node_count, layer_input, layer_af, bias)

        def backward_pass(self, target: np.ndarray):
            n = self._nodes.__len__()
            if target.__len__() == n:
                for i in range(0, n):
                    a = self._nodes[i].output
                    act = self._nodes[i].af
                    if act == Activation.SIGMOID:
                        self._nodes[i].local_gradient = (target[i] - a) * a * (1 - a)
                    elif act == Activation.TANH:
                        self._nodes[i].local_gradient = (target[i] - a) * (1 + a) * (1 - a)

        def error_energy_per_pattern(self, target: np.ndarray):
            self.forward_pass()
            n = self._nodes.__len__()
            summation = 0
            if target.__len__() == n:
                for i in range(0, n):
                    e = target[i] - self._nodes[i].output
                    summation = summation + e * e
                return 0.5 * summation
            else:
                return - 1

    def __init__(self, af: Activation=None, is_biased: bool=True,

                 stopping_criterion: StoppingCriterion=None, eta: float=None, epochs: int=None, mse_threshold: float=None):
        self.__data = {}
        self.__training = {}
        self.__testing = {}
        self.__class_names = []
        self.__epochs = epochs
        self.__stopping_criterion = stopping_criterion
        self.__eta = eta
        self.__af = af
        self.__mse_threshold = mse_threshold
        self.__maxes = []
        self.__mins = []
        self.__is_biased = is_biased
        self.__means = []
        self.__hidden_layers: List[self.__HiddenLayer] = []
        self.__output_layer: self.__OutputLayer = None
        self.__mse_per_epoch = []

    @property
    def mse_per_epoch(self):
        return self.__mse_per_epoch.copy()

    def read_data(self, path: str):
        self.__data = {}
        self.__class_names = []
        f = open(path, 'r')
        f.readline()
        line = f.readline()
        line = line.split(',')
        key = line[line.__len__() - 1]
        key = key.rstrip()
        self.__class_names.append(key)
        line.pop()
        self.__data[key] = [line]
        while True:
            line = f.readline()
            if line == '':
                break
            line = line.split(',')
            if line[line.__len__() - 1].rstrip() != key:
                key = line[line.__len__() - 1]
                key = key.rstrip()
                self.__class_names.append(key)
                line.pop()
                self.__data[key] = [line]
            else:
                key = line[line.__len__() - 1]
                key = key.rstrip()
                line.pop()
                self.__data[key].append(line)
        for item in self.__data:
            self.__data[item] = np.array(self.__data[item]).astype(float)

        f.close()

    def read_excel(self, path: str):
        dftr: pd.DataFrame = pd.read_excel(path, 'Training', header=None)
        dfts: pd.DataFrame = pd.read_excel(path, 'Testing', header=None)
        valuestr: list = dftr.values.tolist()
        valuests: list = dfts.values.tolist()
        keytr = valuestr[0][0]
        keyts = valuests[0][0]
        self.__class_names.append(keytr)
        self.__training[keytr] = [valuestr[0][1:valuestr[0].__len__()]]
        self.__testing[keyts] = [valuests[0][1:valuests[0].__len__()]]
        for i in range(1, valuestr.__len__()):
            if keytr == valuestr[i][0]:
                self.__training[keytr].append(valuestr[i][1:valuestr[0].__len__()])
            else:
                keytr = valuestr[i][0]
                self.__training[keytr] = [valuestr[i][1:valuestr[0].__len__()]]
                self.__class_names.append(keytr)
        for key in self.__training:
            self.__training[key] = np.array(self.__training[key]).astype(float)

        for i in range(1, valuests.__len__()):
            if keyts == valuests[i][0]:
                self.__testing[keyts].append(valuests[i][1:valuests[0].__len__()])
            else:
                keyts = valuests[i][0]
                self.__testing[keyts] = [valuests[i][1:valuests[0].__len__()]]
        for key in self.__training:
            self.__testing[key] = np.array(self.__testing[key]).astype(float)

    def write_xml(self):
        # root = ET.Element("root")
        # doc = ET.SubElement(root, "doc")
        #
        # ET.SubElement(doc, "field1", name="blah").text = "some value1"
        # ET.SubElement(doc, "field2", name="asdfasd").text = "some vlaue2"
        #
        # tree = ET.ElementTree(root)
        # tree.write("filename.xml")
        root = ET.Element("MLP")
        cnl = self.__class_names.__len__()
        class_names_string = ""
        means_string = ""
        maxes_string = ""
        mins_string = ""
        for i in range(0, cnl):
            if i == 0:
                class_names_string = str(self.__class_names[i])
            else:
                class_names_string = class_names_string + "," + str(self.__class_names[i])

        for i in range(0, self.__means.__len__()):
            if i == 0:
                means_string = str(self.__means[i])
            else:
                means_string = means_string + "," + str(self.__means[i])

        for i in range(0, self.__maxes.__len__()):
            if i == 0:
                maxes_string = str(self.__maxes[i])
            else:
                maxes_string = maxes_string + "," + str(self.__maxes[i])

        for i in range(0, self.__mins.__len__()):
            if i == 0:
                mins_string = str(self.__mins[i])
            else:
                mins_string = mins_string + "," + str(self.__mins[i])

        ET.SubElement(root, "means").text = means_string
        ET.SubElement(root, "maxes").text = maxes_string
        ET.SubElement(root, "mins").text = mins_string
        ET.SubElement(root, "classes").text = class_names_string
        ET.SubElement(root, "nfeatures").text = str(self.__training[self.__class_names[0]].shape[1])
        ET.SubElement(root, "eta").text = str(self.__eta)
        ET.SubElement(root, "stopping").text = str(self.__stopping_criterion)
        if self.__stopping_criterion == StoppingCriterion.MSE:
            ET.SubElement(root, "mse").text = str(self.__mse_threshold)
        elif self.__stopping_criterion == StoppingCriterion.EPOCHS:
            ET.SubElement(root, "epochs").text = str(self.__epochs)
        ET.SubElement(root, "af").text = str(self.__af)
        ET.SubElement(root, "bias").text = str(self.__is_biased)
        node_list = ""
        nhl = self.__hidden_layers.__len__()
        for i in range(0, nhl):
            if i == 0:
                node_list = str(self.__hidden_layers[i].nodes.__len__())
            else:
                node_list = node_list + "," + str(self.__hidden_layers[i].nodes.__len__())

        ET.SubElement(root, "node_list").text = node_list
        for i in range(0, nhl):
            hl = ET.SubElement(root, "HiddenLayer" + str(i + 1))
            node_count = self.__hidden_layers[i].nodes.__len__()
            nodes = self.__hidden_layers[i].nodes
            for j in range(0, node_count):
                weight_count = nodes[j].weight.__len__()
                weight = nodes[j].weight
                weight_string = ""
                for k in range(0, weight_count):
                    if k == 0:
                        weight_string = str(weight[k])
                    else:
                        weight_string = weight_string + "," + str(weight[k])
                node_elem = ET.SubElement(hl, "Node" + str(j + 1))
                ET.SubElement(node_elem, "weight").text = weight_string
                if self.__is_biased:
                    ET.SubElement(node_elem, "bias").text = str(nodes[j].bias)

        ol = ET.SubElement(root, "OutputLayer")
        node_count = self.__output_layer.nodes.__len__()
        nodes = self.__output_layer.nodes
        for j in range(0, node_count):
            weight_count = nodes[j].weight.__len__()
            weight = nodes[j].weight
            weight_string = ""
            for k in range(0, weight_count):
                if k == 0:
                    weight_string = str(weight[k])
                else:
                    weight_string = weight_string + "," + str(weight[k])
            node_elem = ET.SubElement(ol, "Node" + str(j + 1))
            ET.SubElement(node_elem, "weight").text = weight_string
            if self.__is_biased:
                ET.SubElement(node_elem, "bias").text = str(nodes[j].bias)

        tree = ET.ElementTree(root)
        tree.write("MLP.xml")

    def read_xml(self, path):
        x_root: ER.Element = ER.parse(path).getroot()
        x_children = list(x_root)
        w = 0
        node_list = []
        index = 0
        for child in x_children:
            if child.tag == 'means':
                self.__means = np.array(child.text.split(',')).astype(float).tolist()
                index = index + 1
            elif child.tag == 'maxes':
                self.__maxes = np.array(child.text.split(',')).astype(float).tolist()
                index = index + 1
            elif child.tag == 'mins':
                self.__mins = np.array(child.text.split(',')).astype(float).tolist()
                index = index + 1
            elif child.tag == 'classes':
                self.__class_names = np.array(child.text.split(',')).astype(float).tolist()
                index = index + 1
            elif child.tag == 'nfeatures':
                w = int(child.text)
                index = index + 1
            elif child.tag == 'eta':
                self.__eta = float(child.text)
                index = index + 1
            elif child.tag == 'stopping':
                self.__stopping_criterion = StoppingCriterion[child.text.split('.')[1]]
                index = index + 1
            elif child.tag == 'mse':
                self.__mse_threshold = float(child.text)
                index = index + 1
            elif child.tag == 'epochs':
                self.__epochs = int(child.text)
                index = index + 1
            elif child.tag == 'af':
                self.__af = Activation[child.text.split('.')[1]]
                index = index + 1
            elif child.tag == 'bias':
                self.__is_biased = bool(child.text)
                index = index + 1
            elif child.tag == 'node_list':
                node_list = child.text.split(',')
                node_list = np.array(node_list).astype(int).tolist()
                index = index + 1
            else:
                break
        x_children = x_children[index:x_children.__len__()]
        for child in x_children:
            for i in range(0, node_list.__len__()):
                if child.tag == 'HiddenLayer' + str(i + 1):
                    x_grandchildren = list(child)
                    if i == 0:
                        some_input = np.empty(w)
                    else:
                        some_input = np.empty(node_list[i - 1])
                    some_input.fill(0)
                    some_layer = self.__HiddenLayer(node_list[i], some_input, self.__af, self.__is_biased)
                    weight: np.ndarray = None
                    node_index = 0
                    bias: float = 0
                    for grandchild in x_grandchildren:
                        node_contents = list(grandchild)
                        for content in node_contents:
                            if content.tag == 'weight':
                                weight = np.array(content.text.split(',')).astype(float)
                            else:
                                bias = float(content.text)
                        some_layer.modify_nodes[node_index].weight = weight
                        some_layer.modify_nodes[node_index].bias = bias
                        node_index = node_index + 1
                    self.__hidden_layers.append(some_layer)
            if child.tag == 'OutputLayer':
                x_grandchildren = list(child)
                some_input = np.empty(node_list[-1])
                some_input.fill(0)
                self.__output_layer = \
                    self.__OutputLayer(self.__class_names.__len__(), some_input, self.__af, self.__is_biased)
                weight: np.ndarray = None
                node_index = 0
                bias: float = 0
                for grandchild in x_grandchildren:
                    node_contents = list(grandchild)
                    for content in node_contents:
                        if content.tag == 'weight':
                            weight = np.array(content.text.split(',')).astype(float)
                        else:
                            bias = float(content.text)
                    self.__output_layer.modify_nodes[node_index].weight = weight
                    self.__output_layer.modify_nodes[node_index].bias = bias
                    node_index = node_index + 1

    def construct_network(self, node_list: list):
        w = self.__training[self.__class_names[0]].shape[1]
        random.seed(node_list.__len__())
        some_input = np.empty(w)
        some_weight = np.empty(w)
        for i in range(0, some_weight.__len__()):
            some_weight[i] = random.random()
        some_input.fill(0)
        n = node_list.__len__()
        some_layer = self.__HiddenLayer(node_list[0], some_input, self.__af, self.__is_biased)
        some_layer.initialize_weights()
        self.__hidden_layers.append(some_layer)
        for i in range(1, n):
            some_input = np.empty(node_list[i - 1])
            some_input.fill(0)
            some_layer = self.__HiddenLayer(node_list[i], some_input, self.__af, self.__is_biased)
            some_layer.initialize_weights()
            self.__hidden_layers.append(some_layer)
        some_input = np.empty(node_list[-1])
        some_input.fill(0)
        self.__output_layer = self.__OutputLayer(self.__class_names.__len__(), some_input, self.__af, self.__is_biased)
        self.__output_layer.initialize_weights()

    def train(self):
        cross_mse = float("inf")
        train_data = self.__training.copy()
        for key in train_data:
            train_data[key] = train_data[key][0:int(np.ceil(0.8 * train_data[key].__len__())), :]
        cross_data = self.__training.copy()
        for key in cross_data:
            cross_data[key] = \
                cross_data[key][int(np.ceil(0.8 * cross_data[key].__len__())):cross_data[key].__len__(), :]
        w = train_data[self.__class_names[0]].shape[1]

        for i in range(0, w):
            col = []
            for key in train_data:
                temp = train_data[key][:, i].tolist()
                col.extend([temp])
            col = np.array(col)
            self.__means.append(col.mean())
            self.__maxes.append(col.max())
            self.__mins.append(col.min())
        if self.__af == Activation.SIGMOID:
            for i in range(0, w):
                for key in train_data:
                    train_data[key][:, i] = train_data[key][:, i] - self.__means[i]
                    train_data[key][:, i] = (train_data[key][:, i] - self.__mins[i]) / (
                            self.__maxes[i] - self.__mins[i])
        elif self.__af == Activation.TANH:
            for i in range(0, w):
                for key in train_data:
                    train_data[key][:, i] = train_data[key][:, i] - self.__means[i]
                    train_data[key][:, i] = 2 * (train_data[key][:, i] - self.__mins[i]) / (
                            self.__maxes[i] - self.__mins[i]) - 1

        t_vec = np.empty(self.__class_names.__len__(), dtype=int)
        t_vec.fill(0)
        i = 0
        while True:
            t_index = 0
            t_vec.fill(0)
            for key in train_data:
                t_vec.fill(0)
                t_vec[t_index] = 1
                for a_row in train_data[key]:
                    n = self.__hidden_layers.__len__()
                    # forward pass
                    self.__hidden_layers[0].supply_input(a_row)
                    self.__hidden_layers[0].forward_pass()
                    for h in range(1, n):
                        self.__hidden_layers[h].supply_input(self.__hidden_layers[h - 1].layer_output)
                        self.__hidden_layers[h].forward_pass()
                    self.__output_layer.supply_input(self.__hidden_layers[-1].layer_output)
                    self.__output_layer.forward_pass()
                    # backward pass
                    self.__output_layer.backward_pass(t_vec)
                    self.__hidden_layers[-1].backward_pass(self.__output_layer)
                    for h in range(n - 2, -1, -1):
                        self.__hidden_layers[h].backward_pass(self.__hidden_layers[h + 1])
                    # update weights
                    for h in range(0, n):
                        self.__hidden_layers[h].update_weights(self.__eta)
                    self.__output_layer.update_weights(self.__eta)
                t_index = t_index + 1
            if self.__stopping_criterion == StoppingCriterion.MSE:
                t_index = 0
                t_vec.fill(0)
                summation = 0
                pattern_count = 0
                for key in train_data:
                    t_vec.fill(0)
                    t_vec[t_index] = 1
                    for a_row in train_data[key]:
                        pattern_count = pattern_count + 1
                        n = self.__hidden_layers.__len__()
                        # forward pass
                        self.__hidden_layers[0].supply_input(a_row)
                        self.__hidden_layers[0].forward_pass()
                        for h in range(1, n):
                            self.__hidden_layers[h].supply_input(self.__hidden_layers[h - 1].layer_output)
                            self.__hidden_layers[h].forward_pass()
                        self.__output_layer.supply_input(self.__hidden_layers[-1].layer_output)
                        self.__output_layer.forward_pass()
                        summation = summation + self.__output_layer.error_energy_per_pattern(t_vec)
                    t_index = t_index + 1
                summation = summation / float(pattern_count)
                self.__mse_per_epoch.append(summation)
                if summation <= self.__mse_threshold:
                    break
            i = i + 1
            if self.__stopping_criterion == StoppingCriterion.EPOCHS:
                if i == self.__epochs:
                    break
            if self.__stopping_criterion == StoppingCriterion.CROSS_VALIDATION:
                if i % 50 == 0:
                    t_index = 0
                    t_vec.fill(0)
                    summation = 0
                    pattern_count = 0
                    for key in cross_data:
                        t_vec.fill(0)
                        t_vec[t_index] = 1
                        for a_row in cross_data[key]:
                            pattern_count = pattern_count + 1
                            n = self.__hidden_layers.__len__()
                            # forward pass
                            self.__hidden_layers[0].supply_input(a_row)
                            self.__hidden_layers[0].forward_pass()
                            for h in range(1, n):
                                self.__hidden_layers[h].supply_input(self.__hidden_layers[h - 1].layer_output)
                                self.__hidden_layers[h].forward_pass()
                            self.__output_layer.supply_input(self.__hidden_layers[-1].layer_output)
                            self.__output_layer.forward_pass()
                            summation = summation + self.__output_layer.error_energy_per_pattern(t_vec)
                        t_index = t_index + 1
                    summation = summation / float(pattern_count)
                    if summation <= cross_mse:
                        cross_mse = summation
                    else:
                        break

    def determine_class(self, sample: np.ndarray):
        w = self.__hidden_layers[0].nodes[0].node_input.__len__()
        if self.__af == Activation.SIGMOID:
            for i in range(0, w):
                sample[i] = sample[i] - self.__means[i]
                sample[i] = (sample[i] - self.__mins[i])/(self.__maxes[i] - self.__mins[i])
        elif self.__af == Activation.TANH:
            for i in range(0, w):
                sample[i] = sample[i] - self.__means[i]
                sample[i] = 2 * (sample[i] - self.__mins[i])/(self.__maxes[i] - self.__mins[i]) - 1
        n = self.__hidden_layers.__len__()
        self.__hidden_layers[0].supply_input(sample)
        self.__hidden_layers[0].forward_pass()
        for h in range(1, n):
            self.__hidden_layers[h].supply_input(self.__hidden_layers[h - 1].layer_output)
            self.__hidden_layers[h].forward_pass()
        self.__output_layer.supply_input(self.__hidden_layers[-1].layer_output)
        self.__output_layer.forward_pass()
        return self.__output_layer.layer_output.tolist().index(self.__output_layer.layer_output.max()) + 1

    def test(self):
        test_data = self.__testing.copy()
        w = test_data[self.__class_names[0]].shape[1]
        if self.__af == Activation.SIGMOID:
            for i in range(0, w):
                for key in test_data:
                    test_data[key][:, i] = test_data[key][:, i] - self.__means[i]
                    test_data[key][:, i] = (test_data[key][:, i] - self.__mins[i]) / (
                            self.__maxes[i] - self.__mins[i])
        elif self.__af == Activation.TANH:
            for i in range(0, w):
                for key in test_data:
                    test_data[key][:, i] = test_data[key][:, i] - self.__means[i]
                    test_data[key][:, i] = 2 * (test_data[key][:, i] - self.__mins[i]) / (
                            self.__maxes[i] - self.__mins[i]) - 1
        nc = self.__class_names.__len__()
        confusion_matrix = np.empty([nc, nc], dtype=int)
        confusion_matrix.fill(0)
        t_vec = np.empty(nc)
        t_vec.fill(0)
        t_index = 0
        for key in test_data:
            t_vec.fill(0)
            t_vec[t_index] = 1
            for a_row in test_data[key]:
                n = self.__hidden_layers.__len__()
                # forward pass
                self.__hidden_layers[0].supply_input(a_row)
                self.__hidden_layers[0].forward_pass()
                for h in range(1, n):
                    self.__hidden_layers[h].supply_input(self.__hidden_layers[h - 1].layer_output)
                    self.__hidden_layers[h].forward_pass()
                self.__output_layer.supply_input(self.__hidden_layers[-1].layer_output)
                self.__output_layer.forward_pass()
                actual = self.__output_layer.layer_output.tolist().index(self.__output_layer.layer_output.max())
                confusion_matrix[t_index][actual] = confusion_matrix[t_index][actual] + 1
            t_index = t_index + 1
        accuracy = ((confusion_matrix.diagonal().sum()) / float(confusion_matrix.sum())) * 100
        return accuracy, confusion_matrix


# multi = MLP(Activation.SIGMOID, True, StoppingCriterion.MSE, eta=0.05, mse_threshold=0.05)
'''multi = MLP()
multi.read_excel('ObjectRecognition2.xls')
multi.read_xml("MLP.xml")
s = multi.determine_class(np.array([-1042.706097, -1695.761646, 242.0806125, 396.9620878, 1088.126413
                                    , -46.8562763, 597.2557591, -570.8149451, -105.2313088, 93.56346076
                                    , -60.11626973, 316.1830735, 497.9019359, -26.74334951, -214.5986072
                                    , 280.4265774, 153.6478501, -174.2923481, -44.88774026, 50.76160399]))
# multi.construct_network([6])
# multi.train()
acc, conf = multi.test()
# multi.write_xml()
print(acc)
print(conf)
print(s)'''
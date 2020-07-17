
# Comp 131 Assignment 4
# ANN with 1 hidden layer to classify Irises
#
#
# NOTE: MUST BE RAN IN A DIRECTORY THAT CONTAINS IRIS DATA
# TEXT FILE

import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


# Class for an ANN with one hidden layer
class NeuralNet:

    # initializes all relevant member variables
    def __init__(self, num_features=1, given_bias=1, act_func=lambda x: x,
                 learning_rate=0.1, output_bits=1):

        self.num_features = num_features
        self.hidden_nodes = num_features
        self.output_bits = output_bits

        self.input_layer = np.zeros(num_features)
        self.hidden_layer = np.zeros(self.hidden_nodes)
        self.output = np.zeros(output_bits)
        self.expected_output = np.zeros(output_bits)
        self.bias = given_bias

        self.weights1 = np.random.randn(self.hidden_nodes, self.num_features)
        self.weights2 = np.random.randn(output_bits, self.hidden_nodes)
        self.bias_weights = np.random.randn(self.hidden_nodes + output_bits)

        self.total_error = 0
        self.learning_rate = learning_rate
        self.activation_func = act_func
        self.scaler = None
        print("W1: ", self.weights1)
        print("W2: ", self.weights2)
        print("WB: ", self.bias_weights)

    # Trains the ANN based on a given training set and validation set (along with a list of
    # associated labels).
    def train(self, trn_smpls, trn_lbls, val_smpls, val_lbls, threshold=1, max_iters=280):
        iters = 0
        # normalizes inputs
        trn_smpls = self.normalize(trn_smpls)
        val_smpls = self.normalize(val_smpls)
        # trains until threshold iterations, or min error on the validation set
        while True:
            # uses training set to train
            for i in range(len(trn_smpls)):
                self.input_layer = trn_smpls[i]
                self.expected_output = trn_lbls[i]
                self.forward_pass()
                self.backward_pass()
            self.total_error = 0
            # uses validation set to calculate error
            for i in range(len(val_smpls)):
                self.input_layer = val_smpls[i]
                self.expected_output = val_lbls[i]
                self.forward_pass()
                for i in range(self.output_bits):
                    self.total_error += (self.expected_output[i] - self.output[i]) ** 2
            self.total_error /= 2
            if self.total_error < threshold or iters > max_iters:
                break
            iters += 1

    # Performs forward pass of the back propagation algorithm (as described in the lecture slides)
    def forward_pass(self):
        # calculate activation of hidden nodes due to input layer
        for i in range(self.hidden_nodes):
            self.hidden_layer[i] = 0
            for j, inpt in enumerate(self.input_layer):
                self.hidden_layer[i] += inpt * self.weights1[i][j]
            self.hidden_layer[i] += self.bias * self.bias_weights[i]
            self.hidden_layer[i] = self.activation_func(self.hidden_layer[i])

        # calculate activation of output nodes due to hidden layer
        for i in range(self.output_bits):
            self.output[i] = 0
            for j, activation in enumerate(self.hidden_layer):
                self.output[i] += activation * self.weights2[i][j]
            self.output[i] += self.bias * self.bias_weights[i+self.hidden_nodes]
            self.output[i] = self.activation_func(self.output[i])

    # Performs backward pass of the back propagation algorithm (as described in the lecture slides)
    def backward_pass(self):
        # find error of output node
        output_error = [0 for _ in range(self.output_bits)]
        for i in range(self.output_bits):
            output_error[i] = self.output[i] * (1 - self.output[i]) * (self.expected_output[i] - self.output[i])

        hidden_errors = [0 for _ in range(self.hidden_nodes)]
        # find error contribution from hidden nodes
        for i in range(self.hidden_nodes):
            prev_errors = 0
            for j in range(self.output_bits):
                prev_errors += self.weights2[j][i] * output_error[j]
            hidden_errors[i] = self.hidden_layer[i]*(1-self.hidden_layer[i])*self.weights2[j][i]*prev_errors

        # update weights connecting hidden nodes to output node
        for i in range(self.output_bits):
            for j in range(self.hidden_nodes):
                self.weights2[i][j] = self.weights2[i][j] + \
                                      self.learning_rate*self.hidden_layer[j]*output_error[i]

        # update weights connecting input nodes to hidden nodes
        for i in range(self.hidden_nodes):
            for j in range(self.num_features):
                self.weights1[i][j] = self.weights1[i][j] + \
                             self.learning_rate*hidden_errors[i]*self.input_layer[j]

        # update weight connecting bias to output nodes
        for x in range(self.hidden_nodes, len(self.bias_weights)):
            self.bias_weights[x] += self.learning_rate * self.bias * output_error[x - self.hidden_nodes]

        # update weights connecting bias to hidden nodes
        for x in range(self.hidden_nodes):
            self.bias_weights[x] += self.learning_rate * self.bias * hidden_errors[x]

    # Predicts label based on an input vector
    def query(self, vector):
        vector = self.scaler.transform(vector.reshape(1, -1))[0]
        self.input_layer = vector
        self.forward_pass()
        return self.output

    # Normalizes inputs by linearly mapping them into the ranges (-6, 6),
    # since beyond this range the sigmoid function always returns 0 or 1.
    def normalize(self, data):
        scaler = MinMaxScaler((-6, 6))
        scaler.fit(data)
        self.scaler = scaler
        return scaler.transform(data)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# takes a label (string, as in the data set) and converts it to a one-hot encoding
def encode_label(label):
    if label == "Iris-versicolor":
        return [1, 0, 0]
    elif label == "Iris-virginica":
        return [0, 1, 0]
    elif label == "Iris-setosa":
        return [0, 0, 1]
    else:
        print("Error: invalid label")
        return []


# takes an output from the neural network and returns
def decode_label(encoding):
    least_diff, index = float("inf"), 0
    for i, num in enumerate(encoding):
        diff = abs(1-num)
        if diff < least_diff:
            least_diff, index = diff, i

    if index == 0:
        return "Iris-versicolor"
    if index == 1:
        return "Iris-virginica"
    else:
        return "Iris-setosa"


# reads iris_data file and constructs data matrix.
def read_file():
    raw_data = []
    with open("test.txt") as file:
        for line in file.readlines():
            line = line.split(",")
            if len(line) != 5:
                continue
            encoding = encode_label(line[4][:-1])
            line = [float(value) for value in line[:4]]
            line.append(encoding)
            raw_data.append(line)

    # corrections described in "Iris description" file
    raw_data[34][3] = 0.2
    raw_data[37][1] = 3.6
    raw_data[37][2] = 1.4

    return raw_data


# randomizes ordering of samples and separates data matrix into two
# matrices: samples and labels
def parse_data(raw_data):

    # np.random.shuffle(raw_data)
    samples, labels = [], []
    for entry in raw_data:
        samples.append(entry[:-1])
        labels.append(entry[-1])

    samples, labels = np.array(samples), np.array(labels)
    return samples, labels
    exit()


# Trains and then tests ANN on Iris data set. Reports accuracy
if __name__ == "__main__":
    # np.random.seed(5)
    raw_data = read_file()
    print(raw_data)
    samples, labels = parse_data(raw_data)
    training_samples, training_labels = samples[:90], labels[:90]
    validation_samples, validation_labels = samples[90:120], labels[90:120]
    testing_samples, testing_labels = samples[120:150], labels[120:150]

    net = NeuralNet(num_features=4, act_func=sigmoid, output_bits=3, learning_rate=0.1)
    net.train(training_samples, training_labels, validation_samples, validation_labels,
                  threshold=1, max_iters=280)

    accuracy = 0
    for i in range(len(testing_samples)):
        if decode_label(testing_labels[i]) == decode_label(net.query(testing_samples[i])):
            accuracy += 1
    accuracy *= 100 / len(testing_samples)
    print("Accuracy:", accuracy, "%")

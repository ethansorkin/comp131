#
#   hw4.py
#
#    Ethan Sorkin
#    4/28/2020
#    COMP 131: Artificial Intelligence
#


import math
import csv
import random
from sklearn.preprocessing import MinMaxScaler

# Constants:
MAX_ITERS = 300
THRESHOLD = 3


# ANN Class with 1 hidden layer
class ANN:
    def __init__(self, num_inputs, num_outputs, output_neurons, bias, af, l_rate):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = self.num_inputs
        self.output_neurons = output_neurons
        self.bias = bias

        self.b_weights = [random.uniform(-1.0, 1.0) for _ in range(self.num_hidden +
                                                                   self.num_outputs)]
        self.h_weights = [[random.uniform(-1.0, 1.0) for _ in range(self.num_inputs)]
                                                     for _ in range(self.num_hidden)]
        self.o_weights = [[random.uniform(-1.0, 1.0) for _ in range(self.num_hidden)]
                                                    for _ in range(self.num_outputs)]

        self.inputs = [0] * self.num_inputs
        self.hidden_layer = [0] * self.num_hidden
        self.outputs = [0] * self.num_outputs
        self.exp_output = [0] * self.num_outputs

        self.activation_func = af
        self.learning_rate = l_rate
        self.total_error = 0
        self.scaler = None


    # Trains the ANN given training and validation sets
    def train(self, training_set, validation_set):
        # Scale inputs to fit range of (-6, 6)
        temp = [p[:-1] for p in training_set]
        temp = self.scale(temp)
        for i in range(len(training_set)):
            training_set[i][:-1] = temp[i]

        temp = [p[:-1] for p in validation_set]
        temp = self.scale(temp)
        for i in range(len(validation_set)):
            validation_set[i][:-1] = temp[i]

        # Trains ANN until total error hits threshold, or when the max # of iterations is reached
        count = 0
        while True:
            # Training set
            for i in range(len(training_set)):
                self.inputs = training_set[i][:-1]
                self.exp_output = training_set[i][-1]
                self.forward_prop()
                self.backward_prop()
            self.total_error = 0

            # Validation set
            for i in range(len(validation_set)):
                self.inputs = validation_set[i][:-1]
                self.exp_output = validation_set[i][-1]
                self.forward_prop()
                for j in range(self.num_outputs):
                    self.total_error += (self.exp_output[j] - self.outputs[j]) ** 2
            self.total_error *= (1 / self.output_neurons)
            if self.total_error < THRESHOLD or count > MAX_ITERS:
                break
            count += 1

    # Forward propagation
    def forward_prop(self):
        # Hidden neurons
        for h in range(self.num_hidden):
            self.hidden_layer[h] = 0
            # Calculate membrane potential
            for i in range(self.num_inputs):
                self.hidden_layer[h] += self.h_weights[h][i] * self.inputs[i]
            self.hidden_layer[h] += self.b_weights[h] * self.bias
            # Calculate output activation
            self.hidden_layer[h] = self.activation_func(self.hidden_layer[h])

        # Output neurons
        for o in range(self.num_outputs):
            self.outputs[o] = 0
            # Calculate membrane potential
            for h in range(self.num_hidden):
                self.outputs[o] += self.o_weights[o][h] * self.hidden_layer[h]
            self.outputs[o] += self.b_weights[o + self.num_hidden] + self.bias
            # Calculate output activation
            self.outputs[o] = self.activation_func(self.outputs[o])


    # Backward propagation
    def backward_prop(self):
        # Calculate error from output
        err = [0] * self.num_outputs
        for i in range(self.num_outputs):
            err[i] = self.outputs[i] * (1 - self.outputs[i]) * \
                     (self.exp_output[i] - self.outputs[i])

        # Calculate error from hidden layer
        h_err = [0] * self.num_hidden
        for i in range(self.num_hidden):
            prev = 0
            for j in range(self.num_outputs):
                prev += self.o_weights[j][i] * err[j]
            h_err[i] = self.hidden_layer[i] * (1 - self.hidden_layer[i]) * \
                                                    self.o_weights[j][i] * prev

        # Update output weights
        for i in range(self.num_outputs):
            for j in range(self.num_hidden):
                # Adding rate of change of err
                self.o_weights[i][j] += self.learning_rate * self.hidden_layer[j] * err[i]

        # Update hidden layer weights
        for i in range(self.num_hidden):
            for j in range(self.num_inputs):
                # Adding rate of change of err
                self.h_weights[i][j] += self.learning_rate * self.inputs[j] * h_err[i]

        # Update bias weights (for output and hidden neurons)
        for i in range(len(self.b_weights)):
            # Adding rate of change of err
            if i < self.num_hidden:
                self.b_weights[i] += self.learning_rate * self.hidden_layer[i] * h_err[i]
            else:
                self.b_weights[i] += self.learning_rate * self.outputs[i - self.num_hidden] * err[i - self.num_hidden]

    # Predicts classification of plants in testing set
    def classify(self, v):
        v = self.scaler.transform([v])[0]
        self.inputs = v
        self.forward_prop()

        # Compares outputs to one-hot encoding
        min_d, count, i = float("inf"), 0, 0
        for x in self.outputs:
            diff = abs(1 - x)
            if diff < min_d:
                min_d, i = diff, count
            count += 1

        # Classifies plant based on encoding with smallest differennce
        if i == 0:
            return [1, 0, 0]
        elif i == 1:
            return [0, 1, 0]
        elif i == 2:
            return [0, 0, 1]


    # Scales input values by linearly mapping them into the range (-6, 6)
    def scale(self, data):
        scaler = MinMaxScaler((-6, 6))
        scaler.fit(data)
        self.scaler = scaler
        return scaler.transform(data)


# Converts plant class to one-hot encoding
def one_hot(p_class):
    if p_class == "Iris-versicolor":
        return [1, 0, 0]
    elif p_class == "Iris-virginica":
        return [0, 1, 0]
    elif p_class == "Iris-setosa":
        return [0, 0, 1]
    else:
        print("Error: invalid plant class")
        return []

# Reads file in CSV format, encodes plants classes, and randomizes data
def read_file(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = list(csv_reader)
        data.pop()  # removing empty set at the EOF
        for row in data:
            for col in range(len(row) - 1):
                row[col] = float(row[col])  # converting strings to floats
            row[-1] = one_hot(row[-1])
    random.shuffle(data)
    return data



def main():
    data = read_file("ANN - Iris data.txt")

    # Randomly assigns training, validation, and test sets with 60:20:20 ratio
    training_set = data[:90]
    validation_set = data[90:120]
    test_set = data[120:150]

    iris_net = ANN(4, 3, 2, 1, lambda x: 1 / (1 + math.exp(-x)), 0.1)
    iris_net.train(training_set, validation_set)

    # Calculate accuracy
    correct = 0
    for x in test_set:
        if x[-1] == iris_net.classify(x[:-1]):
            correct += 1
    accuracy = (correct / len(test_set)) * 100
    print("ANN Accuracy: " + str(accuracy) + "%")


if __name__ == "__main__":
    main()
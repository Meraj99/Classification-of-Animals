import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


def process(f, dim=(64, 64)):
    imgnames = os.listdir("raw-img/" + f)[:100]
    data = []
    for i in imgnames:
        data.append(cv2.imread("raw-img/" + f + "/" + i))
        dataresized = []
        final = []
    for i in range(len(data)):
        dataresized.append(cv2.cvtColor(cv2.resize(data[i], dim, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    for i in range(len(dataresized)):
        bluredimg = cv2.GaussianBlur(dataresized[i], (5, 5), 0)
        hequalized = cv2.equalizeHist(bluredimg)
        tresh = cv2.adaptiveThreshold(hequalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
        final.append(tresh)

    os.mkdir("raw-img/" + str(dim[0]) + "/" + f + "-processed")
    for i in range(len(final)):
        cv2.imwrite("raw-img/" + str(dim[0]) + "/" + f + "-processed/" + str(i) + ".jpg", final[i])


nn_architecture = [
    {"input_dim": 200, "output_dim": 16, "activation": "sigmoid"}, # TODO: Modify input_dim as required
    {"input_dim": 16, "output_dim": 16, "activation": "sigmoid"},
    {"input_dim": 16, "output_dim": 10, "activation": "sigmoid"},
    {"input_dim": 10, "output_dim": 10, "activation": "linear"} # Ten output classes
]


def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}


    # Initialize weights and biases to random value
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1

    return params_values


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

def sigmoid_derivative(dA, x):
    sig = sigmoid(x)
    return dA * sig * (1 - sig)

def relu_derivative(dA, x):
    dZ = np.array(dA, copy = True)
    dZ[x <= 0] = 0;
    return dZ


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="sigmoid"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    elif activation is "linear":
        activation_func = linear
    else:
        raise Exception('Invalid activation function')

    return activation_func(Z_curr), Z_curr # Z_curr is needed for backward pass


def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    return A_curr, memory
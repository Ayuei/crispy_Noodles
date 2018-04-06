import os
# import sys
import scipy
# import sklearn
import numpy as np

# import sklearn.datasets
# import sklearn.linear_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize

# Todo: normalization / regularization (L2 / dropout)
# Batch /mini batch gradient descent - compare performance

# Hyperparameters
NUM_PIX = 64
IMG_SIZE = NUM_PIX * NUM_PIX * 3  # Num features (flattened)
NN_DIMENSION = [imgSize, 12, 12, 8, 8, 4, 4, 1]  # 6 hidden layers + 1 output layer (with 1 neuron)
LAMBDA = 0.03  # Update weights
ITERATIONS = 3000  # Number of iterations


def loadImage(imagePath):
    # Loads image (of any size) into flattened numpy array
    fname = imagePath
    image = np.array(scipy.ndimage.imread(fname, flatten=False))
    img = resize(image, (NUM_PIX, NUM_PIX, 3)).reshape((1, IMG_SIZE))
    img = img / 255  # Norm
    return img


def addImagesToDataSet(directory_in_str):
    # Adds all images within a directory to the datasets 'X' and 'Y' (with Y just being true/false labels)
    # If the image starts with 1 it's labelled as a 'True' image, otherwise 'False'
    directory = os.fspath(directory_in_str)
    X = np.zeros((0, IMG_SIZE))
    Y = np.zeros((0, 1))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            if filename.startswith("1"):
                Y = np.r_[Y, [[1]]]
            else:
                Y = np.r_[Y, [[0]]]
            # print(filename)
            fp = os.path.join(directory, filename)
            img = loadImage(fp)
            X = np.r_[X, img]
        else:
            continue

    return X.T, Y.T


def appendImagesToDataSet(directory_in_str, typeVal, X, Y):
    # Appends images in the specified directory to the dataset X and labels Y.
    # TypeVal specifies how to label the images (e.g. True or False)

    directory = os.fspath(directory_in_str)


    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith("pg"):
            Y = np.r_[Y, [[typeVal]]]
            fp = os.path.join(directory, filename)
            img = loadImage(fp)
            X = np.r_[X, img]
            # print('Appending data to X, new shape: %s' %  (str(np.shape(X))))
            # print('Appending data to Y, new shape: %s' %  (str(np.shape(Y))))
        else:
            continue

    return X, Y


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu(x):
    xr = np.maximum(x, 0)
    return xr


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def get_parameters():
    # Goes through each layer of the nn desired structure and returns a dictionary with each parameter
    # E.g. params[W1] will store a matrix of size (W1 layer size x input data size)
    # E.g. params[W2] will store a matrix of size (W2 layer size x W1 layer size) ..etc..
    params = {}
    for layer_index, neural_layer_vals in enumerate(NN_DIMENSION):
        layer_index += 1
        if layer_index < len(NN_DIMENSION):
            print('Initializing layer ', 'W' + str(layer_index), ' with size ', NN_DIMENSION[layer_index], 'x',
                  NN_DIMENSION[layer_index - 1])
            params['W' + str(layer_index)] = np.random.randn(NN_DIMENSION[layer_index],
                                                             NN_DIMENSION[layer_index - 1]) / 1000
            params['b' + str(layer_index)] = np.zeros((NN_DIMENSION[layer_index], 1))

    return params


def forward_propagation(X, parameters):
    linear_caches = []  # stores previous activation, weight and bias
    activation_caches = []  # stores neuron activation values

    neuronOutput = X

    NeuralDepth = len(NN_DIMENSION) - 1

    # Iterate through each layer
    for l in range(1, NeuralDepth):
        A_prev = neuronOutput

        W = parameters['W' + str(l)]

        b = parameters['b' + str(l)]

        linear_cache = (A_prev, W, b)
        linear_caches.append(linear_cache)

        # Neuron linear transformation
        linear_trans = np.dot(W, A_prev) + b

        # Neuron activation function
        neuronOutput = relu(linear_trans)

        activation_caches.append(neuronOutput)

    # Same as above but just on the last layer
    W = parameters['W' + str(NeuralDepth)]
    b = parameters['b' + str(NeuralDepth)]

    final_linear_cache = (A_prev, W, b)

    final_linear_trans = np.dot(W, A_prev) + b

    # Final output of Network
    nn_output = sigmoid(final_linear_trans)

    linear_caches.append(final_linear_cache)

    activation_caches.append(nn_output)

    return nn_output, linear_caches, activation_caches


def backward_propagation(OutputData, Y, linear_caches, activation_caches):
    gradients = {}

    NeuralDepth = len(NN_DIMENSION) - 1

    # Number of training examples
    size = OutputData.shape[1]

    Y = Y.reshape(OutputData.shape)  # after this line, Y is the same shape as OutputData

    # InitiOutputDataizing the backpropagation
    # Derivative of last layer
    dOutputData = - (np.divide(Y, OutputData) - np.divide(1 - Y, 1 - OutputData))

    linear_cache = linear_caches[NeuralDepth - 1]
    activation_cache = activation_caches[NeuralDepth - 1]

    # Activation derivative of final layer
    dZ = sigmoid_backward(dOutputData, activation_cache)

    A_prev, W, b = linear_cache

    size = A_prev.shape[1]

    # Weight derivative of final layer
    dW = (1 / size) * np.dot(dZ, A_prev.T)

    # Bias derivative of final layer
    db = (1 / size) * np.sum(dZ, axis=1, keepdims=True)

    dA_prev = np.dot(W.T, dZ)

    gradients["dA" + str(NeuralDepth)] = dA_prev
    gradients["dW" + str(NeuralDepth)] = dW
    gradients["db" + str(NeuralDepth)] = db

    for l in reversed(range(NeuralDepth - 1)):
        dA = gradients["dA" + str(l + 2)]

        linear_cache = linear_caches[l]
        activation_cache = activation_caches[l]

        dZ = relu_backward(dA, activation_cache)

        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        dW = (1 / size) * np.dot(dZ, A_prev.T)

        db = (1 / size) * np.sum(dZ, axis=1, keepdims=True)

        dA_prev = np.dot(W.T, dZ)

        gradients["dA" + str(l + 1)] = dA_prev
        gradients["dW" + str(l + 1)] = dW
        gradients["db" + str(l + 1)] = db

    return gradients


def get_cost(outputValue, Y, numrecords):
    # Calculates cost of NN
    cost = - (1 / numrecords) * np.sum(
        np.multiply(Y, np.log(outputValue)) + np.multiply(1 - Y, np.log(1 - outputValue)))
    return np.squeeze(cost)


def run_neural(X, Y, parameters):
    cost_cache = []

    # Loop (gradient descent)
    for i in range(0, ITERATIONS):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        nn_output, linear_caches, activation_caches = forward_propagation(X, parameters)

        # Compute cost.
        cost = get_cost(nn_output, Y, Y.shape[1])

        # Backward propagation.
        gradients = backward_propagation(nn_output, Y, linear_caches, activation_caches)

        # Update parameters.
        NeuralDepth = len(NN_DIMENSION) - 1

        # Update rule for each parameter. Use a for loop.
        for l in range(1, NeuralDepth + 1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - LAMBDA * gradients["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - LAMBDA * gradients["db" + str(l)]

        # Print cost
        if i % 200 == 0:
            print ("Cost at iteration ", i, ' is ', cost)
            cost_cache.append(cost)

    # plot the cost
    plt.plot(np.squeeze(cost_cache))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(SIGMA))
    plt.show()

    return params


def testNN(imageFilePath, actualLabel, trainedWeights):
    my_label_y = [actualLabel]
    imgToDisplay = img = mpimg.imread(imageFilePath)
    my_image = loadMyImg(imageFilePath, NUM_PIX).T
    print(np.shape(my_image))
    my_imageNorm = my_image
    classifier_percentage = deep.predict(my_imageNorm, my_label_y, weights)

    result = True

    if classifier_percentage[0] < .5:
        result = False

    plt.imshow(imgToDisplay)

    print ("Provided label is:", bool(actualLabel), ". NN predicts:", result, ".  Predicted chance of TRUE =",
           '{:.1%}'.format(classifier_percentage.item(0)))


X = np.zeros((0, IMG_SIZE))

Y = np.zeros((0, 1))

# Add 'True' images and labels to matrix's X and Y (e.g. pics of stars)
trueX, trueY = appendImagesToDataSet("C:/TestData/Stars", typeVal=1, X=X, Y=Y)

# Add 'False images and labels to the matrix X and Y (e.g. pics of faces)
X, Y = appendImagesToDataSet("C:/TestData/Basketballs", typeVal=0, X=trueX, Y=trueY)

X = X.T
Y = Y.T
params = get_parameters()

# Training the neural network
weights = run_neural(X=X, Y=Y, parameters=params)

# Testing the neural network on a new image
testNN('C:/TestData/Stars/1dfgkje.jpg', 1, weights)

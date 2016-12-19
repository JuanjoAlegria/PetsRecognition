import os
import numpy as np
from NeuralPython.Utils import Builders
import json

basePath = os.path.dirname(os.path.realpath(__file__))

xValidationPath = "xs_test_pca.npy"
yValidationPath = "ys_test_full.npy"

xValidation = np.load(os.path.join(basePath, xValidationPath))[7060:7129]
yValidation = np.load(os.path.join(basePath, yValidationPath))[7060:7129]

config = json.load(open("ConfigNeuralNetwork.json"))
net = Builders.buildNetwork(config)
train = Builders.buildTraining(config)
for x,y in zip(xValidation, yValidation):
	print np.argmax(net.forward(x)), np.argmax(y)

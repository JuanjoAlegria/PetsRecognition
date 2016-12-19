import os
import numpy as np
from NeuralPython.Utils import Builders
import json

basePath = os.path.dirname(os.path.realpath(__file__))
xTrainPath = "xs_train_full.npy"
yTrainPath = "xs_train_full.npy"
xValidationPath = "xs_test_full.npy"
yValidationPath = "xs_test_full.npy"

xTrain = np.load(os.path.join(basePath, xTrainPath))
yTrain = np.load(os.path.join(basePath, yTrainPath))
xValidation = np.load(os.path.join(basePath, xValidationPath))
yValidation = np.load(os.path.join(basePath, yValidationPath))

trainData = [xTrain, yTrain]
validationData = [xValidation, yValidation]
testData = [xValidation, yValidation]

config = json.load(open("ConfigAutoencoder.json"))
net = Builders.buildNetwork(config)
train = Builders.buildTraining(config)
train.setNetwork(net)
train.setData(trainData, validationData, testData)
train.run()
confussionMatrix = train.confussionMatrix(bestResults = True)
print confussionMatrix
np.save("confussionMatrix_autoencoder.npy", confussionMatrix)

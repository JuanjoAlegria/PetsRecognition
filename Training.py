import os
import numpy as np
from NeuralPython.Utils import Builders
import json

basePath = os.path.dirname(os.path.realpath(__file__))
# best = 1469344937, 1469346896, 1469347679, 1469430527, 1469436408, 1469439175/
# daBest = 1469485540, 1469532766
xTrainPath = "xs_train_pca.npy"
yTrainPath = "ys_train_full.npy"
xValidationPath = "xs_test_pca.npy"
yValidationPath = "ys_test_full.npy"

xTrain = np.load(os.path.join(basePath, xTrainPath))
yTrain = np.load(os.path.join(basePath, yTrainPath))
xValidation = np.load(os.path.join(basePath, xValidationPath))
yValidation = np.load(os.path.join(basePath, yValidationPath))

trainData = [xTrain, yTrain]
validationData = [xValidation, yValidation]
testData = [xValidation, yValidation]

config = json.load(open("ConfigNeuralNetwork.json"))
net = Builders.buildNetwork(config)
train = Builders.buildTraining(config)
train.setNetwork(net)
train.setData(trainData, validationData, testData)
train.run()
confussionMatrix = train.confussionMatrix(bestResults = True)
print confussionMatrix
np.save("confussionMatrix.npy", confussionMatrix)

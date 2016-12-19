# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from NeuralPython.Utils import Builders
import json


def plot_confusion_matrix(cm, title=u'Matriz de Confusión Normalizada', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # n_classes = 120
    # tick_marks = np.arange(len(range(n_classes)))
    # plt.xticks(tick_marks, range(n_classes), rotation=45)
    # plt.yticks(tick_marks, range(n_classes))
    plt.tight_layout()
    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predicha')
    plt.show()


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


yTrain = np.argmax(yTrain, axis = 1)
yValidation = np.argmax(yValidation, axis = 1)

config = json.load(open("ConfigNeuralNetwork.json"))
net = Builders.buildNetwork(config)

results = []
confusionMatrix = np.zeros((120, 120))

for i in range(len(xValidation)):
	x = xValidation[i]
	y = yValidation[i]
	result = net.forward(x)
	label = np.argmax(result)
	results.append(result)
	confusionMatrix[y, label] += 1

results = np.array(results)
mAP = 0.0

for i in range(120):
	classScores = results[:, i]
	indexes = np.argsort(-classScores)
	classScores = classScores[indexes]
	groundTruthIndexes = np.where(yValidation == i)[0]
	truePositive = np.zeros(len(yValidation))
	falsePositive = np.zeros(len(yValidation))

	for j in range(len(yValidation)):
		if indexes[j] in groundTruthIndexes:
			truePositive[j] = 1
		else:
			falsePositive[j] = 1

	truePositive = np.cumsum(truePositive)
	falsePositive = np.cumsum(falsePositive)
	recall = truePositive / len(groundTruthIndexes)
	precision = truePositive / (falsePositive + truePositive)
	avgPrecision = 0;


	for t in np.arange(0, 1.1, 0.1):
		p = np.max(precision[recall >= t])
		avgPrecision += p
	avgPrecision /= 11.0
	mAP += avgPrecision
	print i, avgPrecision
	print "Recall = 0.9: ", np.where(recall >= 0.9)[0][0]
	print "Size: ", len(groundTruthIndexes)
	print "Ratio: ", 1.0 * np.where(recall >= 0.9)[0][0] / len(groundTruthIndexes)

print "Correctly classified images: ", confusionMatrix.trace()
print "Mean Accuracy: ", confusionMatrix.trace() / confusionMatrix.sum()
print "Mean Average Precision", mAP / 120

normalized = confusionMatrix / confusionMatrix.sum(axis = 1)
plot_confusion_matrix(normalized)
plot_confusion_matrix(confusionMatrix)

np.save("confusionMatrixFinal.npy", confusionMatrix)




import pdb; pdb.set_trace()  # breakpoint acc39cee //

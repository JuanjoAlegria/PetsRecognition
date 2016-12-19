# -*- coding: utf-8 -*-
import overfeat
import numpy as np
import json
import pickle
import os
from scipy.misc import imresize
from scipy.ndimage import imread
from NeuralPython.Utils import Builders

def preProcessImage(image):
    image = imresize(image, (231, 231)).astype(np.float32)

    h = image.shape[0]
    w = image.shape[1]
    c = image.shape[2]

    # numpy loads image with colors as last dimension, transpose tensor
    image = image.reshape(w * h, c)
    image = image.transpose()
    image = image.reshape(c, h, w)
    return image

def overfeatFeatures(image):
    overfeat.fprop(image)
    features = overfeat.get_output(19).copy()
    return features

def pcaDimensionalityReduction(vector, pca, nFeatures):
    newVector = pca.transform(vector)
    return newVector[:, :nFeatures]


def extractFeatures(image, pca, nFeatures):
    image = preProcessImage(image)
    fullVector = overfeatFeatures(image).reshape((1, 4096))
    pcaVector = pcaDimensionalityReduction(fullVector, pca, nFeatures).reshape((nFeatures))
    return pcaVector


def get5Best(classes, x):
    sortedIndex = np.argsort(-x)
    s = ""
    for i in range(5):
        index = sortedIndex[i]
        prob = x[index]
        className = classes[index][1]
        s += "Predicción " + str(i + 1) + ": Raza " + className + ", con probabilidad " + str(prob) + "\n"
    return s

overfeat.init('/home/juanjo/U/overfeat/data/default/net_weight_0', 0)
rootDir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/TestImages/"
config = json.load(open("ConfigNeuralNetwork.json"))
pca = pickle.load(open("pca.txt"))
net = Builders.buildNetwork(config)
classes = pickle.load(open("classes.pkl"))

for filename in os.listdir(rootDir):
    if filename == ".directory":
        continue
    path = os.path.join(rootDir, filename)
    image = imread(path).astype(np.float32)
    features = extractFeatures(image, pca, 280)
    x = net.forward(features, test = True)
    print filename
    print get5Best(classes, x)

while True:
    path = raw_input("Ingrese ruta a imagen: ")
    if path == "0":
        print "Gracias :)"
        break
    elif not os.path.exists(path):
        print "Ruta no encontrada, intente nuevamente"
        continue
    image = imread(path)

    features = extractFeatures(image, pca, 280)
    result = net.forward(features, test = True)
    print get5Best(classes, result)

# -*- coding: utf-8 -*-
import overfeat
import numpy as np
import os
import json
import pickle
from scipy.misc import imresize
from scipy.ndimage import imread
from NeuralPython.Utils import Builders

def overfeatFeatures(image):
    image = imresize(image, (231, 231)).astype(np.float32)

    h = image.shape[0]
    w = image.shape[1]
    c = image.shape[2]

    # numpy loads image with colors as last dimension, transpose tensor
    image = image.reshape(w * h, c)
    image = image.transpose()
    image = image.reshape(c, h, w)
    # run overfeat on the image
    overfeat.fprop(image)
    features = overfeat.get_output(19).copy()
    return features

def pcaDimensionalityReduction(vector, pca, nFeatures):
    img2 = pca.transform(vector)
    return img2[:, :nFeatures]


def extractFeatures(img, pca, nFeatures):
    fullVector = overfeatFeatures(img).reshape((1, 4096))
    pcaVector = pcaDimensionalityReduction(fullVector, pca, nFeatures).reshape((nFeatures))
    return pcaVector



def print5Best(classes, x):
    sortedIndex = np.argsort(-x)
    s = ""
    for i in range(5):
        index = sortedIndex[i]
        prob = x[index]
        className = classes[index][1]
        s += "Predicción " + str(i + 1) + ": Raza " + className + ", con probabilidad " + str(prob) + "\n"
    print s


overfeat.init('/home/juanjo/U/overfeat/data/default/net_weight_0', 0)
rootDir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/TestImages/"
config = json.load(open("ConfigNeuralNetwork.json"))
pca = pickle.load(open("pca.txt"))
net = Builders.buildNetwork(config)
classes = pickle.load(open("classes.pkl"))


for filename in os.listdir(rootDir):
    path = os.path.join(rootDir, filename)
    image = imread(path).astype(np.float32)
    features = extractFeatures(image, pca, 280)
    x = net.forward(features, test = True)
    print filename
    print5Best(classes, x)




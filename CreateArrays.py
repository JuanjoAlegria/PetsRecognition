# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os


def createArrays(featuresDict, fileDict):
    indexImages = 0
    indexClasses = 0
    xs = []
    ys = []
    string = ""

    for key in featuresDict:
        print key
        y = np.zeros(120)
        y[indexClasses] = 1
        trainList = fileDict[key]
        currentDict = featuresDict[key]
        for filename in trainList:
            string += str(indexImages) + "   " + filename + "\n"
            indexImages += 1
            fileExt = str(filename) + ".jpg"
            if fileExt not in currentDict:
                continue
            vector = currentDict[fileExt]
            vector = vector.reshape((4096, ))
            xs.append(vector)
            ys.append(y)
        indexClasses += 1

    return np.array(xs), np.array(ys), string

if __name__ == "__main__":
    imagesDir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/ResizedImages_5/"
    trainDictDir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/DogsRecognition/Experiment2/trainDict.txt"
    testDictDir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/DogsRecognition/Experiment2/testDict.txt"
    featuresDictDir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/DogsRecognition/Experiment3/features.txt"
    trainDict = pickle.load(open(trainDictDir))
    testDict = pickle.load(open(testDictDir))
    # featuresDict = pickle.load(open(featuresDictDir))
    featuresDict = {}
    for subdir in os.listdir(imagesDir):
        featuresDict[subdir] = pickle.load(open(subdir + "_full.txt"))

    xs, ys, string = createArrays(featuresDict, trainDict)
    np.save("xs_train_full", xs)
    np.save("ys_train_full", ys)
    f = open("index_train_full.txt", "w")
    for line in string:
        f.write(line)
    f.close()

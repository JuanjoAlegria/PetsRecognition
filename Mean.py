# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave


def f(directory, means, n):
    i = 0
    for filename in os.listdir(directory):
        if filename == ".directory": continue
        image = imread(os.path.join(directory, filename)).astype(np.float32)
        means += image.mean(axis = 0).mean(axis = 0)
        i += 1
    n[0] += i

rootDir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/ResizedImages_5/"


def g(directory, means, newDir):
    for filename in os.listdir(directory):
        if filename == ".directory": continue
        image = imread(os.path.join(directory, filename)).astype(np.float32)
        centeredImage = image - means
        imsave(newDir + filename, centeredImage)

means = np.array([121.35576063, 112.08520383, 97.81316867])
baseNewDir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/ResizedImages_6/"

for subdir in os.listdir(rootDir):
    newDir = baseNewDir + subdir + "/"
    if not os.path.exists(newDir):
        os.mkdir(newDir)
    print newDir
    g(os.path.join(rootDir, subdir), means, newDir)

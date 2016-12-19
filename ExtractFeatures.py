# -*- coding: utf-8 -*-
import overfeat
import numpy
import os
import pickle
from scipy.ndimage import imread


def getDictOfFeatures(directory, dirname):
    d = {}
    for filename in os.listdir(directory):
        if filename == ".directory":
            continue
        image = imread(os.path.join(directory, filename)).astype(numpy.float32)
        # numpy loads image with colors as last dimension, transpose tensor
        h = image.shape[0]
        w = image.shape[1]
        c = image.shape[2]
        image = image.reshape(w * h, c)
        image = image.transpose()
        image = image.reshape(c, h, w)
        # run overfeat on the image
        overfeat.fprop(image)
        features = overfeat.get_output(19)
        d[filename] = features.copy()
    pickle.dump(d, open(dirname + "_full.txt", "w"))
    return d


overfeat.init('/home/juanjo/U/overfeat/data/default/net_weight_0', 0)
rootDir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/ResizedImages_6/"
# dTotal = {}

for subdir in os.listdir(rootDir):
    print subdir
    if os.path.exists(subdir + "_full.txt"):
        continue
    d = getDictOfFeatures(os.path.join(rootDir, subdir), subdir)
    # dTotal[subdir] = d


# pickle.dump(dTotal, open("features_full.txt", "w"))

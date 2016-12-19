# -*- coding: utf-8 -*-
import os, pickle

files = open("index_train_full.txt")
d = {}
current = ""
i = 0
for line in files:
    line = line.split("   ")[1]
    x = line.index("_")
    if line[:x] != current:
        current = line[:x]
        d[i] = current
        i += 1

rootdir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/ResizedImages_5"

d2 = {}
for subdir in os.listdir(rootdir):
    i = subdir.index("-")
    d2[subdir[:i]] = subdir[i + 1:]

d3 = {}
for key in d:
    idClass = d[key]
    className = d2[idClass]
    d3[key] = (idClass, className)

pickle.dump(d3, open("classes.pkl", "w"))
classes = open("classes.txt", "w")
for i in d:
    classes.write(str(i) + "   " + d3[i][0] + "   " +  d3[i][1] + "\n")

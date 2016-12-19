# -*- coding: utf-8 -*-
import os
import cv2


def resizeImage(imgPath):
    img = cv2.imread(imgPath)
    h, w, d = img.shape
    resized = cv2.resize(img, (231, 231))
    return resized


def resizeImages(rootDirectory, newDirectory):
    for relativeName in os.listdir(rootDirectory):
        print relativeName
        currentPath = os.path.join(rootDirectory, relativeName)
        if os.path.isdir(currentPath):
            savePath = os.path.join(newDirectory, relativeName)
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            for relativeFile in os.listdir(currentPath):
                filename, file_ext = os.path.splitext(relativeFile)
                resized = resizeImage(os.path.join(currentPath, relativeFile))
                if resized is not None:
                    cv2.imwrite(os.path.join(savePath, relativeFile), resized)

if __name__ == "__main__":
    croppedDir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/CroppedImages/"
    resizedDir = "/home/juanjo/U/Búsqueda por Contenido en Imágenes y Videos/Proyecto/ResizedImages_5/"
    resizeImages(croppedDir, resizedDir)

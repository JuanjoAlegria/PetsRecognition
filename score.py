import numpy as np
import pickle

cm = np.load("confusionMatrixFinal.npy")
classes = pickle.load(open("classes.pkl"))
normalized = cm / cm.sum(axis = 1)
a = np.zeros(120)

for i in range(120):
    a[i] = normalized[i][i]

sortedIndex = np.argsort(-a)
for index in sortedIndex:
    print classes[index][1], a[index]

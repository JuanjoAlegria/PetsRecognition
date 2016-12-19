# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def linear_kernel(x, y):
    """Retorna el producto punto entre x e y."""
    return np.dot(x, y.T)

def gram_matrix_linear(X, Y):
    """Retorna la matriz de gram usando linear_kernel en la evaluación."""
    G = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            G[i,j] = linear_kernel(x, y)
    return G

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    n_classes = 10
    tick_marks = np.arange(len(range(n_classes)))
    plt.xticks(tick_marks, range(n_classes), rotation=45)
    plt.yticks(tick_marks, range(n_classes))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


basePath = os.path.dirname(os.path.realpath(__file__))

xTrainPath = "xs_train_pca.npy"
yTrainPath = "ys_train_full.npy"
xValidationPath = "xs_test_pca.npy"
yValidationPath = "ys_test_full.npy"

xTrain = np.load(os.path.join(basePath, xTrainPath))
yTrain = np.load(os.path.join(basePath, yTrainPath))
xValidation = np.load(os.path.join(basePath, xValidationPath))
yValidation = np.load(os.path.join(basePath, yValidationPath))

yTrain2 = []
for y in yTrain:
    yTrain2.append(np.argmax(y))
yTrain = np.array(yTrain2)

yValidation2 = []
for y in yValidation:
    yValidation2.append(np.argmax(y))
yValidation = np.array(yValidation2)


print "Ajustando"

clf = svm.SVC(kernel=gram_matrix_linear)
clf.fit(xTrain, yTrain)

print "Guardando"
pickle.dump(clf, open("clf_pca.txt", "w"))

y_pred = clf.predict(xValidation)
print "Accuracy on test set: %7.4f" % accuracy_score(yValidation, y_pred, normalize=True)

cm_linear = confusion_matrix(yValidation, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization, linear kernel')
print(cm_linear)
plt.figure()
plot_confusion_matrix(cm_linear)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_linear_normalized = cm_linear.astype('float') / cm_linear.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix, linear kernel')
print(cm_linear_normalized)
plt.figure()
plot_confusion_matrix(cm_linear_normalized, title='Normalized confusion matrix, linear kernel')

plt.show()



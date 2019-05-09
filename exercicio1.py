#!/usr/bin/env python
# -*- coding: utf-8 -*-

# test knn loop n_neighbors = 1 to 25
import array as arr
from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)
values_performance = {}

for i in range(1,26):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train )
    prediction = knn.predict(x_test)
    score =  metrics.accuracy_score(y_test,prediction)
    values_performance[i] = round(score,4)

print values_performance 


import matplotlib.pyplot as plt


plt.plot(list(values_performance.keys()), list(values_performance.values()))
plt.xlabel("Valores de K")
plt.ylabel("Performance")
plt.show()
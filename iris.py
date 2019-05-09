#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
iris = load_iris()

#observações
x = iris.data
print (x)

#target
y  = iris.target
print (y)

#shape(length) das observações
print (iris.data.shape)

#shape(length) da target
print (iris.target.shape)

# Usando KNN para treinamento dos dados

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)

#fazer previsão

species =  knn.predict([[5.1,3.5,1.4,0.2]])

print "Previsão: Está relacionado com a planta ", iris.target_names[species]


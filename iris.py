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

# Separar dados em dois grupos(um para teste e outro para treinamento)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

# Treinar com os novos valores para teste
knn.fit(x_train,y_train)
previsao = knn.predict(x_test)
print "Previsão a partir do exemplos teste : ", previsao
print "O resultado final deveria ser :       ", y_test
# Verificar a acuracia dos resultados

from sklearn import metrics
acertos = metrics.accuracy_score(y_test, previsao)
print acertos





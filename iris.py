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


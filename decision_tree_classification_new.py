# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:50:52 2023

@author: ivoto
"""

#Plantilla de clasificacion 

#Importamo librerias
import numpy as np #Para trabajar con math
import matplotlib.pyplot as plt # Para la vizualizacion de datos 
import pandas as pd #para la carga de datos 

#Importamos el dataSet 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values 
y = dataset.iloc[:, 4].values 


#Training & Test 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0 )


#Ajustar el clasificador
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy")
classifier.fit(X_train, y_train )



#Prediccion de los resultados con el conjunto de test 
y_pred = classifier.predict(X_test)

#Elaboramos la matriz de confusion 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

 
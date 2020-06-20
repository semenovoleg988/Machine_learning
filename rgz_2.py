#!/usr/bin/env python
# coding: utf-8
# 3 датасет

import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing, metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from scipy.stats import sem
import csv
from collections import Counter


#Загрузка датасета
data = []
fieldnames = []
target = []

with open('smartphone_activity_dataset.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    fieldnames = reader.fieldnames
    
    for row in reader:
            data.append(row)
            target.append(row['activity'])

del  fieldnames[0]
# for row in data:
#     del row["url"]
del data[150:]
del target[150:]



X = pd.DataFrame(data, columns=fieldnames)
y = pd.Series(target, name='activity')
df = X.copy()
df['activity'] = y
df.info()
data = y
duplicates = {k for k, v in Counter(data).items() if v < 2}
print("__________", duplicates)


print("__________")
print('classes:', y.nunique())


# Нормализация данных

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled)
X.head()
print(X)


def find_best_model_by_LOO(knn, param_grid=None):
    if param_grid is None:
        metrics = ['manhattan', 'euclidean', 'chebyshev']
        neighbors = np.arange(1, 50)
        param_grid = {'metric': metrics, 'n_neighbors': neighbors}
    
    gs = GridSearchCV(estimator=knn,
                      cv=LeaveOneOut(), 
                      param_grid=param_grid, 
                      scoring='accuracy', 
                      verbose=1, 
                      n_jobs=-1)
    gs.fit(X, y)
    
    return gs


# ### Алгоритм ближайшего соседа

# knn1 = KNeighborsClassifier(n_neighbors=1)


# # **Метод скользящего контроля**

# scores = cross_val_score(knn1, X, y, cv=10, scoring='accuracy')
# print("Алгоритм ближайшего соседа\n**Метод скользящего контроля**")
# print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


# ### Алгоритм k ближайших соседей

# knn = KNeighborsClassifier()


# # **Метод скользящего контроля**

# scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
# print("Алгоритм k ближайших соседей\n**Метод скользящего контроля**")
# print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


# # **Метод Leave-one-out**

# gs = find_best_model_by_LOO(knn)
# scores = cross_val_score(gs.best_estimator_, X, y, cv=10, scoring='accuracy')

# print('Best model params by LOO:', gs.best_params_)
# print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


# # ### Алгоритм k взвешенных ближайших соседей

# knn_w = KNeighborsClassifier(weights='distance')


# # **Метод скользящего контроля**


# scores = cross_val_score(knn_w, X, y, cv=10)
# print("лгоритм k взвешенных ближайших соседей\n**Метод скользящего контроля**")
# print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


# # **Метод Leave-one-out**


# gs = find_best_model_by_LOO(knn_w)
# scores = cross_val_score(gs.best_estimator_, X, y, cv=10, scoring='accuracy')

# print('Best model params by LOO:', gs.best_params_)
# print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


### Метод парзеновского окна постоянной ширины

knn = RadiusNeighborsClassifier()
metrics = ['manhattan', 'euclidean', 'chebyshev']
param_grid = {'metric': metrics, 'radius': np.arange(100, 130)}

gs = find_best_model_by_LOO(knn, param_grid)
scores = cross_val_score(gs.best_estimator_, X, y, cv=10, scoring='accuracy')
print("Метод парзеновского окна постоянной ширины")
print('Best model params by LOO:', gs.best_params_)
print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))


# ### Метод парзеновского окна переменной ширины

# def weights(distances):
#     return 1.0 - distances/distances.max()

# knn_weighted = KNeighborsClassifier(weights=weights)
# gs = find_best_model_by_LOO(knn_weighted)
# scores = cross_val_score(gs.best_estimator_, X, y, cv=10, scoring='accuracy')

# print("Метод парзеновского окна переменной ширины")
# print('Best model params by LOO:', gs.best_params_)
# print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))

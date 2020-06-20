#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix


# ## Датасет
# ### Стандартный MNIST из sklearn

# In[2]:


from sklearn.datasets import load_digits
digits = load_digits()
data = digits['data']
target = digits['target']


# In[3]:


data.shape


# In[4]:


def display_digits(digits):
    N = len(digits)
    for i, digit in enumerate(digits):
        plt.subplot((N//4)+1, 4, i+1)
        pixels = digit.reshape((8,8))
        plt.imshow(pixels, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return 

display_digits(data[:12])


# ### Нормализация данных по яркости 

# In[5]:


from sklearn.preprocessing import normalize
data = normalize(data, norm='max', axis=1)

display_digits(data[:12])


# ### В качестве предварительной обработки используем уменьшение  размерности с помошью PCA, исходя из того что в картинках встречается много кореллирующих фич 

# In[6]:


from sklearn.decomposition import PCA


X = PCA(16).fit_transform(data)


# ### Изобразим две первых главных компонент на проскости

# In[7]:


def plot_digits(X, y):
    plt.figure(figsize=(8,8))
    for i in range(10):
        plt.scatter(X[y==i][:,0], X[y==i][:,1])
        
    plt.legend(range(10))
    plt.show()
    
plot_digits(X, target)


# ### Определим метрики и перебор по соседям

# In[8]:


metrics = ['manhattan', 
           'euclidean', 
           'chebyshev']

neighbors = [1,3,5,10,20,30]


# ### Опеределим метод для подбора пораметров с помощью LOO и метод для оценки модели с помощью скользящего контроля 

# In[9]:


from sklearn.model_selection import cross_val_score, GridSearchCV, LeaveOneOut

def evaluate_model(clf):
    return cross_val_score(clf, X, target, cv=5, scoring='accuracy').mean()


def grid_search_model(clf, param_grid):
    gs = GridSearchCV(estimator=clf,
                      cv=LeaveOneOut(), 
                      param_grid=param_grid, 
                      scoring='accuracy', 
                      verbose=1, n_jobs=16)
    gs.fit(X, target)
    return gs


def experiment(clf, param_grid):
    gs = grid_search_model(clf, param_grid)
    score = evaluate_model(gs.best_estimator_)
    
    print('Best model by LOO is:', gs.best_estimator_)
    print('Best model params by LOO are:', gs.best_params_)
    print('Best model score on CV(5) is:', score)


# ### 1. Narest Neighbour method

# In[10]:


from sklearn.neighbors import KNeighborsClassifier, KernelDensity, RadiusNeighborsClassifier
nn = KNeighborsClassifier(n_neighbors=1)
param_grid = {'metric': metrics}
experiment(nn, param_grid)


# ### 2. k-NN method

# In[11]:


knn = KNeighborsClassifier(n_jobs=16)
param_grid = {'metric': metrics, 'n_neighbors': neighbors}
experiment(knn, param_grid)


# ### 3. Weighted k-NN method

# In[12]:


knn_weighted = KNeighborsClassifier(weights='distance',n_jobs=16)
param_grid = {'metric': metrics, 'n_neighbors': neighbors}
experiment(knn_weighted, param_grid)


# ### 4. Fixed parzen window 

# In[13]:


knn_weighted = RadiusNeighborsClassifier(n_jobs=16)
param_grid = {'metric': metrics, 'radius': [7, 7.5,8,9,10]}
experiment(knn_weighted, param_grid)


# ### 5. Non-fixed parzen window 

# In[14]:


def weights(distances):
    return 1.0 - distances/(distances.max() + 0.0000000001)

non_fixed_parzen_window = KNeighborsClassifier(weights=weights,n_jobs=-1)
param_grid = {'metric': metrics,'n_neighbors': neighbors}
experiment(non_fixed_parzen_window, param_grid)


# ### Результаты

# In[15]:


algorithms = ['NN', 'kNN', 'Weighted kNN', 'Fixed parzen window', 'Parzen window']
best_params = [{'metric': 'euclidean'},
               {'metric': 'euclidean', 'n_neighbors': 5},
               {'metric': 'euclidean', 'n_neighbors': 5},
              {'radius': 7, 'metric': 'manhattan'},
              {'metric': 'manhattan', 'n_neighbors': 10}]
scores = [0.9555873489383526,
          0.9589470055133636, 
          0.9583836252316736,
          0.8601322619311172,
          0.9516821717961242]
pd.DataFrame(dict(algorithm=algorithms, best_params=best_params, score=scores))


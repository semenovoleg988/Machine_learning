# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Загружаем выборку точек

# In[2]:


df = pd.read_excel('RGZ3-MF5152.xlsx', header=None)
df = df.fillna(method='ffill').dropna().set_index(0)
X = df.loc['Семенов О.К.'][1:].values.astype(np.float64)


# In[3]:


plt.figure(figsize=(8,8))
plt.scatter(X[:,0], X[:,1])
plt.show()


# ### Отрисуем гистограмму попарных расстояний между точками

# In[4]:


from sklearn.metrics import pairwise_distances
rho = pairwise_distances(X)


# In[5]:


plt.figure(figsize=(8,6))
plt.hist(np.reshape(rho, -1), bins=100, normed=True)
plt.grid()
plt.show()


# ### Отрисуем зависимость числа компонент от R

# In[6]:


from scipy.cluster.hierarchy import fclusterdata

rs = np.arange(0.01, 0.8, 0.01)
n_clusters = [len(set(fclusterdata(X, r, criterion='distance'))) for r in rs]


# In[7]:


plt.figure(figsize=(8,8))
plt.plot(rs, n_clusters, 'o-', markersize=5)
plt.grid()
plt.show()


# In[8]:


n_clusters_r_map = dict(zip(n_clusters, rs))
n_clusters_r_map

print(n_clusters_r_map)
# ### Отрисуем разбиения на интересующее нас количество кластеров

# In[17]:


for i in range(19, 23):
   labels = fclusterdata(X, n_clusters_r_map[i], criterion='distance')
   plt.figure(figsize=(8,8))
   plt.scatter(X[:,0], X[:,1], c=labels)
   plt.title('R: {}, n_clusters: {}'.format(n_clusters_r_map[i], i))
   plt.show()


# ### Отрисуем разбиение на кластеры с помощью агломеративной кластеризации, имея возможность задать колличество кластеров явно

# In[22]:


from sklearn.cluster import AgglomerativeClustering

for i in range(1, 10):
    labels = AgglomerativeClustering(n_clusters=i, linkage='ward').fit(X).labels_
    plt.figure(figsize=(8,8))
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.title('n_clusters: {}'.format(i))
    plt.show()
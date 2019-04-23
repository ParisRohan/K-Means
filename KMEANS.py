import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset=pd.read_csv("DATAset.csv")	
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

kmeans=KMeans(n_clusters=2)
kmeans.fit(x,y)

print("Final generated cluster's centroid values= ")
print(kmeans.cluster_centers_)

plt.scatter(x,y,c=kmeans.labels_,cmap="rainbow")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Clustering using K-Means")
plt.show()


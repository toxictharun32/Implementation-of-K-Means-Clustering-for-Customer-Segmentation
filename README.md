# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize the Dataset Load the customer dataset and select the required features such as Annual Income and Spending Score.

2.Choose the Number of Clusters (K) Decide the number of clusters to form using methods like the Elbow Method.

3.Assign Data Points to Clusters Calculate the distance between each data point and cluster centroid, then assign each point to the nearest centroid.

4.Update Cluster Centroids Recalculate the centroid of each cluster by taking the mean of all data points assigned to it.

5.Repeat Until Convergence Continue assigning points and updating centroids until the centroids no longer change significantly and final customer segments are formed.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: 212224240171
RegisterNumber: tharun.b
*/

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df= pd.read_csv("Mall_Customers.csv")
df.head()
df.info()
df.isnull().sum()
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
from sklearn.cluster import KMeans 
wcss=[] 
for i in range(1,11): 
    kmeans=KMeans(n_clusters=i,init="k-means++",n_init=10) 
    kmeans.fit(df.iloc[:,3:]) 
    wcss.append(kmeans.inertia_) 
import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.xlabel("No of clusters")
plt.ylabel("wcss")
plt.title("Elbow method")
plt.show()
km=KMeans(n_clusters=5,n_init=10)
km.fit(df.iloc[:,3:])
y_pred=km.predict(df.iloc[:,3:])
print(y_pred)
df["cluster"]=y_pred
dt0=df[df["cluster"]==0]
dt1=df[df["cluster"]==1]
dt2=df[df["cluster"]==2]
dt3=df[df["cluster"]==3]
dt4=df[df["cluster"]==4] 
plt.figure()
plt.scatter(dt0["Annual Income (k$)"],dt0["Spending Score (1-100)"],c="red",label="cluster1") 
plt.scatter(dt1["Annual Income (k$)"],dt1["Spending Score (1-100)"],c="black",label="cluster2")
plt.scatter(dt2["Annual Income (k$)"],dt2["Spending Score (1-100)"],c="blue",label="cluster3")
plt.scatter(dt3["Annual Income (k$)"],dt3["Spending Score (1-100)"],c="green",label="cluster4") 
plt.scatter(dt4["Annual Income (k$)"],dt4["Spending Score (1-100)"],c="magenta",label="cluster5")
plt.legend()
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.show()
```

## Output:
<img width="704" height="461" alt="image" src="https://github.com/user-attachments/assets/c2113315-95ee-42f5-9aff-06d18258556f" />

<img width="661" height="390" alt="image" src="https://github.com/user-attachments/assets/ad4e07c1-ffbd-48c9-bd3e-2cba3425271d" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

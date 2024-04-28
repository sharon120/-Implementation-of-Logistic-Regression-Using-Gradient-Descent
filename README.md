# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1: Use the standard libraries in python for finding linear regression.

STEP 2: Set variables for assigning dataset values

STEP 3: Import linear regression from sklearn.

STEP 4: Predict the values of array

STEP 5: Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sharon Harshini L M
RegisterNumber: 212223040193

import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r'Mall_Customers.csv')

data.head()
data.info()
data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])

y_pred = km.predict(data.iloc[:,3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")

*/
```

## Output:
Head()

![325059598-b456b834-cd05-4854-ae81-9bdfed9b3c6a](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/d739863b-393f-4a50-be8f-c55e0c10a66b)
Info()

![325059830-e57cb1e9-621e-46e2-83bb-b4886f663fa8](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/3d5a59c7-d617-43da-9658-f68ce2456e09)
isnull.sum()

![325060452-c3e12196-8ca0-40c6-bdfe-e9a30411566c](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/430d995f-744f-4b60-8996-428f9fe985d8)
Elbow method

![325060077-75bcf084-8bd2-425d-ac42-72ae8b6b4870](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/4ecf2b3f-6427-444d-9ea5-d1f52d0dff5a)
Fitting the no of clusters

![325060182-a6d6d758-7171-479b-9d41-6a7d23000f07](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/ab483e14-15cb-4159-bf34-e0bc60ddba36)
Prediction of Y

![325060267-7582d183-b350-4875-b532-39cab3dcc4cb](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/9934a8ac-90c1-4bb1-9af5-1f26e8d03134)
Customer Segments

![325060545-3e08df8f-5472-4583-aeef-3019bf0b9f85](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/67874183-f6cc-4334-a9fb-f0462be9f7d7)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


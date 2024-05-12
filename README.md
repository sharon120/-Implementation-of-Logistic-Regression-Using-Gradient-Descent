# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sharon Harshini L M
RegisterNumber:  212223040193

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset= dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y ):
    h = sigmoid(X.dot(theta)) 
    return -np.sum(y *np.log(h)+ (1- y) *np.log(1-h))
def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot (h-y) /m
        theta-=alpha * gradient
    return theta
theta= gradient_descent (theta,X,y,alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where( h >= 0.5,1 , 0)
    return y_pred

y_pred= predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y) 
print("Accuracy:", accuracy)
print(Y)
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
DATASET:

![326230954-37f57e48-ca78-414a-8602-ba1732d1d449](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/864060b7-5a1f-418b-918d-e758030c2877)

LABELLING DATA:

![326230987-43d27412-c749-4776-babf-fb7a4098d924](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/bdd37ed4-5e58-4725-9d93-9a9d59655d63)

LABELLING THE COLUMN:

![326231009-921e5ebe-e913-421a-afc6-9c5dfb637ce9](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/8500eb23-3eb5-4572-ab34-48bdf8da893d)

DEPENDENT VARIABLES:

![326231032-167f9a29-cd37-4d88-93a3-75db61629bab](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/fb1cc7fe-4aac-41a5-a91f-c3671b58c8c1)

ACCURACY:

![326231082-7dc8c795-1068-4811-9bc3-31c0135670b3](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/01b3bbf8-d1b1-464c-9308-5e5c7721adb6)

Y:

![326231098-00a9d0da-d3c7-478a-a244-365579b0c59f](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/794f863b-5bab-4ac7-86f1-ddb20ba485b5)

Y_PRED:

![326231105-a2a0216f-a843-4f2e-ad18-ff190132e9a9](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/8ed2082c-e7a7-4c6f-b026-df1bd4a7ed3c)

NEW PREDICTED DATA:

![326231136-34787685-822b-4740-92ed-28275ebf2c18](https://github.com/sharon120/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149555539/32be41b3-d621-456e-883f-3c8cf0e3a7f1)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


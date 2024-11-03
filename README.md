# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### STEP 1 : Start
#### STEP 2 : Preprocessing the data
#### STEP 3 : Feature Extraction
#### STEP 4 : Training the SVM model
#### STEP 5 : Model Evalutaion
#### STEP 6 : Stop


## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: Nithilan S
RegisterNumber:  212223240108

```
```py
import pandas as pd
data = pd.read_csv("spam.csv",encoding = 'windows-1252')
from sklearn.model_selection import train_test_split
data
data.shape
x = data['v2'].values
y = data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.35,random_state = 48)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc = accuracy_score(y_test,y_pred)
acc
con = confusion_matrix(y_test,y_pred)
print(con)
cl = classification_report(y_test,y_pred)
print(cl)
```

## Output:
#### Data:
![image](https://github.com/user-attachments/assets/18f78bdb-bd26-49fc-a60b-daf2c7a47827)
#### Data.shape:
![image](https://github.com/user-attachments/assets/6df4d79e-a214-415b-bb9f-728b82192669)
#### x_train:
![image](https://github.com/user-attachments/assets/3dc2d767-2da9-41d7-8ca7-3a2438050091)
#### Accuracy:
![image](https://github.com/user-attachments/assets/900ecb44-34a3-4c5b-8d44-e0c6e7a998ba)
#### Confusion Matrix and Classification report:
![image](https://github.com/user-attachments/assets/1025a96d-05d0-4741-b0d6-ddc661fa2e71)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

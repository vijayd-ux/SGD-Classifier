# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: VIJAY D
RegisterNumber:  212225230300

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("Placement_Data.csv")

X = data.drop(["status", "salary", "sl_no"], axis=1)
y = data["status"]

X = pd.get_dummies(X, drop_first=True)

feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_names
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names
)

model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

new_student_dict = dict.fromkeys(feature_names, 0)
new_student_dict["ssc_p"] = 67
new_student_dict["hsc_p"] = 91
new_student_dict["degree_p"] = 58
new_student_dict["etest_p"] = 88
new_student_dict["mba_p"] = 67

new_student_df = pd.DataFrame([new_student_dict])

new_student_scaled = pd.DataFrame(
    scaler.transform(new_student_df),
    columns=feature_names
)

pred = model.predict(new_student_scaled)
print(pred[0])



*/
```

## Output:
![prediction of iris species using SGD Classifier](sam.png)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and separate input and output variables.

2.Split the data into training and testing sets.

3.Train the linear regression model using the training data.

4.Predict the output for the test data and evaluate the results.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: BUSHRA FATHIMA I
RegisterNumber: 212225040051
*/
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier(max_iter=1000, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

```

## Output:
<img width="572" height="300" alt="Screenshot 2026-05-12 114253" src="https://github.com/user-attachments/assets/dae221d7-3657-4ae3-b61a-aef557f1dc84" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

# IEE 520: Fall 2019
# Naive Bayes (9/5/19)

# For compatibility with Python 2
from __future__ import print_function

# To load datasets
from sklearn import datasets

# To import the classifier (Naive Bayes)
from sklearn.naive_bayes import GaussianNB

# To measure accuracy
from sklearn import metrics

# To support plots
import matplotlib.pyplot as plt

# Load data
# Variable DESCR contains a nice description of the database
mydata = datasets.load_breast_cancer()
print('Description of the data:')
print(mydata.DESCR)
X = mydata.data
y = mydata.target
print('Features:')
print(X)
print('Targets:')
print(y)

# Train the model and predict
model = GaussianNB()
model.fit(X, y)
yhat = model.predict(X)

# Model evaluation
print('Accuracy:')
print(metrics.accuracy_score(y, yhat))
print('Classification report:')
print(metrics.classification_report(y, yhat))

# Confusion matrix
# You need to install pandas_ml in order to use that!
# conda install -c conda-forge pandas_ml
from pandas_ml import ConfusionMatrix
print('Confusion matrix:')
cm = ConfusionMatrix(y, yhat)
print(cm)
print('Stats:')
cm.print_stats()
cm.plot(backend='seaborn', annot=True, fmt='g')
plt.show()

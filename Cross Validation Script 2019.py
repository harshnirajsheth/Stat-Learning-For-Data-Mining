# IEE 520: Fall 2019
# Naive Bayes (9/5/19)
# Klim Drobnyh (klim.drobnyh@asu.edu)

# For compatibility with Python 2
from __future__ import print_function

# To load numpy
import numpy as np

# To load datasets
from sklearn import datasets

# To import the classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# To measure accuracy
from sklearn import metrics
from sklearn import model_selection

# To import the scalers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer

# To support plots
import matplotlib.pyplot as plt

# Set to True if you want to scale the data
SCALE = False

# You can choose a different scaler
SCALER_CLASS = StandardScaler
# SCALER_CLASS = MinMaxScaler
# SCALER_CLASS = Binarizer

# You can choose a different classifier
CLASSIFIER_CLASS = GaussianNB
# CLASSIFIER_CLASS = MLPClassifier

# Load the data
# Variable DESCR (mydata.DESCR) contains a nice description of the database
mydata = datasets.load_breast_cancer()
X = mydata.data[:, 0:9]
y = mydata.target
seedMLP = 2357

# Train the model and predict
scaler = SCALER_CLASS()
model = CLASSIFIER_CLASS()
model.fit(X, y)
yhat = model.predict(X)

# Model evaluation
print('Accuracy:')
print(metrics.accuracy_score(y, yhat))
print('Classification report:')
print(metrics.classification_report(y, yhat))

# Train confusion matrix
# You need to install pandas_ml in order to use that!
# conda install -c conda-forge pandas_ml
from pandas_ml import ConfusionMatrix

cm = ConfusionMatrix(y, yhat)
print(cm)

cm.print_stats()
ax = cm.plot(backend='seaborn', annot=True, fmt='g')
ax.set_title('Train Confusion Matrix')
plt.show()


# Cross-validation
seed = 3421
np.random.seed(seed)
actuals = []
probs = []
hats = []
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, y):
    # print('train: %s, test: %s' % (train, test))
    # Train classifier on training data, predict test data
    if SCALE:
        scaler.fit(X[train]) # Learn scaling parameters on training data
        Xtrain = scaler.transform(X[train])
        Xtest = scaler.transform(X[test]) # Apply transform to test data
    else:
        Xtrain = X[train]
        Xtest = X[test]
    model.fit(Xtrain, y[train])
    foldhats = model.predict(Xtest)
    foldprobs = model.predict_proba(Xtest)[:,1] # Class probability estimates for ROC curve
    actuals = np.append(actuals, y[test]) # Combine targets, then probs, and then predictions from each fold
    probs = np.append(probs, foldprobs)
    hats = np.append(hats, foldhats)

# Model evaluation
print('Accuracy:')
print(metrics.accuracy_score(y, yhat))
print('Classification report:')
print(metrics.classification_report(y, yhat))

# Test (cross-validation) confusion matrix
# You need to install pandas_ml in order to use that!
# conda install -c conda-forge pandas_ml
from pandas_ml import ConfusionMatrix

cm = ConfusionMatrix(y, yhat)
print(cm)

cm.print_stats()
ax = cm.plot(backend='seaborn', annot=True, fmt='g')
ax.set_title('Test Confusion Matrix')
plt.show()

# ROC curve code here is for 2 classes only
if len(mydata.target_names) == 2: 
    fpr, tpr, threshold = metrics.roc_curve(actuals, probs)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

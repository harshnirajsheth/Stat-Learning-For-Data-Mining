# IEE 520: Fall 2019
# Neural Networks (9/19/19)

# For compatibility with Python 2
from __future__ import print_function

# To not to show warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# To load datasets
from sklearn import datasets

# To import the classifier (Neural Networks)
import sklearn.neural_network as NN

# To scale the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# To measure accuracy
from sklearn import metrics

# To split data to train and test
from sklearn.model_selection import train_test_split

# To perform parameter search
from sklearn.model_selection import GridSearchCV

# To support plots
import matplotlib.pyplot as plt


# Part 1. Breast cancer dataset (classification)

print('Load the dataset')
# The dataset consists of 569 instances, 
# 357 benign and 212 malignant. 
# 30 features were collected. 
# This is a classification problem with 2 target classes.
X, y = datasets.load_breast_cancer(True)
X.shape

print('Split to train and test')
# Here, the data will be split to train and test. 
# Only 10% of data will be used for testing purpose.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=520)

print('Train a simple neural network')
# Now, let's train a simple neural network 
# with default parameters, just 10 neurons and one hidden layer.
model = NN.MLPClassifier(hidden_layer_sizes=(10,), random_state=520)
model.fit(X_train, y_train)
y_hat_test = model.predict(X_test)

print('Train score:', model.score(X_train, y_train))
print('Test score:', model.score(X_test, y_test))

# You need to install pandas_ml in order to use that!
# conda install -c conda-forge pandas_ml
from pandas_ml import ConfusionMatrix
cm = ConfusionMatrix(y_test, y_hat_test)
cm.print_stats()
cm.plot(backend='seaborn', annot=True, fmt='g')
plt.show()

# Conclusion: the model is simple and underfits the data

print('Random state')
model = NN.MLPClassifier(hidden_layer_sizes=(10,), random_state=520)
model.fit(X_train, y_train)
print('Train score:', model.score(X_train, y_train))
print('Test score:', model.score(X_test, y_test))

model = NN.MLPClassifier(hidden_layer_sizes=(10,), random_state=521)
model.fit(X_train, y_train)
print('Train score:', model.score(X_train, y_train))
print('Test score:', model.score(X_test, y_test))

model = NN.MLPClassifier(hidden_layer_sizes=(10,), random_state=522)
model.fit(X_train, y_train)
print('Train score:', model.score(X_train, y_train))
print('Test score:', model.score(X_test, y_test))

print('We can see here that result might change depending just on weight initialization.')

print('Scale the data')
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = NN.MLPClassifier(hidden_layer_sizes=(10,), random_state=520)
model.fit(X_train_scaled, y_train)
y_hat_test = model.predict(X_test_scaled)
print('Train score:', model.score(X_train_scaled, y_train))
print('Test score:', model.score(X_test_scaled, y_test))
cm = ConfusionMatrix(y_test, y_hat_test)
cm.print_stats()
cm.plot(backend='seaborn', annot=True, fmt='g')
plt.show()


# Part 2. Diabetes dataset (regression)

print('Load the dataset')
# The dataset consists of 442 instances. 
# 10 features were collected. 
# This is a regression problem, 
# the target is a quantitative measure of disease progression one year after baseline.

# Here, the data will be split to train and test. 
# Only 10% of data will be used for testing purpose.
X, y = datasets.load_diabetes(True)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=520)

print('Scale the data')
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Scale target variable, so we can use logistic activation function
scaler = MinMaxScaler(feature_range=(0.1, 0.9))
scaler.fit(y_train.reshape((y_train.shape[0], 1)))
y_train = scaler.transform(y_train.reshape((y_train.shape[0], 1))).reshape((y_train.shape[0],))
y_test = scaler.transform(y_test.reshape((y_test.shape[0], 1))).reshape((y_test.shape[0],))

print('Train a simple neural network')
model = NN.MLPRegressor(random_state=520)
model.fit(X_train, y_train)
y_hat_test = model.predict(X_test)
y_hat_train = model.predict(X_train)

print('Train score:', model.score(X_train, y_train))
print('Test score:', model.score(X_test, y_test))

def predicted_vs_actual(predicted, actual):
    plt.plot(predicted, actual, 'ro')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Predicted vs Actual')
    plt.show()
    
results_plot = predicted_vs_actual

results_plot(y_hat_train, y_train)
results_plot(y_hat_test, y_test)

print('Train more complex neural networks')
model = NN.MLPRegressor(hidden_layer_sizes = (40, 40, 40, 40), activation='relu', random_state=520, max_iter = 10000, tol=1e-10)
model.fit(X_train, y_train)
y_hat_train = model.predict(X_train)
y_hat_test = model.predict(X_test)
print('Train score:', model.score(X_train, y_train))
print('Test score:', model.score(X_test, y_test))
results_plot(y_hat_train, y_train)
results_plot(y_hat_test, y_test)

# What problem do we have? Overfitting!

# Based on the following:
# Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

import numpy as np
from sklearn.model_selection import validation_curve

def plot_validation_curve(estimator, title, X, y, ylim=None, cv=None,
                          n_jobs=1, iterations=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of iterations")
    plt.ylabel("Score")
    train_scores, test_scores = validation_curve(model, X, y, "max_iter",
                                                 iterations,
                                                 cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(iterations, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(iterations, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(iterations, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(iterations, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

print('Let\'s plot a validation curve')
model = NN.MLPRegressor(hidden_layer_sizes=(40, 40, 40, 40), activation='relu', random_state=520, tol=1e-10)
plot_validation_curve(model, "Validation Curve (Neural Network, MLP)", X_train, y_train, ylim=(0.0, 1.01), cv=5, iterations=list(np.arange(5, 150, 3)))
plt.show()

print('Let\'s train the optimal number of iterations')
model = NN.MLPRegressor(hidden_layer_sizes=(40, 40, 40, 40), activation='relu', random_state=520, max_iter=23, tol=1e-10, verbose=True)
model.fit(X_train, y_train)
y_hat_train = model.predict(X_train)
y_hat_test = model.predict(X_test)
print('Train score:', model.score(X_train, y_train))
print('Test score:', model.score(X_test, y_test))
results_plot(y_hat_train, y_train)
results_plot(y_hat_test, y_test)

print('Let\'s use cross-validation to find optimal parameters')
NN_CV = GridSearchCV(NN.MLPRegressor(hidden_layer_sizes=(40, 40, 40, 40), activation='relu', random_state=520, max_iter=10000, tol=1e-10, early_stopping=True),
                     cv=5,
                     param_grid={
                         "alpha": [0.001, 0.1, 1, 10],
                         "learning_rate_init": [0.001, 0.01, 0.1, 1]
                     })
NN_CV.fit(X_train, y_train)
print('The parameters found by CV search:')
print(NN_CV.best_params_)
model = NN_CV
y_hat_train = model.predict(X_train)
y_hat_test = model.predict(X_test)
print('Train score:', model.score(X_train, y_train))
print('Test score:', model.score(X_test, y_test))
results_plot(y_hat_train, y_train)
results_plot(y_hat_test, y_test)

print('Now, let\'s try logistic activation function')
NN_CV = GridSearchCV(NN.MLPRegressor(hidden_layer_sizes=(40, 40, 40, 40), activation='logistic', random_state=520, max_iter=10000, early_stopping=True),
                     cv=5,
                     param_grid={
                         "alpha": [0.0001, 0.001, 0.1, 1],
                         "learning_rate_init": [0.001, 0.01, 0.1, 1]
                     })
NN_CV.fit(X_train, y_train)
print('The parameters found by CV search:')
print(NN_CV.best_params_)
model = NN_CV
y_hat_train = model.predict(X_train)
y_hat_test = model.predict(X_test)
print('Train score:', model.score(X_train, y_train))
print('Test score:', model.score(X_test, y_test))
results_plot(y_hat_train, y_train)
results_plot(y_hat_test, y_test)

import numpy as np
import pandas as pd

# For chapter 4
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# For chapter 5
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import time

# For chapter 6
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from google.colab import files
uploaded = files.upload()

run_walk = pd.read_csv("dataset.csv")

run_walk.head()

#Data pre-processing
x = run_walk.drop(['date', 'time', 'username', 'activity'], axis=1)
y = run_walk['activity']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)

#SGD-Classifier
#Logistic Regression with SGD training
clf = SGDClassifier(loss="log", penalty="l2")
clf.fit(trainX, trainY)

y_pred = clf.predict(testX)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

n_iters = [5, 10, 20, 50, 100, 1000]
scores = []
for n_iter in n_iters:
    clf = SGDClassifier(loss="log", penalty="l2", max_iter=n_iter)
    clf.fit(trainX, trainY)
    scores.append(clf.score(testX, testY))
  
plt.title("Effect of n_iter")
plt.xlabel("n_iter")
plt.ylabel("score")
plt.plot(n_iters, scores)

#Linear SVM with SGD training
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(trainX, trainY)

y_pred = clf.predict(testX)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

n_iters = [5, 10, 20, 50, 100, 1000]
scores = []
for n_iter in n_iters:
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter)
    clf.fit(trainX, trainY)
    scores.append(clf.score(testX, testY))
  
plt.title("Effect of n_iter")
plt.xlabel("n_iter")
plt.ylabel("score")
plt.plot(n_iters, scores)

start = time.time()
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(trainX, trainY)
stop = time.time()
print(f"Training time for linear SVM with SGD training: {stop - start}s")

start = time.time()
#clf = SVC(kernel='linear')
clf.fit(trainX, trainY)
stop = time.time()
print(f"Training time for linear SVM without SGD training: {stop - start}s")

losses = ["hinge", "log", "modified_huber", "perceptron", "squared_hinge"]
scores = []
for loss in losses:
    clf = SGDClassifier(loss=loss, penalty="l2", max_iter=1000)
    clf.fit(trainX, trainY)
    scores.append(clf.score(testX, testY))
  
plt.title("Effect of loss")
plt.xlabel("loss")
plt.ylabel("score")
x = np.arange(len(losses))
plt.xticks(x, losses)
plt.plot(x, scores)

losses = ["hinge", "log", "modified_huber", "perceptron", "squared_hinge"]
scores = []
for loss in losses:
    clf = SGDClassifier(loss=loss, penalty="l2", max_iter=1000)
    clf.fit(trainX, trainY)
    scores.append(clf.score(testX, testY))
  
plt.title("Effect of loss")
plt.xlabel("loss")
plt.ylabel("score")
x = np.arange(len(losses))
plt.xticks(x, losses)
plt.plot(x, scores)

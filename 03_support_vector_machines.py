from lib.prep_terrain_data import make_terrain_data
from lib.email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import time

# create dataset
features_train, labels_train, features_test, labels_test = make_terrain_data()

# set classifier
clf = SVC(kernel='linear')

# fit classifier on training data
clf.fit(features_train, labels_train)

# predict test data
y_pred = clf.predict(features_test)

# find accuracy of test data vs predictions
print(accuracy_score(labels_test, y_pred))  # 0.92

# -------------------------------------------------------------------------------------------------------------------- #

# mimi project

# load data
features_train, features_test, labels_train, labels_test = preprocess()

# ------------------------------------------------- #
# SVM
# set classifier
# clf = SVC(kernel='linear')
clf = SVC(kernel='rbf')

# fit classifier on training data
t0 = time.time()
clf.fit(features_train, labels_train)

# predict on test set
y_pred = clf.predict(features_test)
t1 = time.time()

print("Train and Predict Time:", np.round(t1-t0, 2))
# Train and Predict Time: 204.09
# Train and Predict Time: 258.85

# find accuracy of test data vs predictions
print(accuracy_score(labels_test, y_pred))  # 0.98
print(accuracy_score(labels_test, y_pred))  # 0.99

# ------------------------------------------------- #
# Naive Bayes Comparison
# set classifier
clf = GaussianNB()

# fit classifier on training data
t0 = time.time()
clf.fit(features_train, labels_train)

# predict on test set
y_pred = clf.predict(features_test)
t1 = time.time()

print("Train and Predict Time:", np.round(t1-t0, 2))
# Train and Predict Time: 1.28

# find accuracy of test data vs predictions
print(accuracy_score(labels_test, y_pred))  # 0.97

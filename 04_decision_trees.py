from sklearn import tree
from sklearn.metrics import accuracy_score
from lib.prep_terrain_data import make_terrain_data
from lib.class_vis import pretty_picture, output_image
import math

# create dataset
features_train, labels_train, features_test, labels_test = make_terrain_data()


def classify(features_train, labels_train, min_sample_split=2):
    # your code goes here--should return a trained decision tree classifier

    clf = tree.DecisionTreeClassifier(min_samples_split=min_sample_split)
    clf.fit(features_train, labels_train)

    return clf


clf = classify(features_train, labels_train, min_sample_split=50)

# predict test data
y_pred = clf.predict(features_test)

print(accuracy_score(labels_test, y_pred))  # 0.912

pretty_picture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())

# entropy quiz
# entropy = sum_E(-Pi * log_2(Pi))

Pi_slow = 0.5
Pi_fast = 0.5

(-Pi_slow * math.log(Pi_slow, 2)) + (-Pi_fast * math.log(Pi_fast, 2))  # 1.0

Pi_slow = 0.67
Pi_fast = 0.33

print(-Pi_slow * math.log(Pi_slow, 2) + -Pi_fast * math.log(Pi_fast, 2))


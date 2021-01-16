from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from lib.prep_terrain_data import make_terrain_data


# create dataset
features_train, labels_train, features_test, labels_test = make_terrain_data()


# loop through 3 sklearn models
def choose_your_own_algo(x_train=features_train, y_train=labels_train, x_test=features_test, y_test=labels_test):

    clf_list = [KNeighborsClassifier(), RandomForestClassifier(), AdaBoostClassifier()]

    for clf in clf_list:
        model = clf.fit(x_train, y_train)
        preds = model.predict(x_test)
        print(type(clf).__name__, ":", accuracy_score(y_test, preds))


choose_your_own_algo(features_train, labels_train, features_test, labels_test)

# KNeighborsClassifier : 0.92
# RandomForestClassifier : 0.92
# AdaBoostClassifier : 0.924


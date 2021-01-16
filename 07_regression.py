from lib.age_net_worth import age_net_worth_data
from lib.feature_format import feature_format, target_feature_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle


# create dataset
ages_train, ages_test, net_worths_train, net_worths_test = age_net_worth_data()

# linear regression model
clf = linear_model.LinearRegression()
clf.fit(ages_train, net_worths_train)

# make predictions to get train and test scores
y_preds_train = clf.predict(ages_train)
y_preds_test = clf.predict(ages_test)

# view coefficients or slope and intercept
print("Coefficients:", clf.coef_)    # 6.309
print("Intercept:", clf.intercept_)  # -7.447

# view r-squared
print("R-Squared Train:", r2_score(net_worths_train, y_preds_train))  # 0.877
print("R-Squared Test:", r2_score(net_worths_test, y_preds_test))  # 0.789

# mini project
enron_data = pickle.load(open("../ud120-projects/final_project/final_project_dataset_modified.pkl", "rb"))

features_list = ["bonus", "salary"]
data = feature_format(enron_data, features_list, remove_any_zeroes=True)
target, features = target_feature_split(data)

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)

clf = linear_model.LinearRegression()
clf.fit(feature_train, target_train)

# make predictions to get train and test scores
y_preds_test = clf.predict(feature_test)
y_preds_train = clf.predict(feature_train)

# slope, intercept, r^2
print("Coefficients:", clf.coef_)
print("Intercept:", clf.intercept_)
print("R-Squared Test:", r2_score(target_test, y_preds_test))
print("R-Squared Train:", r2_score(target_train, y_preds_train))



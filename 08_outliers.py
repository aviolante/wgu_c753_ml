import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from lib.outlier_cleaner import outlier_cleaner

# load outlier data
ages = pickle.load(open("../ud120-projects/outliers/practice_outliers_ages.pkl", "rb"))
net_worths = pickle.load(open("../ud120-projects/outliers/practice_outliers_net_worths.pkl", "rb"))

# reshape data
ages = np.reshape(np.array(ages), (len(ages), 1))
net_worths = np.reshape(np.array(net_worths), (len(net_worths), 1))

# split data into train and test
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)


# instantiate and fit linear model
clf = LinearRegression()
clf.fit(ages_train, net_worths_train)

# view coefficient and intercept
print("Coefficient:", np.round(clf.coef_, 2))
print("Intercept:", np.round(clf.intercept_, 2))

# Coefficient: [[5.08]]
# Intercept: [25.21]

# make predictions on train and test
y_preds_train = clf.predict(ages_train)
y_preds_test = clf.predict(ages_test)


# score data (test is better than training likely due to no outliers in test set)
print("Training R-Squared:", np.round(r2_score(net_worths_train, y_preds_train), 2))
print("Test R-Squared:", np.round(r2_score(net_worths_test, y_preds_test), 2))

# Training R-Squared: 0.49
# Test R-Squared: 0.88

# find and remove outliers from training data
ages_train_outliers_rm, net_worths_train_outliers_rm = outlier_cleaner(y_preds_train, ages_train, net_worths_train, top_n_percent=10)


# fit linear model on new data
clf.fit(ages_train_outliers_rm, net_worths_train_outliers_rm)

# view coefficient and intercept
print("Coefficient:", np.round(clf.coef_, 2))
print("Intercept:", np.round(clf.intercept_, 2))

# Coefficient: [6.37]
# Intercept: -6.92

# make predictions on outlier removed train and test
y_preds_train_out_rm = clf.predict(ages_train_outliers_rm)
y_preds_test_out_rm = clf.predict(ages_test)

# score data
print("Training R-Squared:", np.round(r2_score(net_worths_train_outliers_rm, y_preds_train_out_rm), 2))
print("Test R-Squared:", np.round(r2_score(net_worths_test, y_preds_test_out_rm), 2))

# Training R-Squared: 0.95
# Test R-Squared: 0.98

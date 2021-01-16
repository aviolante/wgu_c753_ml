import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.cluster import KMeans
from lib.feature_format import feature_format, target_feature_split

# min max scaler
data = [115, 140, 175]


def feature_scaler(input_list):

    min_val = np.min(input_list)
    max_val = np.max(input_list)

    scaled_values = []

    for i in input_list:
        x_prime = (i - min_val)/(max_val - min_val)

        scaled_values.append(x_prime)

    return scaled_values


scaled_values = feature_scaler(data)
# [0.0, 0.4166666666666667, 1.0]

# sklearn implementation
# make numpy array and reshape since single feature
data = np.array(data).reshape(-1,1)

scaler = MinMaxScaler()
scaler.fit_transform(data)
# array([[0.        ],
#        [0.41666667],
#        [1.        ]])

# mini project
data = pickle.load(open("../ud120-projects/final_project/final_project_dataset.pkl", "rb"))

# remove given outlier
data.pop("TOTAL", 0)

feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi = "poi"

features_list = [poi, feature_1, feature_2]
data = feature_format(data, features_list)
poi, finance_features = target_feature_split(data)

# scale data
scaler = MinMaxScaler()
scaler.fit(finance_features)
finance_features_scaled = scaler.transform(finance_features)

# scale new input
new_input = np.array([[200000., 1000000.]])

print(scaler.transform(new_input))
# [[0.17997621 0.02911345]]

# cluster on scaled feature
clust = KMeans(n_clusters=2)
clust.fit(finance_features_scaled)

clust_preds = clust.labels_
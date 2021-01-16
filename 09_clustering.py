import pickle
from sklearn.cluster import KMeans
from lib.feature_format import feature_format, target_feature_split

data = pickle.load(open("../ud120-projects/final_project/final_project_dataset.pkl", "rb"))

# remove given outlier
data.pop("TOTAL", 0)

feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = feature_format(data, features_list)
poi, finance_features = target_feature_split(data)

clust = KMeans(n_clusters=2)
clust.fit(finance_features)

clust_preds = clust.labels_



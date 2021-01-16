import pickle
import pandas as pd
from lib.poi_emails import poi_emails


enron_data = pickle.load(open("../ud120-projects/final_project/final_project_dataset.pkl", "rb"))

enron_data.keys()
# dict_keys(['METTS MARK', 'BAXTER JOHN C', 'ELLIOTT STEVEN', 'CORDES WILLIAM R', 'HANNON KEVIN P', ...])

enron_data["SKILLING JEFFREY K"].keys()
# dict_keys(['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances',
#            'bonus', 'email_address', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
#            'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other',
#            'from_this_person_to_poi', 'poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock',
#            'director_fees'])

enron_data["SKILLING JEFFREY K"]["bonus"]
# 5600000

# convert dict to df
enron_df = pd.DataFrame.from_dict(enron_data).T

# how many people are in the dataset?
len(enron_data.keys())  # 146
enron_df.shape  # (146, 21)

# for each person, how many features are available?
len(enron_data["SKILLING JEFFREY K"].keys())  # 21
enron_df.shape  # (146, 21)

# how many POIs are there in the E+F dataset?
enron_df = pd.DataFrame.from_dict(enron_data).T

enron_df["poi"].value_counts()
# False    128
# True      18

# how many poi emails (names)?
poi_names = pd.read_csv("../ud120-projects/final_project/poi_names.txt", sep=" ", header=None, skiprows=1,
                        names=["poi_bin", "last_name", "first_name"])

poi_names["full_name"] = poi_names["last_name"] + " " + poi_names["first_name"]

len(poi_names["full_name"].unique())  # 35


# total stock value for james prentice
enron_data["PRENTICE JAMES"]["total_stock_value"]

# ... to be continued











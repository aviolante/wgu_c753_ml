import numpy as np

"""

function to remove top-n percent of outliers based on absolute error

"""


def outlier_cleaner(predictions, ages, net_worths, top_n_percent=10):

    abs_error = np.abs(net_worths - predictions)
    percent_thresh = np.percentile(abs_error, 100-top_n_percent)

    ages_train_outliers_rm = ages[np.where(abs_error < percent_thresh)]
    net_worths_train_outliers_rm = net_worths[np.where(abs_error < percent_thresh)]

    ages_train_outliers_rm = np.reshape(np.array(ages_train_outliers_rm), (len(ages_train_outliers_rm), 1))

    return ages_train_outliers_rm, net_worths_train_outliers_rm

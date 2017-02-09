#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sys.path.append("../tools/")
from feature_format import featureFormat


def is_nan(x):
    return str(float(x)).lower() == 'nan'


def find_highest_value(data_dict,feature):
    highest_value_name= ''
    highest_value=0
    for key in data_dict:
        value = data_dict[key][feature]
        if not is_nan(value):
            if value > highest_value:
                highest_value = value
                highest_value_name = key

    return highest_value_name


def plot_dict_data(data_dict,features):
    data = featureFormat(data_dict, features)
    df = pd.DataFrame(data, columns=features)
    sns.boxplot(x=df)
    plt.xticks(rotation=45)
    plt.show()


def outlierCleaner(data_dict, features):
    plot_dict_data(data_dict,features)
    key = find_highest_value(data_dict,'total_payments')
    print ' ---- Outliers ---- '
    if key in data_dict:
        del data_dict[key]
        print ' Deleted : ',key

    plot_dict_data(data_dict,features)
    return data_dict



# if __name__ == "__main__":
#     data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
#     features = ["salary", "bonus", 'total_stock_value','total_payments']
#     outlierCleaner(data_dict,features)



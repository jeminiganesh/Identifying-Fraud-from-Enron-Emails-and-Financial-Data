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


def plot_dict_data(data_dict,features,desc):
    data = featureFormat(data_dict, features)
    df = pd.DataFrame(data, columns=features)
    sns.boxplot(x=df,orient='h')
    plt.tight_layout()
    plt.savefig(desc)


def outlierCleaner(data_dict, features):
    plot_dict_data(data_dict, features, 'enron_outliers_fig1')
    print ' ---- Outliers ---- '
    key = find_highest_value(data_dict,'total_payments')
    if key in data_dict:
        del data_dict[key]
        print ' Deleted : ',key
    return data_dict





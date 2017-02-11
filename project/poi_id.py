import sys
import pickle
from tester import dump_classifier_and_data, test_classifier
from sklearn.svm import LinearSVC


from time import time

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import copy

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score
from enron_outliers import outlierCleaner

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn import linear_model

import pandas as pd

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

INPUT_FILE_NAME = 'final_project_dataset.pkl'
ALL_CLF_EVAL_LIST = []

def load_data(input_file):
    with open(input_file, "r") as data_file:
        data_dict = pickle.load(data_file)
    return data_dict


def explore_data(data_dict):
    print ' ----------------------- Data Exploration ------------------------------ '
    print ' Number of Data Points : ', len(data_dict)
    print ' Number of Features : ', len(data_dict[data_dict.keys()[0]])
    num_poi = 0
    for dic in data_dict.values():
        if dic['poi'] == 1: num_poi += 1
    print "Numboer of POI'S", num_poi

    print '\n ----- Data set Featuers ------ \n ', data_dict[data_dict.keys()[0]].keys()


def create_new_features(data_dict):
    my_dataset = data_dict
    new_features = []
    new_features.append('fraction_from_poi')
    new_features.append('fraction_to_poi')
    new_features.append('wealth')

    for item in my_dataset:
        person = my_dataset[item]
        if (all([person['from_poi_to_this_person'] != 'NaN',
                 person['from_this_person_to_poi'] != 'NaN',
                 person['to_messages'] != 'NaN',
                 person['from_messages'] != 'NaN'
                 ])):
            person["fraction_from_poi"] = float(person["from_poi_to_this_person"]) / float(person["to_messages"])
            person["fraction_to_poi"] = float(person["from_this_person_to_poi"]) / float(person["from_messages"])
        else:
            person["fraction_from_poi"] = 0
            person["fraction_to_poi"] = 0
        if (all([person['salary'] != 'NaN',
                 person['total_stock_value'] != 'NaN',
                 person['exercised_stock_options'] != 'NaN',
                 person['bonus'] != 'NaN'
                 ])):
            person['wealth'] = sum([person[field] for field in ['salary',
                                                                'total_stock_value',
                                                                'exercised_stock_options',
                                                                'bonus']])
        else:
            person['wealth'] = 'NaN'

    return my_dataset, new_features


def get_kbest_features(features, labels, features_names, k=10):
    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    results_list = zip(k_best.get_support(), features_names, k_best.scores_)
    results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
    best_features = []
    print ' \n -------- K Best Features ------- \n'
    for i, result in enumerate(results_list):
        if i >= k:
            break
        print result[1], result[2]
        best_features.append(result[1])
    return best_features



def get_classifiers():
    ### Stochastic Gradient Descent
    classifiers =[]
    sdg_clf = linear_model.SGDClassifier(class_weight="balanced")
    classifiers.append(sdg_clf)

    ### Gaussian Naive Bayes
    gaussianNB_clf = GaussianNB()
    classifiers.append(gaussianNB_clf)
    #
    # ### DecisionTree
    decisionTree_clf = tree.DecisionTreeClassifier()
    classifiers.append(decisionTree_clf)

    ### Random Forests
    randomForest_clf = RandomForestClassifier()
    classifiers.append(randomForest_clf)
    #
    # ### AdaBoost
    adaBoost_clf = AdaBoostClassifier()
    classifiers.append(adaBoost_clf)

    return classifiers


def get_clf_parameters(clf_name):
    return {
        'DecisionTreeClassifier': {'criterion': ['gini', 'entropy'],
                                   'min_samples_split': [2, 10],
                                   'max_depth': [None, 2, 5],
                                   'min_samples_leaf': [1, 5],
                                   'max_leaf_nodes': [None, 5, 10]}
        ,
        'AdaBoostClassifier': {'n_estimators': [19, 20],
                               'algorithm': ['SAMME'],
                               'learning_rate': [.5]}
    }.get(clf_name, None)


def grid_search_eval(grid_search, features, labels, iterations=100):
    precision, recall, accuracy = [], [], []
    for iteration in range(iterations):
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
                                                                                    random_state=iteration)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        precision = precision + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
        accuracy = accuracy + [accuracy_score(labels_test, predictions)]

    acc_score = round(np.mean(accuracy), 2)
    prec_score = round(np.mean(precision), 2)
    rec_score = round(np.mean(recall), 2)
    best_params = grid_search.best_params_

    return [acc_score, prec_score, rec_score, best_params]


def scalar_reqd(clf_name):
    if clf_name == 'SGDClassifier':
        return True
    return False


def classifier_eval(clf, features, labels, k=10, iterations=100):
    clf_name = clf.__class__.__name__
    estimators = []
    params = {}

    if scalar_reqd(clf_name):
        estimators.append(('minMaxScalar', MinMaxScaler(copy=False)))
    estimators.append(('kbest', SelectKBest(k=k)))
    estimators.append(('clf', clf))

    pipe = Pipeline(estimators)

    clf_parameters = get_clf_parameters(clf_name)
    if clf_parameters:
        for key, value in clf_parameters.items():
            new_key = '{}__{}'.format('clf', key)
            params[new_key] = value

    grid_search = GridSearchCV(pipe, param_grid=params, n_jobs=-1)

    t0 = time()
    clf_eval = grid_search_eval(grid_search, features, labels, iterations)
    clf_eval = [clf_name] + clf_eval + [k, '{} Secs'.format(round((time() - t0), 2))]
    ALL_CLF_EVAL_LIST.append(clf_eval)
    print clf_eval


def print_best_clf(clf_eval_list):
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_accuracy_clf = None
    best_precision_clf = None
    best_recall_clf = None

    for clf_eval in clf_eval_list:
        if clf_eval[1] > best_accuracy:
            best_accuracy = clf_eval[1]
            best_accuracy_clf = clf_eval
        if clf_eval[2] > best_precision:
            best_precision = clf_eval[2]
            best_precision_clf = clf_eval
        if clf_eval[3] > best_recall:
            best_recall = clf_eval[3]
            best_recall_clf = clf_eval

    print ' \n --- Best Accuracy Clf --- \n ', best_accuracy_clf
    print ' \n --- Best Precision Clf --- \n ', best_precision_clf
    print ' \n --- Best Recall Clf --- \n', best_recall_clf


def plot_data(eval_list):

    eval_dataframe = pd.DataFrame(eval_list,columns=['clfName','Accuracy','Precision','Recall','best_params','k','time'])

    import seaborn as sns
    import matplotlib.pyplot as plt

    grid = sns.FacetGrid(eval_dataframe, col="clfName", col_wrap=3)
    grid.map(plt.plot, "k", "Precision", marker="o", ms=5,color='green')
    grid.map(plt.plot, "k", "Recall", marker="s", ms=5, color='red')
    grid.set_ylabels('Score')
    grid.set_xlabels('K best Features')
    plt.savefig('ModelsPerformance.png')



if __name__ == "__main__":
    start_time = time()

    import warnings

    warnings.filterwarnings("ignore")

    ### Load the dictionary containing the dataset
    data_dict = load_data(INPUT_FILE_NAME)

    ### Explore the Data
    explore_data(data_dict)

    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    features_list = ['poi', 'salary', 'to_messages', 'deferral_payments',
                     'total_payments', 'exercised_stock_options', 'bonus',
                     'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred',
                     'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other',
                     'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive',
                     'from_poi_to_this_person']

    ## Check and remove Outliers
    data_dict =  outlierCleaner(data_dict,features_list[1:])


    my_dataset, new_features = create_new_features(data_dict)
    features_list = features_list + new_features

    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    classifiers = get_classifiers()

    for clf in classifiers:
        print '-----------------------'
        for k in range(1,len(features_list[1:]),1):
            classifier_eval(clf,features,labels,k=k,iterations=100)

    plot_data(ALL_CLF_EVAL_LIST)

    my_best_features_list = get_kbest_features(features,labels,features_list[1:],6)
    my_best_features_list = ['poi'] + my_best_features_list
    print '\n ----------- GaussianNB Cross Validation with best 6 Features --------------- \n '
    clf = GaussianNB()
    t0 = time()
    test_classifier(clf , my_dataset, my_best_features_list)
    print "----Execution Time---- : ", round(time() - t0, 3), "s"

    my_best_features_list = get_kbest_features(features, labels, features_list[1:], 5)
    my_best_features_list = ['poi'] + my_best_features_list
    print '\n ----------- GaussianNB Cross Validation with best 5 Features --------------- \n '
    clf = GaussianNB()
    t0 = time()
    test_classifier(clf, my_dataset, my_best_features_list)
    print "----Execution Time---- : ", round(time() - t0, 3), "s"

    dump_classifier_and_data(clf, my_dataset, my_best_features_list)

    print "---- Total Execution Time-------- : ", round(time() - start_time, 3), "s"












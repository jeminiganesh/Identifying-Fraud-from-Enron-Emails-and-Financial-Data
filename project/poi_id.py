import sys
import pickle
from tester import dump_classifier_and_data,test_classifier

from sklearn.feature_selection import SelectKBest
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score
from enron_outliers import outlierCleaner


from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import linear_model


sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

INPUT_FILE_NAME = 'final_project_dataset.pkl'


def load_data(input_file):
    with open(input_file, "r") as data_file:
        data_dict = pickle.load(data_file)
    return data_dict



def explore_data(data_dict):
    print ' Number of Data Points : ', len(data_dict)
    print ' Number of Features : ', len(data_dict[data_dict.keys()[0]])
    num_poi = 0
    for dic in data_dict.values():
        if dic['poi'] == 1: num_poi += 1
    print "Numboer of POI'S", num_poi

    print '\n ----- Featuers ------ \n ', data_dict[data_dict.keys()[0]].keys()


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


def get_kbest_features(features, labels, features_list, k=10):
    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    results_list = zip(k_best.get_support(), features_list[1:], k_best.scores_)
    results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
    best_features = []
    print ' \n -------- K Best Features ------- \n'
    for i, result in enumerate(results_list):
        if i > k:
            break
        print result[1],result[2]
        best_features.append(result[1])

    return best_features


def get_classifiers():


    ### Stochastic Gradient Descent
    sdg_clf = linear_model.SGDClassifier(class_weight="balanced")


    ### Gaussian Naive Bayes
    gaussianNB_clf = GaussianNB()

    decisionTree_clf = tree.DecisionTreeClassifier()

    ### Random Forests
    from sklearn.ensemble import RandomForestClassifier
    randomForest_clf = RandomForestClassifier()


    from sklearn.ensemble import AdaBoostClassifier
    adaBoost_clf = AdaBoostClassifier()

    classifiers = [sdg_clf, gaussianNB_clf, randomForest_clf, decisionTree_clf, adaBoost_clf]
    return classifiers


def get_clf_parameters(clf_name):
    return {
        'DecisionTreeClassifier': {'criterion': ['gini', 'entropy'],
                                   'min_samples_split': [2, 10, 20],
                                   'max_depth': [None, 2, 5, 10],
                                   'min_samples_leaf': [1, 5, 10],
                                   'max_leaf_nodes': [None, 5, 10, 20]}
        ,
        'AdaBoostClassifier': {'n_estimators': [19, 20, 21],
                               'algorithm': ['SAMME'],
                               'learning_rate': [.5]}
        ,
        'RandomForestClassifier': {
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 10, 20],
            'max_depth': [None, 2, 5, 10],
            'min_samples_leaf': [1, 5, 10],
            'max_leaf_nodes': [None, 5, 10, 20]
        }
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

    acc_score = round(np.mean(accuracy), 5)
    prec_score = round(np.mean(precision), 5)
    rec_score = round(np.mean(recall), 5)
    best_params = grid_search.best_params_

    return [acc_score, prec_score, rec_score,best_params]


def classifier_eval(classifiers,features, labels, applyPCA=True,iterations=10):
    clf_eval_list = []
    for clf in classifiers:
        clf_name = clf.__class__.__name__
        estimators = []
        if applyPCA:
            estimators.append(('pca', PCA(n_components=2)))
        estimators.append(('clf',clf))
        pipe = Pipeline(estimators)
        params = dict(pca__n_components=['mle'])
        clf_parameters = get_clf_parameters(clf_name)
        if clf_parameters:
            for key, value in clf_parameters.items():
                new_key = '{}__{}'.format('clf', key)
                params[new_key] = value
        grid_search = GridSearchCV(pipe, param_grid=params,n_jobs=-1)
        clf_eval = grid_search_eval(grid_search, features, labels, iterations)
        clf_eval = [clf_name] + clf_eval
        print clf_eval
        clf_eval_list.append(clf_eval)

    return clf_eval_list


def print_best_clf(clf_eval_list):
    best_accuracy =0
    best_precision =0
    best_recall = 0
    best_accuracy_clf= None
    best_precision_clf=None
    best_recall_clf = None

    for clf_eval in clf_eval_list:
        if clf_eval[1] > best_accuracy:
            best_accuracy = clf_eval[1]
            best_accuracy_clf = clf_eval
        if clf_eval[2] > best_precision:
            best_precision = clf_eval[2]
            best_precision_clf = clf_eval
        if clf_eval[3]>best_recall:
            best_recall = clf_eval[3]
            best_recall_clf= clf_eval

    print ' \n --- Best Accuracy Clf --- \n ',best_accuracy_clf
    print ' \n --- Best Precision Clf --- \n ',best_precision_clf
    print ' \n --- Best Recall Clf --- \n',best_recall_clf


if __name__ == "__main__":

    ### Load the dictionary containing the dataset
    data_dict = load_data(INPUT_FILE_NAME)

    ### Explore the Data
    explore_data(data_dict)

    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    features_list = ['poi', 'salary', 'bonus',
                     'total_stock_value', 'expenses', 'loan_advances',
                     'exercised_stock_options', 'shared_receipt_with_poi',
                     'long_term_incentive', 'other', 'restricted_stock',
                     'restricted_stock_deferred', 'deferral_payments', 'deferred_income',
                     'total_payments']

    ### Check and remove Outliers
    data_dict =  outlierCleaner(data_dict,features_list[1:])


    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    my_dataset, new_features = create_new_features(data_dict)

    # features_list = features_list + ['fraction_from_poi',
    #                                  'fraction_to_poi',
    #                                  'wealth']

    features_list = features_list + new_features


    ### Extract features and labels from dataset for local testing

    print ' ------  Initial Featues   ------- \n', features_list
    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    my_best_features_list = get_kbest_features(features, labels, features_list, k=10)
    my_best_features_list = ['poi'] + my_best_features_list

    classifiers = get_classifiers()

    print ' \n ------  Classifiers Evaluation  ------- \n'
    clf_eval_list = classifier_eval(classifiers,features,labels,applyPCA=True,iterations=15)

    print_best_clf(clf_eval_list)

    clf = GaussianNB()

    print '\n ------ Cross Validation --------------- \n '

    ## Cross Validation
    test_classifier(clf, my_dataset, my_best_features_list)

    dump_classifier_and_data(clf, my_dataset, my_best_features_list)

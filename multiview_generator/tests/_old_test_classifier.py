#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:14:46 2019

@author: bernardet
"""

from multiviews_datasets import generator_multiviews_dataset, results_to_csv
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def majority_list(predictions_list):
    """
    Returns an array which on each row the majority class of the same row in predictions_list
    
    Parameters:
    -----------
    predictions_list : list of 1D array
        
    Returns:
    --------
    an 1D array
    """
    n_samples = len(predictions_list[0])
    # majority_prediction[i] = prediction of predictions_list[i] which appears the most on predictions_list[i]
    majority_prediction = np.array([-1]*n_samples)
    # concatenate_predictions_list[i] = list contains prediction of the i-th data per view
    reshape_predictions_list = [predictions_list[i].reshape(len(predictions_list[i]), 1) for i in range(len(predictions_list))]
    concatenate_predictions_list = np.hstack(reshape_predictions_list)
    for sample in range(n_samples):
        # dictionary contains predictions (key) and its occurences in concatenate_predictions_list[sample]
        count = Counter(concatenate_predictions_list[sample])
        maj_value = max(count.values())  # maximal number of a prediction
        for key in count.keys():  # searchs the prediction with the maximal occurence number
            if count[key] == maj_value:
                majority_prediction[sample] = key
                break
        
    return majority_prediction


def majority_score(views_dictionary, integer_labels, cv=10, classifier="SVM", classifier_dictionary={'SVM':SVC(kernel='linear'), 'NB':GaussianNB()}):
    """
    Returns the mean and the standard deviation of accuracy score when predictions are selected by majority of predictions of different views
    
    Parameters:
    -----------
    views_dictionary : dict
    integer_labels = array
    cv : int
    classifier : str
    classifier_dictionary : dict    
        
    Returns:
    --------
    Two floats
    """   
    skf = StratifiedKFold(n_splits=cv, random_state=1, shuffle=True)  # provides cv train/test indices to split data in cv train/test sets.
    prediction_list = [[] for i in range(cv)]  # for majority_list function
    test_list =  [[] for i in range(cv)]  # for score
    
    for key in views_dictionary.keys():
        i = 0
        for train_index, test_index in skf.split(views_dictionary[key], integer_labels):
            # splits data and integer label of one view in test and train sets
            X = views_dictionary[key]
            train, test = X[train_index], X[test_index]         
            y_train, y_test =  integer_labels[train_index], integer_labels[test_index]
            # trains the classifier and tests it with test set
            clf = classifier_dictionary[classifier]
            clf.fit(train, y_train.ravel())
            y_pred = clf.predict(test)
            
            prediction_list[i].append(y_pred)
            if len(test_list[i]) == 0:  # same y_test for all views
                test_list[i] = y_test
            i += 1
            
    score = []
    for i in range(len(prediction_list)):
        y_pred_majority = majority_list(prediction_list[i])  # majority of views predictions
        score.append(accuracy_score(test_list[i].ravel(), y_pred_majority))  # score of majority of views predictions vs expected predictions
    score = np.array(score)
    return score.mean(), score.std()


def score_one_multiview_dataset(cv=10, classifier="SVM", classifier_dictionary={'SVM':SVC(kernel='linear'), 'NB':GaussianNB()}, n_samples=1000, n_views=3, n_classes=2, Z_factor=1, R=2/3, n_clusters_per_class=2, class_sep_factor=2, n_informative_divid=1, d=4, D=10, standard_deviation=2):
    """
    Returns 3 Series (first with dimensions of latent space, views and percentage of dimensions of latent space unsued in views, the second with accuracy score and the third with the standard deivation of accuracy score) of latent space, views, 
    early fusion predictions (concatenate views predictions) and late fusion predictions (majority views predictions)
    
    Parameters:
    -----------
    cv : int
    classifier : str
    classifier_dictionary : dict
    n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation : parameters of generator_multiviews_dataset
            
    Returns:
    --------
    3 Series
    """
    # dictionary contains percentage of unsued dimension of latent space and dimension of latent space and views
    dimensions = {'unsued dimension of latent space':0, "number of informative features":0, 'latent space':0}
    dimensions.update({'view'+str(i):0 for i in range(n_views)})
    # dictionary contains and mean of accuracy scores
    dict_scores_means = {'latent space':0}
    dict_scores_means.update({'view'+str(i):0 for i in range(n_views)})
    dict_scores_means.update({'early fusion':0, 'late fusion':0})
    # dictionary contains standard deviation of accuracy scores
    dict_scores_std = {'latent space':[]}
    dict_scores_std.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_std.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains data of each view
    dict_views = {'view'+str(i):0 for i in range(n_views)}
    
    Z, y, multiviews_list, unsued_dimensions_percent, n_informative = generator_multiviews_dataset(n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
    dimensions["unsued dimension of latent space"] = unsued_dimensions_percent
    dimensions["number of informative features"] = n_informative
    dimensions["latent space"] = Z.shape

    
    for i in range(n_views):
        # multiviews_list[i] = (columns / data of view i, numeros of columns of view i)
        dict_views['view'+str(i)] = multiviews_list[i][0]
        dimensions['view'+str(i)] = multiviews_list[i][0].shape
        
    early_fusion = np.concatenate([dict_views[key] for key in dict_views.keys()], axis=1)  # = concatenation of all views
    # dictionary of data
    dict_data_df = {'latent space':Z}
    dict_data_df.update({'view'+str(i):dict_views['view'+str(i)] for i in range(n_views)})
    dict_data_df.update({'early fusion':early_fusion})
            
    for key in dict_data_df.keys():
        clf = classifier_dictionary[classifier]
        score = cross_val_score(clf, dict_data_df[key], y, scoring='accuracy', cv=cv)
        dict_scores_means[key] = score.mean()
        dict_scores_std[key] = score.std()
    
    mean_majority, std_majority = majority_score(dict_views, y, cv, classifier, classifier_dictionary)
    dict_scores_means['late fusion'] = mean_majority
    dict_scores_std['late fusion'] = std_majority
    
    df_dimensions = pd.Series(dimensions)
    df_scores_means = pd.Series(dict_scores_means)
    df_scores_std = pd.Series(dict_scores_std)
            
    return df_dimensions, df_scores_means, df_scores_std
 

def score_multiviews_n_samples(n_samples_list, path_graph, cv=10, classifier="SVM", classifier_dictionary={'SVM':SVC(kernel='linear'), 'NB':GaussianNB()}, n_views=3, n_classes=2, Z_factor=1, R=2/3, n_clusters_per_class=2, class_sep_factor=2, n_informative_divid=1, d=4, D=10, standard_deviation=2):
    """
    Returns 2 DataFrames (first with accuracy score and the second with the standard deivation of accuracy score) of latent space, views, 
    early fusion predictions (concatenate views predictions) and late fusion predictions (majority views predictions) with n_samples_list as index for the indicated classifier
    Creates and saves (at the indicated path path_graph) a graph represented accuracy score (with confidence interval) vs n_samples_list
    
    Parameters:
    -----------
    n_samples_list : list
                     each element from n_samples_list defines a new dataset with element samples
    path_graph : str
                 path to save graphics
    cv : int
    classifier : str
    classifier_dictionary : dict
    n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation : parameters of generator_multiviews_dataset
        
    Returns:
    --------
    2 DataFrames with n_samples_list as index
    """
    # n_samples_list  = list of samples dimension from the lowest to the highest
    n_samples_list.sort(reverse=False)
    # list of percentage of unsued columns of latent space in views
    unsued_dimensions_percent_list = []
    # list of number of informative features of latent space
    n_informative_list = []
    # dictionary contains mean of accuracy scores per n_samples
    dict_scores_means = {'latent space':[]}
    dict_scores_means.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_means.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains standard deviation of accuracy scores per n_samples
    dict_scores_std = {'latent space':[]}
    dict_scores_std.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_std.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains data of each view
    dict_views = {'view'+str(i):0 for i in range(n_views)}
    
    for n_samples in n_samples_list:
        Z, y, multiviews_list, unsued_dimensions_percent, n_informative = generator_multiviews_dataset(n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
        unsued_dimensions_percent_list.append(unsued_dimensions_percent)
        n_informative_list.append(n_informative)
        

        for i in range(n_views):
            # multiviews_list[i] = (columns / data of view i, numeros of columns of view i)
            dict_views['view'+str(i)] = multiviews_list[i][0]
                    
        early_fusion = np.concatenate([dict_views[key] for key in dict_views.keys()], axis=1)  # = concatenation of all views
        # dictionary of data
        dict_data = {'latent space':Z}
        dict_data.update({'view'+str(i):dict_views['view'+str(i)] for i in range(n_views)})
        dict_data.update({'early fusion':early_fusion})
        
        for key in dict_data.keys():
            clf = classifier_dictionary[classifier]
            score = cross_val_score(clf, dict_data[key], y, scoring='accuracy', cv=cv)
            dict_scores_means[key].append(score.mean())
            dict_scores_std[key].append(score.std())
                
        mean_majority, std_majority = majority_score(dict_views, y, cv, classifier, classifier_dictionary)
        dict_scores_means['late fusion'].append(mean_majority)
        dict_scores_std['late fusion'].append(std_majority)
            
    df_scores_means = pd.DataFrame(dict_scores_means, index=n_samples_list)
    df_scores_std = pd.DataFrame(dict_scores_std, index=n_samples_list)

    plt.figure()
    for key in dict_scores_means.keys():
        plt.errorbar(n_samples_list, dict_scores_means[key], 1.96*np.array(dict_scores_std[key])/sqrt(cv), label=key)
    # index and label for graphic
    label_index = []
    for n_samples, percent, n_informative in zip(n_samples_list, unsued_dimensions_percent_list, n_informative_list):
        label_index.append(str(n_samples)+'\n'+str(percent)+'\n'+str(n_informative))

    plt.xticks(n_samples_list, label_index, fontsize='medium', multialignment='center')  # new x indexes
    plt.xlabel("Number of samples\nPercentage of dimensions of latent space unsued in views\nNumber of informative features")
    plt.ylabel("Accuracy score for "+classifier)
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.title("number of views = "+str(n_views)+" - R = "+str(round(R, 4))+"\nfactor of latent space dimension = "+str(Z_factor)+" - number of classes = "+str(n_classes)+"\nAccuracy score vs number of samples for classifier "+classifier)
    plt.savefig(path_graph+"score_samples_"+str(n_views)+"_"+classifier+".png", bbox_inches='tight')
    plt.show()
    plt.close()

    return df_scores_means, df_scores_std


def graph_comparaison_classifier_scores_n_samples(classifier1, classifier2, n_samples_list, path_graph, cv=10, classifier_dictionary={'SVM':SVC(kernel='linear'), 'NB':GaussianNB()}, n_views=3, n_classes=2, Z_factor=1, R=2/3, n_clusters_per_class=2, class_sep_factor=2, n_informative_divid=1, d=4, D=10, standard_deviation=2):
    """
    Creates and saves (at the indicated path path_graph) multiple graphs represented scores of classifier2 vs scores of classifier1 (one graph per column of result of score_multiviews_n_samples)
    
    Parameters:
    -----------
    classifier1 : str
    classifier2 : str
    n_samples_list : list
                     each element from n_samples_list defines a new dataset with element samples
    path_graph : str
                 path to save graphics
    cv : int
    classifier : str
    classifier_dictionary : dict
    n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation : parameters of generator_multiviews_dataset
        
    Returns:
    --------
    None
    """    
    df_scores_clf1_means, df_scores_clf1_std = score_multiviews_n_samples(n_samples_list, path_graph, cv, classifier1, classifier_dictionary, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
    df_scores_clf2_means, df_scores_clf2_std = score_multiviews_n_samples(n_samples_list, path_graph, cv, classifier2, classifier_dictionary, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
    
    n_samples_list = df_scores_clf1_means.index
    keys = df_scores_clf1_means.keys()

    for key in keys:
        plt.figure()
        plt.scatter(df_scores_clf1_means[key].values, df_scores_clf2_means[key].values, c=df_scores_clf1_means[key].values)
        plt.plot([0.0, 1.1], [0.0, 1.1], "--", c=".7")  # diagonal
        plt.xlabel("Accuracy score for "+classifier1)
        plt.ylabel("Accuracy score for "+classifier2)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("number of views = "+str(n_views)+" - R = "+str(round(R, 4))+" - number of classes = "+str(n_classes)+"\nAccuracy score of "+key+" for "+classifier2+" vs "+classifier1)
        plt.savefig(path_graph+classifier1+"_"+classifier2+"_"+str(n_views)+"_"+key+".png")
        plt.show()
        plt.close()
    
    
def score_multiviews_R(R_list, path_graph, cv=10, classifier="SVM", classifier_dictionary={'SVM':SVC(kernel='linear'), 'NB':GaussianNB()}, n_samples=1000, n_views=3, n_classes=2, Z_factor=1, n_clusters_per_class=2, class_sep_factor=2, n_informative_divid=1, d=4, D=10, standard_deviation=2):
    """
    Returns 2 DataFrames (first with accuracy score and the second with the standard deivation of accuracy score) of latent space, views, 
    early fusion predictions (concatenate views predictions) and late fusion predictions (majority views predictions) with R_list as index for the indicated classifier
    Creates and saves (at the indicated path path_graph) a graph represented accuracy score (with confidence interval) vs R_list
    
    Parameters:
    -----------
    R_list : list
             each element from R_list defines a new dataset with element as R  
    path_graph : str
                 path to save graphics
    cv : int
    classifier : str
    classifier_dictionary : dict
    n_samples, n_views, n_classes, Z_factor, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation : parameters of generator_multiviews_dataset
            
    Returns:
    --------
    2 DataFrames with R_list as index
    """
    # R_list  = list of diverse values of R from the lowest to the highest
    R_list.sort(reverse=False)
    # list of percentage of unsued columns of latent space in views
    unsued_dimensions_percent_list = []
    # list of number of informative features of latent space
    n_informative_list = []
    # dictionary contains mean of accuracy scores per R
    dict_scores_means = {'latent space':[]}
    dict_scores_means.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_means.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains standard deviation of accuracy scores per R
    dict_scores_std = {'latent space':[]}
    dict_scores_std.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_std.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains data of each view
    dict_views = {'view'+str(i):0 for i in range(n_views)}
    
    for R in R_list:
        Z, y, multiviews_list, unsued_dimensions_percent, n_informative = generator_multiviews_dataset(n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
        unsued_dimensions_percent_list.append(unsued_dimensions_percent)
        n_informative_list.append(n_informative)
        
        for i in range(n_views):
            # multiviews_list[i] = (columns / data of view i, numeros of columns of view i)
            dict_views['view'+str(i)] = multiviews_list[i][0]
            
        early_fusion = np.concatenate([dict_views[key] for key in dict_views.keys()], axis=1)  # = concatenation of all views
        # dictionary of data
        dict_data_df = {'latent space':Z}
        dict_data_df.update({'view'+str(i):dict_views['view'+str(i)] for i in range(n_views)})
        dict_data_df.update({'early fusion':early_fusion})
                
        for key in dict_data_df.keys():
            clf = classifier_dictionary[classifier]
            score = cross_val_score(clf, dict_data_df[key], y, scoring='accuracy', cv=cv)
            dict_scores_means[key].append(score.mean())
            dict_scores_std[key].append(score.std())
        
        mean_majority, std_majority = majority_score(dict_views, y, cv, classifier, classifier_dictionary)
        dict_scores_means['late fusion'].append(mean_majority)
        dict_scores_std['late fusion'].append(std_majority)
    
    df_scores_means = pd.DataFrame(dict_scores_means, index=R_list)
    df_scores_std = pd.DataFrame(dict_scores_std, index=R_list)
    
    plt.figure()
    for key in dict_scores_means.keys():
        plt.errorbar(R_list, dict_scores_means[key], 1.96*np.array(dict_scores_std[key])/sqrt(cv), label=key)
    # index and label for graphic
    label_index = []
    R_label = []
    for i in range(0, len(R_list), 4):
        R_label.append(R_list[i])
        label_index.append(str(round(R_list[i], 2))+'\n'+str(unsued_dimensions_percent_list[i])+'\n'+str(n_informative_list[i]))
    
    plt.xticks(R_label, label_index, fontsize='medium', multialignment='center')  # new x indexes
    plt.xlabel("R\nPercentage of dimensions of latent space unsued in views\nNumber of informative features")
    plt.ylabel("Accuracy score for "+classifier)
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.title("number of views = "+str(n_views)+" - number of samples = "+str(n_samples)+"\nfactor of latent space dimension = "+str(Z_factor)+" - number of classes = "+str(n_classes)+"\nAccuracy score vs R for classifier "+classifier)
    plt.savefig(path_graph+"score_R_"+str(n_views)+"_"+str(n_samples)+"_"+str(Z_factor)+"_"+classifier+".png", bbox_inches='tight')
    plt.show()
    plt.close()
        
    return df_scores_means, df_scores_std

def score_multiviews_Z_factor(Z_factor_list, path_graph, cv=10, classifier="SVM", classifier_dictionary={'SVM':SVC(kernel='linear'), 'NB':GaussianNB()}, n_samples=1000, n_views=3, n_classes=2, R=2/3, n_clusters_per_class=2, class_sep_factor=2, n_informative_divid=1, d=4, D=10, standard_deviation=2):
    """
    Returns 3 DataFrames (first with accuracy score, the second with the standard deivation of accuracy score and the third with the error rate) of latent space, views, 
    early fusion predictions (concatenate views predictions) and late fusion predictions (majority views predictions) with sum of views dimension divided by Z_factor_list as index for the indicated classifier
    Creates and saves (at the indicated path path_graph) a graph represented accuracy score vs sum of views dimension divided by Z_factor_list and a graph represented error rate (1 - accuracy score) vs sum of views dimension divided by Z_factor_list
    
    Parameters:
    -----------
    Z_factor_list : list
                    each element from Z_factor_list defines a new dataset with element as Z_factor 
    path_graph : str
                 path to save graphics
    cv : int
    classifier : str
    classifier_dictionary : dict
    n_samples, n_views, n_classes, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation : parameters of generator_multiviews_dataset
            
    Returns:
    --------
    3 DataFrames with Z_factor_list as index
    """
    # Z_factor_list  = list of diverse values of Z_factor from the highest to the lowest
    Z_factor_list.sort(reverse=True)
    # list of sum of views dimension for each Z_factor_list item
    d_v = []
    # list of Z dimension for each Z_factor_list item
    Z_dim_list = []
    # list of percentage of unsued columns of latent space in views
    unsued_dimensions_percent_list = []
    # list of number of informative features of latent space
    n_informative_list = []
    # dictionary contains mean of accuracy scores per Z_factor
    dict_scores_means = {'latent space':[]}
    dict_scores_means.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_means.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains error rate per Z_factor
    dict_scores_error = {'latent space':[]}
    dict_scores_error.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_error.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains standard deviation of accuracy scores per Z_factor
    dict_scores_std = {'latent space':[]}
    dict_scores_std.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_std.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains data of each view
    dict_views = {'view'+str(i):0 for i in range(n_views)}
        
    for Z_factor in Z_factor_list:
        Z, y, multiviews_list, unsued_dimensions_percent, n_informative = generator_multiviews_dataset(n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
        unsued_dimensions_percent_list.append(unsued_dimensions_percent)
        n_informative_list.append(n_informative)
        
        for i in range(n_views):
            # multiviews_list[i] = (columns / data of view i, numeros of columns of view i)
            dict_views['view'+str(i)] = multiviews_list[i][0]
            
        early_fusion = np.concatenate([dict_views[key] for key in dict_views.keys()], axis=1)  # = concatenation of all views        
        # dimension = number of columns
        d_v.append(early_fusion.shape[1])
        Z_dim_list.append(Z.shape[1])
        # dictionary of data
        dict_data_df = {'latent space':Z}
        dict_data_df.update({'view'+str(i):dict_views['view'+str(i)] for i in range(n_views)})
        dict_data_df.update({'early fusion':early_fusion})
                
        for key in dict_data_df.keys():
            clf = classifier_dictionary[classifier]
            score = cross_val_score(clf, dict_data_df[key], y, scoring='accuracy', cv=cv)
            dict_scores_means[key].append(score.mean())
            dict_scores_error[key].append(1 - score.mean())
            dict_scores_std[key].append(score.std())
        
        mean_majority, std_majority = majority_score(dict_views, y, cv, classifier, classifier_dictionary)
        dict_scores_means['late fusion'].append(mean_majority)
        dict_scores_error['late fusion'].append(1 - mean_majority)
        dict_scores_std['late fusion'].append(std_majority)
        
    d_v_divid_Z = np.divide(np.array(d_v), np.array(Z_dim_list))
    
    df_scores_means = pd.DataFrame(dict_scores_means, index=d_v_divid_Z)
    df_scores_error = pd.DataFrame(dict_scores_error, index=d_v_divid_Z)
    df_scores_std = pd.DataFrame(dict_scores_std, index=d_v_divid_Z)
    
    # index and label for graphics
    label_index = [chr(i) for i in range(ord('a'),ord('z')+1)]
    label_index = label_index[0:len(d_v)]
    label_value = ""
    for label, v_Z, dim_v, dim_Z, Z_factor, percent, n_informative in zip(label_index, d_v_divid_Z, d_v, Z_dim_list, Z_factor_list, unsued_dimensions_percent_list, n_informative_list):
        label_value = label_value + label+" : V/Z = "+str(round(v_Z, 4))+", V = "+str(dim_v)+", Z = "+str(dim_Z)+", Z_factor = "+str(Z_factor)+", % ="+str(percent)+", n_informative = "+str(n_informative)+'\n'

    x_label = "V/Z = sum of views dimension divided by latent space dimension with :\nV = sum of views dimension\nZ = latent space dimension multiplied by Z_factor\n% = percentage of dimensions of latent space unsued in views\nn_informative = number of informative features"
    
    plt.figure(figsize=(10, 10))  # accuracy score vs d_v_divid_Z
    for key in dict_scores_means.keys():
        plt.semilogx(d_v_divid_Z, dict_scores_means[key], '.-', label=key)
    plt.xticks(d_v_divid_Z, label_index, fontsize='medium', multialignment='center')  # new x indexes
    plt.text(plt.xlim()[1]+0.05, plt.ylim()[1]-(plt.ylim()[1]-plt.ylim()[0])/2, label_value)
    plt.xlabel(x_label)
    plt.ylabel("Accuracy score for "+classifier)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="center left", borderaxespad=0)
    plt.title("number of views = "+str(n_views)+" - number of samples = "+str(n_samples)+"\nR = "+str(round(R, 4))+" - number of classes = "+str(n_classes)+"\nAccuracy score vs ratio sum of views dimension / latent space dimension for classifier "+classifier)    
    plt.savefig(path_graph+"score_Z_factor_"+str(n_views)+"_"+str(n_samples)+"_"+classifier+".png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10, 10))  # error rate vs d_v_divid_Z
    for key in dict_scores_means.keys():
        plt.semilogx(d_v_divid_Z, dict_scores_error[key], '.-', label=key)
    plt.xticks(d_v_divid_Z, label_index, fontsize='medium', multialignment='center')  # new x indexes
    plt.text(plt.xlim()[1]+0.05, plt.ylim()[1]-(plt.ylim()[1]-plt.ylim()[0])/2, label_value)
    plt.xlabel(x_label)
    plt.ylabel("Error rate for "+classifier)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="center left", borderaxespad=0)
    plt.title("number of views = "+str(n_views)+" - number of samples = "+str(n_samples)+"\nR = "+str(round(R, 4))+" - number of classes = "+str(n_classes)+"\nError rate vs ratio sum of views dimension / latent space dimension for classifier "+classifier)    
    plt.savefig(path_graph+"error_Z_factor_"+str(n_views)+"_"+str(n_samples)+"_"+classifier+".png", bbox_inches='tight')
    plt.show()
    plt.close()
    """
    plt.figure(figsize=(10, 10))
    
    for key in dict_scores_means.keys():
        plt.errorbar(d_v_divid_Z, dict_scores_means[key], 1.96*np.array(dict_scores_std[key])/sqrt(cv), label=key)
    plt.xticks(d_v_divid_Z, label_index, fontsize='medium', multialignment='center')
    plt.text(plt.xlim()[1]+0.05, plt.ylim()[1]-(plt.ylim()[1]-plt.ylim()[0])/2, label_value)
    plt.xlabel(x_label)
    plt.ylabel("Accuracy score for "+classifier)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="center left", borderaxespad=0)
    plt.title("number of views = "+str(n_views)+" - R = "+str(round(R, 4))+"\nAccuracy score vs ratio sum of views dimension / latent space dimension for classifier "+classifier)
    plt.savefig(path_graph+"score_Z_factor_errorbar_"+str(n_views)+"_"+classifier+".png", bbox_inches='tight')
    plt.show()
    plt.close()
    """
    plt.figure(figsize=(10, 10))  # accuracy score of early fusion divided by accuracy score of each view vs d_v_divid_Z
    for view in dict_views.keys():
        plt.semilogx(d_v_divid_Z, dict_scores_means['early fusion']/df_scores_means[view], '.-', label='early fusion score divided by '+view+' score')
    plt.xticks(d_v_divid_Z, label_index, fontsize='medium', multialignment='center')  # new x indexes
    plt.text(plt.xlim()[1]+0.05, plt.ylim()[1]-(plt.ylim()[1]-plt.ylim()[0])/2, label_value)
    plt.xlabel(x_label)
    plt.ylabel("Ratio accuracy score for early fusion / accuracy score for each view for "+classifier)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="center left", borderaxespad=0)
    plt.title("number of views = "+str(n_views)+" - R = "+str(round(R, 4))+"\nRatio accuracy score for early fusion / accuracy score for each view \nvs ratio sum of views dimension / latent space dimension for classifier "+classifier)
    plt.savefig(path_graph+"score_Z_factor_majority_view_divid_"+str(n_views)+"_"+classifier+".png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(10, 10))  # accuracy score of late fusion divided by accuracy score of each view vs d_v_divid_Z
    for view in dict_views.keys():
        plt.semilogx(d_v_divid_Z, dict_scores_means['late fusion']/df_scores_means[view], '.-', label='late fusion score divided by '+view+' score')
    plt.xticks(d_v_divid_Z, label_index, fontsize='medium', multialignment='center')  # new x indexes
    plt.text(plt.xlim()[1]+0.05, plt.ylim()[1]-(plt.ylim()[1]-plt.ylim()[0])/2, label_value)
    plt.xlabel(x_label)
    plt.ylabel("Ratio accuracy score for late fusion / accuracy score for each view for "+classifier)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="center left", borderaxespad=0)
    plt.title("number of views = "+str(n_views)+" - R = "+str(round(R, 4))+"\nRation accuracy score for late fusion / accuracy score for each view \nvs ratio sum of views dimension / latent space dimension for classifier "+classifier)
    plt.savefig(path_graph+"score_Z_factor_all_view_divid_"+str(n_views)+"_"+classifier+".png", bbox_inches='tight')
    plt.show()
    plt.close()
        
    return df_scores_means, df_scores_std, df_scores_error


def score_multiviews_n_views_R(n_views_list, R_list, path_graph, cv=10, classifier="SVM", classifier_dictionary={'SVM':SVC(kernel='linear'), 'NB':GaussianNB()}, n_samples=1000, n_classes=2, Z_factor=1, n_clusters_per_class=2, class_sep_factor=2, n_informative_divid=1, d=4, D=10, standard_deviation=2):
    """
    Returns a dictionary with n_views_list as key containing a list of DataFrames (represented accuracy score divided by accuracy score for R=1 <> redundancy null) of views, 
    early fusion predictions (concatenate views predictions and late fusion predictions (majority views predictions) with R_list as index for the indicated classifier per key
    Creates and saves (at the indicated path path_graph) a graph per value of n_views_list represented accuracy score divided by accuracy score for R=1 vs R_list
    
    Parameters:
    -----------
    n_views_list : list
                   each element from n_views_list defines a new dataset with element as n_views 
    R_list : list
             each element from R_list defines a new dataset with element as R                   
    path_graph : str
                 path to save graphics
    cv : int
    classifier : str
    classifier_dictionary : dict
    n_samples, n_classes, Z_factor, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation : parameters of generator_multiviews_dataset
            
    Returns:
    --------
    a dictionary with n_views_list as key containing a list of DataFrames (represented accuracy score divided by accuracy score for R=1 <> redundancy null) with R_list as index per value of n_views_list
    """
    dict_n_views_R_ratio = {key:0 for key in n_views_list}
    # n_views_list  = list of diverse values of n_views from the lowest to the highest
    n_views_list.sort(reverse=False)
    # same views have same colors on each graphs
    dict_colors = {'view'+str(i):0 for i in range(n_views_list[-1])}
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for key, c in zip(dict_colors.keys(), colors):
        dict_colors[key] = c    
    dict_colors.update({'early fusion':'purple', 'late fusion':'maroon'})
    
    for n_views in n_views_list:    
        # R_list  = list of diverse values of R from the lowest to the highest
        R_list.sort(reverse=False)
        # list of percentage of unsued columns of latent space in views
        unsued_dimensions_percent_list = []
        # list of number of informative features of latent space
        n_informative_list = []
        # dictionary contains mean of accuracy scores per R
        dict_scores_means = {'view'+str(i):[] for i in range(n_views)}
        dict_scores_means.update({'early fusion':[], 'late fusion':[]})
        # dictionary of list of scores' mean of view for diverse R divided by score's mean of view for R = 1 (<> redundancy null)
        dict_scores_ratio_R_1 = {'view'+str(i):0 for i in range(n_views)}
        dict_scores_ratio_R_1.update({'early fusion':0, 'late fusion':0})
        # dictionary contains data of each view
        dict_views = {'view'+str(i):0 for i in range(n_views)}
        
        for R in R_list:
            Z, y, multiviews_list, unsued_dimensions_percent, n_informative = generator_multiviews_dataset(n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
            unsued_dimensions_percent_list.append(unsued_dimensions_percent)
            n_informative_list.append(n_informative)
            
            for i in range(n_views):
                # multiviews_list[i] = (columns / data of view i, numeros of columns of view i)
                dict_views['view'+str(i)] = multiviews_list[i][0]
                
            early_fusion = np.concatenate([dict_views[key] for key in dict_views.keys()], axis=1)  # = concatenation of all views
            # dictionary of data
            dict_data_df = {'view'+str(i):dict_views['view'+str(i)] for i in range(n_views)}
            dict_data_df.update({'early fusion':early_fusion})
                    
            for key in dict_data_df.keys():
                clf = classifier_dictionary[classifier]
                score = cross_val_score(clf, dict_data_df[key], y, scoring='accuracy', cv=cv)
                dict_scores_means[key].append(score.mean())
            
            mean_majority, std_majority = majority_score(dict_views, y, cv, classifier, classifier_dictionary)
            dict_scores_means['late fusion'].append(mean_majority)
        
        for key in dict_scores_means.keys():
            score_R_1 = dict_scores_means[key][-1]  # R = 1 = last value of R_list => last score value in dict_scores_means[key]
            dict_scores_ratio_R_1[key] = np.divide(np.array(dict_scores_means[key]), score_R_1)
                
        df_scores_ratio_R_1 = pd.DataFrame(dict_scores_ratio_R_1, index=R_list)

        plt.figure()
        for key in dict_scores_means.keys():
            plt.plot(R_list, dict_scores_ratio_R_1[key], '.-',  color=dict_colors[key], label=key)
        # index and label for graphic
        label_index = []
        R_label = []
        for i in range(0, len(R_list), 4):
            R_label.append(R_list[i])
            label_index.append(str(round(R_list[i], 2))+'\n'+str(unsued_dimensions_percent_list[i])+'\n'+str(n_informative_list[i]))
        
        plt.xticks(R_label, label_index, fontsize='medium', multialignment='center')  # new x indexes
        plt.xlabel("R\nPercentage of dimensions of latent space unsued in views\nNumber of informative features")
        plt.ylabel("Ratio accuracy score / accuracy score for R = 1 for "+classifier)
        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        plt.title("number of views = "+str(n_views)+" - number of samples = "+str(n_samples)+"\nfactor of latent space dimension = "+str(Z_factor)+" - number of classes = "+str(n_classes)+"\nRatio accuracy score / accuracy score for R = 1\n(redundancy null) vs R for classifier "+classifier)
        plt.savefig(path_graph+"score_R_divid_R_1_"+str(n_views)+"_"+str(n_samples)+"_"+str(Z_factor)+"_"+classifier+".png", bbox_inches='tight')
        plt.show()
        plt.close()
            
        dict_n_views_R_ratio[n_views] = df_scores_ratio_R_1
        
    plt.figure()
    ax = plt.axes(projection="3d")
    
    for n_views in n_views_list:
        for key in dict_n_views_R_ratio[n_views].keys():
            if n_views == n_views_list[-1]:  # print legends only once
                ax.plot(R_list, dict_n_views_R_ratio[n_views][key], n_views, color=dict_colors[key], label=key)
            else:
                ax.plot(R_list, dict_n_views_R_ratio[n_views][key], n_views, color=dict_colors[key])
    
    ax.set_xlabel("R")
    ax.set_ylabel("Ratio accuracy score / accuracy score for R = 1 for "+classifier)
    ax.set_zlabel("Number of views")
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.title("number of samples = "+str(n_samples)+" - factor of latent space dimension = "+str(Z_factor)+" - number of classes = "+str(n_classes)+"\nRatio accuracy score / accuracy score for R = 1 (redundancy null) vs R, number of views for classifier "+classifier)
    plt.savefig(path_graph+"score_R_divid_R_1_all_n_views"+"_"+str(n_samples)+"_"+str(Z_factor)+"_"+classifier+".png", bbox_inches='tight')
    plt.show()
    plt.close()

    return dict_n_views_R_ratio


def score_multiviews_class_sep(class_sep_factor_list, path_graph, cv=10, classifier="SVM", classifier_dictionary={'SVM':SVC(kernel='linear'), 'NB':GaussianNB()}, n_views=3, n_samples=1000, n_classes=2, Z_factor=1, R=2/3, n_clusters_per_class=2, n_informative_divid=1, d=4, D=10, standard_deviation=2):
    """
    Returns 2 DataFrames (first with accuracy score and the second with the standard deivation of accuracy score) of latent space, views, 
    early fusion predictions (concatenate views predictions) and late fusion predictions (majority views predictions) with class_sep_factor_list as index for the indicated classifier
    Creates and saves (at the indicated path path_graph) a graph represented accuracy score (with confidence interval) vs class_sep_factor_list
    
    Parameters:
    -----------
    class_sep_factor_list : list
                            each element from n_samples_list defines a new dataset
    path_graph : str
    cv : int
    classifier : str
    classifier_dictionary : dict
    n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, n_informative_divid, d, D, standard_deviation : parameters of generator_multiviews_dataset
        
    Returns:
    --------
    2 DataFrames with n_samples_list as index
    """
    # list of percentage of unsued columns of latent space in views
    unsued_dimensions_percent_list = []
    # list of number of informative features of latent space
    n_informative_list = []
    # dictionary contains mean of accuracy scores per class_sep_factor
    dict_scores_means = {'latent space':[]}
    dict_scores_means.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_means.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains standard deviation of accuracy scores per class_sep_factor
    dict_scores_std = {'latent space':[]}
    dict_scores_std.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_std.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains data of each view
    dict_views = {'view'+str(i):0 for i in range(n_views)}
    
    for class_sep_factor in class_sep_factor_list:
        Z, y, multiviews_list, unsued_dimensions_percent, n_informative = generator_multiviews_dataset(n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
        unsued_dimensions_percent_list.append(unsued_dimensions_percent)
        n_informative_list.append(n_informative)

        for i in range(n_views):
            # multiviews_list[i] = (columns / data of view i, numeros of columns of view i)
            dict_views['view'+str(i)] = multiviews_list[i][0]
        
        early_fusion = np.concatenate([dict_views[key] for key in dict_views.keys()], axis=1)  # = concatenation of all views
        # dictionary of data
        dict_data = {'latent space':Z}
        dict_data.update({'view'+str(i):dict_views['view'+str(i)] for i in range(n_views)})
        dict_data.update({'early fusion':early_fusion})
                
        for key in dict_data.keys():
            print('key', key)
            clf = classifier_dictionary[classifier]
            score = cross_val_score(clf, dict_data[key], y, scoring='accuracy', cv=cv)
            dict_scores_means[key].append(score.mean())
            dict_scores_std[key].append(score.std())
                
        mean_majority, std_majority = majority_score(dict_views, y, cv, classifier, classifier_dictionary)
        dict_scores_means['late fusion'].append(mean_majority)
        dict_scores_std['late fusion'].append(std_majority)
        
        print(dict_scores_means)
                    
    df_scores_means = pd.DataFrame(dict_scores_means, index=class_sep_factor_list)
    df_scores_std = pd.DataFrame(dict_scores_std, index=class_sep_factor_list)
    
    plt.figure()
    for key in dict_scores_means.keys():
        plt.errorbar(class_sep_factor_list, dict_scores_means[key], 1.96*np.array(dict_scores_std[key])/sqrt(cv), label=key)
    # index and label for graphic
    label_index = []
    for n_samples, percent, n_informative in zip(class_sep_factor_list, unsued_dimensions_percent_list, n_informative_list):
        label_index.append(str(n_samples)+'\n'+str(percent)+'\n'+str(n_informative))

    plt.xticks(class_sep_factor_list, label_index, fontsize='medium', multialignment='center')  # new x indexes
    plt.xlabel("Factor (class_sep = factor*n_clusters_per_class)\nPercentage of dimensions of latent space unsued in views\nNumber of informative features")
    plt.ylabel("Accuracy score for "+classifier)
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.title("number of views = "+str(n_views)+" - R = "+str(round(R, 4))+"\nfactor of latent space dimension = "+str(Z_factor)+" - number of classes = "+str(n_classes)+"\nAccuracy score vs factor of class_sep for classifier "+classifier)
    plt.savefig(path_graph+"score_class_sep_"+str(n_views)+"_"+classifier+".png", bbox_inches='tight')
    plt.show()
    plt.close()
        
    return df_scores_means, df_scores_std


def score_multiviews_n_informative_divided(n_informative_divid_list, path_graph, cv=10, classifier="SVM", classifier_dictionary={'SVM':SVC(kernel='linear'), 'NB':GaussianNB()}, n_views=3, n_samples=1000, n_classes=2, Z_factor=1, R=2/3, n_clusters_per_class=2, class_sep_factor=2, d=4, D=10, standard_deviation=2):
    """
    Returns 2 DataFrames (first with accuracy score and the second with the standard deivation of accuracy score) of latent space, views, 
    early fusion predictions (concatenate views predictions) and late fusion predictions (majority views predictions) with n_informative_divid_list as index for the indicated classifier
    Creates and saves (at the indicated path path_graph) a graph represented accuracy score (with confidence interval) vs n_informative_divid_list
    
    Parameters:
    -----------
    n_informative_divid_list : list
                                 each element from n_informative_divid_list defines a new dataset with element as n_informative_divid
    path_graph : str
    cv : int
    classifier : str
    classifier_dictionary : dict
    n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, d, D, standard_deviation : parameters of generator_multiviews_dataset
        
    Returns:
    --------
    2 DataFrames with n_samples_list as index
    """
    # list of percentage of unsued columns of latent space in views
    unsued_dimensions_percent_list = []
    # list of number of informative features of latent space
    n_informative_list = []
    # dictionary contains mean of accuracy scores per n_informative_divid
    dict_scores_means = {'latent space':[]}
    dict_scores_means.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_means.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains standard deviation of accuracy scores per n_informative_divid
    dict_scores_std = {'latent space':[]}
    dict_scores_std.update({'view'+str(i):[] for i in range(n_views)})
    dict_scores_std.update({'early fusion':[], 'late fusion':[]})
    # dictionary contains data of each view
    dict_views = {'view'+str(i):0 for i in range(n_views)}
    
    for n_informative_divid in n_informative_divid_list:
        Z, y, multiviews_list, unsued_dimensions_percent, n_informative = generator_multiviews_dataset(n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
        unsued_dimensions_percent_list.append(unsued_dimensions_percent)
        n_informative_list.append(n_informative)

        for i in range(n_views):
            # multiviews_list[i] = (columns / data of view i, numeros of columns of view i)
            dict_views['view'+str(i)] = multiviews_list[i][0]
        
        early_fusion = np.concatenate([dict_views[key] for key in dict_views.keys()], axis=1)  # = concatenation of all views
        # dictionary of data
        dict_data = {'latent space':Z}
        dict_data.update({'view'+str(i):dict_views['view'+str(i)] for i in range(n_views)})
        dict_data.update({'early fusion':early_fusion})
                
        for key in dict_data.keys():
            clf = classifier_dictionary[classifier]
            score = cross_val_score(clf, dict_data[key], y, scoring='accuracy', cv=cv)
            dict_scores_means[key].append(score.mean())
            dict_scores_std[key].append(score.std())
                
        mean_majority, std_majority = majority_score(dict_views, y, cv, classifier, classifier_dictionary)
        dict_scores_means['late fusion'].append(mean_majority)
        dict_scores_std['late fusion'].append(std_majority)

    df_scores_means = pd.DataFrame(dict_scores_means, index=n_informative_divid_list)
    df_scores_std = pd.DataFrame(dict_scores_std, index=n_informative_divid_list)
    
    plt.figure()
    for key in dict_scores_means.keys():
        plt.errorbar(n_informative_divid_list, dict_scores_means[key], 1.96*np.array(dict_scores_std[key])/sqrt(cv), label=key)
    # index and label for graphic
    label_index = []
    for n_informative_divid, percent, n_informative in zip(n_informative_divid_list, unsued_dimensions_percent_list, n_informative_list):
        label_index.append(str(n_informative_divid)+'\n'+str(percent)+'\n'+str(n_informative))

    plt.xticks(n_informative_divid_list, label_index, fontsize='medium', multialignment='center')  # new x indexes
    plt.xlabel("Factor (n_informative = dimension of latent space / factor)\nPercentage of dimensions of latent space unsued in views\nNumber of informative features")
    plt.ylabel("Accuracy score for "+classifier)
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.title("number of views = "+str(n_views)+" - R = "+str(round(R, 4))+"\nfactor of latent space dimension = "+str(Z_factor)+" - number of classes = "+str(n_classes)+"\nAccuracy score vs n_informative_divid for classifier "+classifier)
    plt.savefig(path_graph+"score_n_informative_"+str(n_views)+"_"+classifier+".png", bbox_inches='tight')
    plt.show()
    plt.close()
        
    return df_scores_means, df_scores_std

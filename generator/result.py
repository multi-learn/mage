#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:14:14 2019

@author: bernardet
"""
import parameters
from multiviews_datasets import generator_multiviews_dataset, results_to_csv
from tests.test_classifier import score_multiviews_n_samples, graph_comparaison_classifier_scores_n_samples, score_multiviews_R, score_multiviews_Z_factor, score_multiviews_n_views_R, score_multiviews_class_sep, score_one_multiview_dataset, score_multiviews_n_informative_divided

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

n_samples = parameters.n_samples
n_views = parameters.n_views
n_classes = 3#parameters.n_classes
Z_factor = parameters.Z_factor
R = parameters.R
n_clusters_per_class = 1#parameters.n_clusters_per_class
class_sep_factor = 2#5#2#parameters.class_sep_factor
n_informative_divid = 2#parameters.n_informative_divid
cv = parameters.cv
classifier = parameters.classifier
classifier_dictionary = parameters.classifier_dictionary
d = parameters.d
D = parameters.D
standard_deviation = parameters.standard_deviation
path_data = parameters.path_data
path_graph = parameters.path_graph
n_samples_list = parameters.n_samples_list
R_list = parameters.R_list
Z_factor_list = parameters.Z_factor_list
n_views_list = parameters.n_views_list
class_sep_factor_list = parameters.class_sep_factor_list
n_informative_divid_list = parameters.n_informative_divid_list


# Generate one dataset
#Z, y, multiviews_list, unsued_columns_percent = generator_multiviews_dataset(n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
#print(Z, y, multiviews_list)

# Register one multiview dataset
#results_to_csv(path, Z, y, multiviews_list)

# Score of one multiview dataset
#df_dimensions, df_scores_means, df_scores_std = score_one_multiview_dataset(cv, classifier, classifier_dictionary, n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
#print(df_dimensions, df_scores_means, df_scores_std)

# Scores of n_samples_list datasets
#mean_samples, std_samples = score_multiviews_n_samples(n_samples_list, path_graph, cv, classifier, classifier_dictionary, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
#print(mean_samples, std_samples)

# Plot scores classifier2 vs score classifier1
classifier1 = "SVM"
classifier2 = "NB"
#graph_comparaison_classifier_scores_n_samples(classifier1, classifier2, n_samples_list, path_graph, cv, classifier_dictionary, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)

# Scores of R_list datasets
#mean_R, std_R = score_multiviews_R(R_list, path_graph, cv, classifier, classifier_dictionary, n_samples, n_views, n_classes, Z_factor, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
#print(mean_R, std_R)

# Scores of Z_factor_list datasets
#mean_Z, std_Z, error_Z = score_multiviews_Z_factor(Z_factor_list, path_graph, cv, classifier, classifier_dictionary, n_samples, n_views, n_classes, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
#print(mean_Z, std_Z, error_Z)

# Scores divided by scores for R=1 (redundancy null) of n_views_list and R_list datasets
#dict_n_views_R_ratio = score_multiviews_n_views_R(n_views_list, R_list, path_graph, cv, classifier, classifier_dictionary, n_samples, n_classes, Z_factor, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)
#print(dict_n_views_R_ratio)

# Scores of class_sep_factor_list datasets 
#df_mean, df_std = score_multiviews_class_sep(class_sep_factor_list, path_data, path_graph, cv, classifier, classifier_dictionary, n_views, n_samples, n_classes, Z_factor, R, n_clusters_per_class, n_informative_divid, d, D, standard_deviation)
#print(df_mean, df_std)

# Scores of n_informative_divid_list datasets 
#mean_n_info, std_n_info = score_multiviews_n_informative_divided(n_informative_divid_list, path_graph, cv, classifier, classifier_dictionary, n_views, n_samples, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, d, D, standard_deviation)
#print(mean_n_info, std_n_info)


Z_factor_list = [1, 3, 10, 25, 100, 250, 1000]
path_graph = "/home/bernardet/Documents/StageL3/Graph/n_views_3_10_1_clus_2_n_info_div/"
n_classes = 2
n_clusters_per_class = 1
class_sep_factor = 2
n_informative_divid = 2
for n_views in range(3, 11):
    n_samples = 500*n_views
    mean_Z, std_Z, error_Z = score_multiviews_Z_factor(Z_factor_list, path_graph, cv, classifier, classifier_dictionary, n_samples, n_views, n_classes, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation)

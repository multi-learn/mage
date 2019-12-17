#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:53:05 2019

@author: bernardet
"""
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import numpy as np

# General parameters
n_samples = 1000
# number of samples (int)
n_views = 3
# number of views >= 2 (int)
n_classes = 2
# number of classes >= 3 (int)
Z_factor = 250
# multiplication factor of Z dimension (default value = 1)
R = 2/3
# redondance (float)
cv = 10
# number of cross-validation splitting (int)
n_clusters_per_class = 2
# number of clusters per class >= 1 (int)
class_sep_factor = 2
# factor >= 1 as class_sep = n_clusters_per_class*class_sep_factor
n_informative_divid = 1
# factor >= 1 as number of informative features = round(dimension of latent space / n_informative_divid)
classifier = "SVM"
# name of classifier (str)
classifier_dictionary={'SVM':SVC(kernel='linear'), 'NB':GaussianNB()}
# dictionary of classifiers
n_samples_list = [100, 500, 1000, 1500, 2000]#, 2500, 3000]#, 3500, 4000, 5000, 7000, 10000]
# list of number of samples to test generator
R_list = list(np.arange(0, 1.05, 0.05))
# list of diverse R
Z_factor_list = [1, 3, 10, 25, 100, 250, 1000]#[25, 50, 75, 100, 150, 200, 250, 500, 600, 750, 800, 900, 1000]
# list of diverse Z_factor
n_views_list = [n_view for n_view in range(2, 10)]
# list of diverse n_views
class_sep_factor_list = [2, 5, 10]
# list of diverse class_sep_factor
n_informative_divid_list = [1, 2, 3]
# list of diverse n_informative_divid
path_data = "/home/bernardet/Documents/StageL3/Data/"
# path to register the multiview dataset
path_graph = "/home/bernardet/Documents/StageL3/Graph/"
# path to register scores graph



# Parameters of gaussian distribution N((d+D)/2,  standard_deviation_2) :
# d <= dim[v] <= D for all v
# (d+D)/2 - 3*sqrt(standard_deviation_2) >= 0
d = 4
# < D, > 0
D = 10
# > d
standard_deviation = 2
# standard deviation of the gaussian distribution


# make_classification parameters :

# a trouver comment les utiliser
part_informative = 0
# proportion of informative features (float between 0 and 1)
part_redundant = 1
# proportion of redundant features (float between 0 and 1)
# n_redundant >= 1 for redundant
part_repeated = 1
# # proportion of repeated features (float between 0 and 1)
# n_repeated >= 1 for useless features and correlation


weights = [0.7, 0.3]
# proportion of samples assigned to each class (list) len(weights) = nbr_features
# != [0.5, 0.5] / = [0.8, 0.2] for imbalance
flip_y = 0.1
# fraction of samples whose class are randomly exchanged (float) 
# > 0 for noise

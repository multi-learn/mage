#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:38:38 2019

@author: bernardet
"""

from sklearn.datasets import make_classification
from random import gauss
from math import ceil, floor
import numpy as np
import pandas as pd


def latent_space_dimension(views_dimensions_list, R):
    """
    Returns the minimal dimension of latent space (enough to build the dataset) for generator_multiviews_dataset compared to views_dimensions_list
    
    Parameters:
    -----------
    views_dimensions_list : list
    R : float
        
    Returns:
    --------
    an int
    """
    max_view_dimension = max(views_dimensions_list)
    dimension = ceil(R*sum(views_dimensions_list))
    
    if dimension < max_view_dimension:
        dimension = max_view_dimension
            
    reduced_dimension = dimension
    remove_sum = 0
    
    for num_view in range(1, len(views_dimensions_list)):
        view_prec = views_dimensions_list[num_view - 1]
        view_current = views_dimensions_list[num_view]
        remove = floor(R*view_prec)
        remove_sum += remove
        if reduced_dimension - remove < view_current:
            dimension += view_current - (reduced_dimension - remove)
        reduced_dimension = dimension - remove_sum
            
    return dimension
  

def projection(latent_space, chosen_columns_list):
    """
    Returns the projection of latent_space on the columns of chosen_columns_list (in chosen_columns_list order)
    
    Parameters:
    -----------
    latent_space : array
    chosen_columns_list : list
        
    Returns:
    --------
    an array of dimension (number of rows of latent_space, length of chosen_columns_list)
    """
    return latent_space[:, chosen_columns_list]


def generator_multiviews_dataset(n_samples, n_views, n_classes, Z_factor, R, n_clusters_per_class, class_sep_factor, n_informative_divid, d, D, standard_deviation):
    """
    Returns a generator multiviews dataset
    
    Parameters:
    -----------
    n_samples : int
                dataset number of samples (number of rows of dataset)
    n_views : int >= 2
              dataset number of views
              one view is a set of some features (columns) of the latent space
    n_classes : int >= 2
                dataset number of classes 
    Z_factor : float >= 1
               minimal dimension of the latent space (enough to build the dataset) is calculed then multiplied by Z_factor 
    R : 0 <= float <= 1
        R = 1 <> no possibility of redundancy between views
        R = 0 <> maximal possibility of redundancy between views
    n_clusters_per_class : int
    class_sep_factor : float
                       class_sep = n_clusters_per_class*class_sep_factor
    n_informative_divid : float >= 1
                          n_informative_divid raises <> number of non-informative features raises
                          n_informative_divid = 1 <> no non-informative features, number of informative features = dimension of latent space
                          number of informative features = round(dimension of latent space / n_informative_divid)
    d : float >= 1
        minimal dimension of views
        dimension of views (int) chosen randomly from N((d+D)/2, standard_deviation^2) with d <= dimension of views <= D
    D : float >= d
        maximal dimension of views
        dimension of views (int) chosen randomly from N((d+D)/2, standard_deviation^2) with d <= dimension of views <= D
    standard_deviation : float
                         standard deviation of the gaussian distribution N((d+D)/2, standard_deviation^2)
                         dimension of views (int) chosen randomly from N((d+D)/2, standard_deviation^2) with d <= dimension of views <= D
        
    Returns:
    --------
    Z : an array of dimension(n_samples, R*n_views) = the generated samples
    y : an array of dimension (n_samples) = the integer labels for class membership of each sample
    a list of n_views tuples (X_v, I_v) with :
        X_v = Z projected along d_v (= dimension of the v-ith views) columns in I_v
        I_v = X_v columns numeros with numberring of Z columns numeros
    unsued_dimensions_percent : percentage of unsued columns of latent space in views
    n_informative : number of informative features (dimension of latent space - n_informative = number of non informative features)
    """
    
    if n_views < 2:
        raise ValueError("n_views >= 2")
    if n_classes < 2:
        raise ValueError("n_classes >= 2")
    if Z_factor < 1:
        raise ValueError("Z_factor >= 1 pour le bon fonctionnement de l'algorithme")
    if d < 1:
        raise ValueError("d >= 1")
    if (d+D)/2 - 3*standard_deviation < 0:
        raise ValueError("Il faut que (d+D)/2 - 3*standard_deviation >= 0 pour avoir des valeurs positives lors de l'emploi de la loi normale")
    
    # n_views dimension of view v values randomly from N((d+D)/2, standard_deviation^2)
    d_v = np.random.normal(loc=(d+D)/2, scale=standard_deviation, size=n_views)
    d_v = list(d_v)
    remove_list, add_list = [], []
    for dim_view in d_v:
        if dim_view < d or dim_view > D:  # 1 <= d <= dim_view <= D
            remove_list.append(dim_view)
            add = -1
            while add < d or add > D:
                add = gauss((d+D)/2, standard_deviation)
            add_list.append(add)
    d_v = [view for view in d_v if view not in remove_list] + add_list
    d_v = [int(view) for view in d_v]  # dimension of views = integer
    # d_v = list of views dimension from the highest to the lowest
    d_v.sort(reverse=True)
    # Dimension of latent space Z (multiplied by Z_factor)
    dim_Z = Z_factor*latent_space_dimension(d_v, R)
    # Number of informative features
    n_informative = round(dim_Z/n_informative_divid)
    # Generation of latent space Z
    Z, y = make_classification(n_samples=n_samples, n_features=dim_Z, n_informative=n_informative, n_redundant=0, 
                               n_repeated=0, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, weights=None, 
                               flip_y=0.01, class_sep=n_clusters_per_class*class_sep_factor, random_state=None)
        
    I_q = np.array([i for i in range(Z.shape[1])])  # 1D-array of Z columns numero
    meta_I_v = []
    results = []
    for view in range(n_views):
        # choice d_v[view] numeros of Z columns uniformly from I_q
        I_v = np.random.choice(I_q, size=d_v[view], replace=False)  # tirage dans I_q sans remise de taille d_v[view]
        meta_I_v += list(I_v)
        # projection of Z along the columns in I_v
        X_v = projection(Z, I_v)
        results.append((X_v, I_v))
        # remove R*d_v[view] columns numeros of I_v form I_q
        elements_to_remove = np.random.choice(I_v, size=floor(R*d_v[view]), replace=False)  # tirage dans I_v sans remise de taille floor(R*d_v[view])
        I_q = np.setdiff1d(I_q, elements_to_remove)  # I_q less elements from elements_to_remove
    unsued_dimensions_list = [column for column in I_q if column not in meta_I_v]
    unsued_dimensions_percent = round((len(unsued_dimensions_list) / dim_Z)*100, 2)
    return Z, y, results, unsued_dimensions_percent, n_informative


def results_to_csv(path, latent_space, integer_labels, multiviews_list):
    """
    Create length of multiviews_list + 2 csv files to the indicated path
    Files name :
        latent_space.csv for latent_space
        integer_labels.csv for integer_labels
        view0.csv for multiviews_list[0]
    
    Parameters:
    -----------
    path : str
    latent_space : array
    integer_labels : 1D array
    multiviews_list : list of tuples
        
    Returns:
    --------
    None
    """
    df_latent_space = pd.DataFrame(latent_space)
    df_latent_space.to_csv(path+'latent_space.csv', index=False)
    
    df_labels = pd.DataFrame(integer_labels)
    df_labels.to_csv(path+'integer_labels.csv', index=False)
    
    cpt = 0
    for view_tuple in multiviews_list:
        df_view = pd.DataFrame(view_tuple[0], columns=view_tuple[1])
        df_view.to_csv(path+'view'+str(cpt)+'.csv', index=False)
        cpt += 1

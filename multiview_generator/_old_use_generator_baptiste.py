import os
import numpy as np

import parameters
from multiviews_datasets import generator_multiviews_dataset, results_to_csv
from tests.test_classifier import score_multiviews_n_samples, graph_comparaison_classifier_scores_n_samples, score_multiviews_R, score_multiviews_Z_factor, score_multiviews_n_views_R, score_multiviews_class_sep, score_one_multiview_dataset, score_multiviews_n_informative_divided

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


n_samples = 100
n_views = 3
n_classes = 2
Z_factor = 1
R = 0
n_clusters_per_class = 1
class_sep_factor = 100
n_informative_divid = 1
standard_deviation = 2
d = 4
D = 10

path = "/home/baptiste/Documents/Datasets/Generated/try_outlier/"
if not os.path.exists(path):
    os.mkdir(path)

Z, y, results, unsued_dimensions_percent, n_informative = generator_multiviews_dataset(n_samples, n_views, n_classes,
                                                                                       Z_factor, R,
                                                                                       n_clusters_per_class,
                                                                                       class_sep_factor,
                                                                                       n_informative_divid, d, D,
                                                                                       standard_deviation)
print(y[:10])
print(unsued_dimensions_percent)
print(n_informative)
print(Z.shape)
y[:10] = np.invert(y[:10].astype(bool)).astype(int)
print(y[:10])
results_to_csv(path, Z, y, results)


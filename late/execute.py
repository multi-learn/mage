import os
import numpy as np

from multiviews_datasets_generator import generator_multiviews_dataset, results_to_csv

n_samples = 200 #Number of samples in tha dataset
n_views = 4 # Number of views in the dataset
n_classes = 2 # Number of classes in the dataset
Z_factor = 1 # Z dim = latent_space_dim * z_factor
R = 0 # Precentage of non-redundant features in the view
n_clusters_per_class = 1 # Number of clusters for each class
class_sep_factor = 100 # Separation between the different classes
n_informative_divid = 1 # Divides the number of informative features in the latent space
standard_deviation = 2
d = 4
D = 10
random_state = 42
n_outliers = 10

path = "/home/baptiste/Documents/Datasets/Generated/outliers_dset/"
if not os.path.exists(path):
    os.mkdir(path)

Z, y, results, unsued_dimensions_percent, n_informative = generator_multiviews_dataset(n_samples, n_views, n_classes,
                                                                                       Z_factor, R,
                                                                                       n_clusters_per_class,
                                                                                       class_sep_factor,
                                                                                       n_informative_divid, d, D,
                                                                                       standard_deviation)
print(unsued_dimensions_percent)
print(n_informative)
print(Z.shape)
changing_labels_indices = np.random.RandomState(random_state).choice(np.arange(y.shape[0]), n_outliers)
y[changing_labels_indices] = np.invert(y[changing_labels_indices].astype(bool)).astype(int)
results_to_csv(path, Z, y, results)
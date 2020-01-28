import os
import yaml
import numpy as np
from sklearn.datasets import make_classification
from math import ceil, floor
import pandas as pd
import h5py

class MultiviewDatasetGenetator():

    def __init__(self, n_samples=100, n_views=2, n_classes=2,
                                Z_factor=2,
                                R=0,
                                n_clusters_per_class=1,
                                class_sep_factor=10,
                                n_informative_divid=2,
                                d=4,
                                D=10,
                                standard_deviation=2,
                                weights=None,
                                flip_y=0.0,
                                random_state=42, config_path=None):
        if config_path is not None:
            with open(config_path) as config_file:
                args = yaml.safe_load(config_file)
                self.__init__(**args)
        else:
            self.n_samples = n_samples
            self.n_views = n_views
            self.n_classes = n_classes
            self.Z_factor = Z_factor
            self.R = R
            self.n_clusters_per_class = n_clusters_per_class
            self.class_sep_factor = class_sep_factor
            self.n_informative_divid = n_informative_divid
            self.d = d
            self.D = D
            self.standard_deviation = standard_deviation
            self.weights = weights
            self.flip_y = flip_y
            if isinstance(random_state, np.random.RandomState):
                self.random_state = random_state
            elif isinstance(random_state, int):
                self.random_state = np.random.RandomState(random_state)
            else:
                raise ValueError("random_sate must be np.random.RandomState or int")

    def generate(self):
        if self.n_views < 2:
            raise ValueError("n_views >= 2")
        if self.n_classes < 2:
            raise ValueError("n_classes >= 2")
        if self.Z_factor < 1:
            raise ValueError(
                "Z_factor >= 1 pour le bon fonctionnement de l'algorithme")
        if (self.R < 0) or (self.R > 1):
            raise ValueError("0 <= R <= 1")
        if self.n_clusters_per_class < 1:
            raise ValueError("n_clusters_per_class >= 1")
        if self.class_sep_factor < 0:
            raise ValueError("class_sep_factor >= 0")
        if self.n_informative_divid < 1:
            raise ValueError("n_informative_divid >= 1")
        if self.d < 1:
            raise ValueError("d >= 1")
        if (self.d + self.D) / 2 - 3 * self.standard_deviation < 1:
            raise ValueError(
                "Il faut que (d+D)/2 - 3*standard_deviation >= 1 pour avoir des valeurs positives non nulles lors de l'emploi de la loi normale")

        # n_views dimension of view v values randomly from N((d+D)/2, standard_deviation^2)
        d_v = self.random_state.normal(loc=(self.d + self.D) / 2,
                               scale=self.standard_deviation,
                               size=self.n_views)
        d_v = list(d_v)
        remove_list, add_list = [], []
        for dim_view in d_v:
            if dim_view < self.d or dim_view > self.D:  # 1 <= d <= dim_view <= D
                remove_list.append(dim_view)
                add = -1
                while add < self.d or add > self.D:
                    add = self.random_state.normal((self.d + self.D) / 2, self.standard_deviation)
                add_list.append(add)
        d_v = [view for view in d_v if view not in remove_list] + add_list
        d_v = [int(view) for view in d_v]  # dimension of views = integer
        # d_v = list of views dimension from the highest to the lowest
        d_v.sort(reverse=True)
        # Dimension of latent space Z (multiplied by Z_factor)
        self.dim_Z = self.Z_factor * self.latent_space_dimension(d_v)
        # Number of informative features
        self.n_informative = round(self.dim_Z / self.n_informative_divid)
        # Generation of latent space Z
        self.Z, self.y = make_classification(n_samples=self.n_samples, n_features=self.dim_Z,
                                   n_informative=self.n_informative, n_redundant=0,
                                   n_repeated=0, n_classes=self.n_classes,
                                   n_clusters_per_class=self.n_clusters_per_class,
                                   weights=self.weights,
                                   flip_y=self.flip_y,
                                   class_sep=self.n_clusters_per_class * self.class_sep_factor,
                                   random_state=self.random_state, shuffle=False)
        I_q = np.arange(self.Z.shape[1])
        meta_I_v = []
        self.results = []
        for view in range(n_views):
            # choice d_v[view] numeros of Z columns uniformly from I_q
            I_v = self.random_state.choice(I_q, size=d_v[view],
                                   replace=False)  # tirage dans I_q sans remise de taille d_v[view]
            meta_I_v += list(I_v)
            # projection of Z along the columns in I_v
            X_v = self.projection( I_v)
            self.results.append((X_v, I_v))
            # remove R*d_v[view] columns numeros of I_v form I_q
            elements_to_remove = self.random_state.choice(I_v,
                                                  size=floor(self.R * d_v[view]),
                                                  replace=False)  # tirage dans I_v sans remise de taille floor(R*d_v[view])
            I_q = np.setdiff1d(I_q,
                               elements_to_remove)  # I_q less elements from elements_to_remove
        self.unsued_dimensions_list = [column for column in I_q if
                                  column not in meta_I_v]
        self.unsued_dimensions_percent = round(
            (len(self.unsued_dimensions_list) / self.dim_Z) * 100, 2)

    def projection(self, chosen_columns_list):
        """
        Returns the projection of latent_space on the columns of chosen_columns_list (in chosen_columns_list order)

        Parameters:
        -----------
        chosen_columns_list : list

        Returns:
        --------
        an array of dimension (number of rows of latent_space, length of chosen_columns_list)
        """
        return self.Z[:, chosen_columns_list]

    def latent_space_dimension(self, views_dimensions_list):
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
        dimension = ceil(self.R * sum(views_dimensions_list))

        if dimension < max_view_dimension:
            dimension = max_view_dimension

        reduced_dimension = dimension
        remove_sum = 0

        for num_view in range(1, len(views_dimensions_list)):
            view_prec = views_dimensions_list[num_view - 1]
            view_current = views_dimensions_list[num_view]
            remove = floor(self.R * view_prec)
            remove_sum += remove
            if reduced_dimension - remove < view_current:
                dimension += view_current - (reduced_dimension - remove)
            reduced_dimension = dimension - remove_sum

        return dimension

    def to_csv(self, saving_path="."):
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
        df_latent_space = pd.DataFrame(self.Z)
        df_latent_space.to_csv(os.path.join(saving_path, 'latent_space.csv')
                               , index=False)

        df_labels = pd.DataFrame(self.y)
        df_labels.to_csv(os.path.join(saving_path, 'integer_labels.csv'),
                         index=False)

        for view_index, view_tuple in enumerate(self.results):
            df_view = pd.DataFrame(view_tuple[0], columns=view_tuple[1])
            df_view.to_csv(os.path.join(saving_path,
                                        'view'+str(view_index)+'.csv'),
                           index=False)

    def to_hdf5(self, saving_path=".", name="generated_dset"):

        dataset_file = h5py.File(os.path.join(saving_path, name+".hdf5"), 'w')

        labels_dataset = dataset_file.create_dataset("Labels",
                                                     shape=self.y.shape,
                                                     data=self.y)

        labels_names = ["Label_1", "Label_0"]

        labels_dataset.attrs["names"] = [
            label_name.encode() if not isinstance(label_name, bytes)
            else label_name for label_name in labels_names]

        for view_index, (data, feature_indices) in enumerate(self.results):
            df_dataset = dataset_file.create_dataset("View" + str(view_index),
                                                     shape=data.shape,
                                                     data=data)

            df_dataset.attrs["sparse"] = False
            df_dataset.attrs["name"] = "GeneratedView"+str(view_index)

        meta_data_grp = dataset_file.create_group("Metadata")

        meta_data_grp.attrs["nbView"] = len(self.results)
        meta_data_grp.attrs["nbClass"] = np.unique(self.y)
        meta_data_grp.attrs["datasetLength"] = \
        self.results[0][0].shape[0]

        meta_data_grp.create_dataset("example_ids", data=np.array(
            ["gen_example_" + str(ex_indx) for ex_indx in
             range(self.results[0][0].shape[0])]).astype(
            np.dtype("S100")), dtype=np.dtype("S100"))

        dataset_file.close()

if __name__=="__main__":
    n_samples = 100  # Number of samples in tha dataset
    n_views = 4  # Number of views in the dataset
    n_classes = 2  # Number of classes in the dataset
    Z_factor = 2  # Z dim = latent_space_dim * z_factor
    R = 0  # Precentage of non-redundant features in the view
    n_clusters_per_class = 1  # Number of clusters for each class
    class_sep_factor = 10000  # Separation between the different classes
    n_informative_divid = 2  # Divides the number of informative features in the latent space
    standard_deviation = 2
    d = 4  # View size lower limit
    D = 10  # View size upper limit
    flip_y = 0.00  # Ratio of label noise
    random_state = 42
    weights = None # The proportions of examples in each class

    path = "/home/baptiste/Documents/Datasets/Generated/metrics_dset/"
    name = "metrics"
    if not os.path.exists(path):
        os.mkdir(path)

    multiview_generator = MultiviewDatasetGenetator(n_samples=n_samples,
                                                    n_views=n_views,
                                                    n_classes=n_classes,
                                                    Z_factor=Z_factor,
                                                    R=R,
                                                    n_clusters_per_class=n_clusters_per_class,
                                                    class_sep_factor=class_sep_factor,
                                                    n_informative_divid=n_informative_divid,
                                                    d=d,
                                                    D=D,
                                                    standard_deviation=standard_deviation,
                                                    flip_y=flip_y,
                                                    weights=weights,
                                                    random_state=random_state)

    multiview_generator.generate()
    multiview_generator.to_hdf5(saving_path=path, name=name)
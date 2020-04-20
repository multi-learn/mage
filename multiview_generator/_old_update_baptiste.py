import os
from abc import abstractmethod, ABC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
import yaml
import numpy as np
from math import ceil, floor
import pandas as pd
import h5py


class MultiviewDatasetGenetator():
    def __init__(self, n_samples=100, n_views=2, n_classes=2,
                 # Z_factor=2,
                 # R=0,
                 n_clusters_per_class=1,
                 class_sep=1.0,
                 # n_informative_divid=2,
                 lower_dim=4,
                 higher_dim=10,
                 # standard_deviation=2,
                 class_weights=None,
                 flip_y=0.0,
                 # n_informative_weights=None,
                 random_state=42, config_path=None,
                 example_subsampling_method="block",
                 example_subsampling_config={},
                 feature_subampling_method="block",
                 feature_subsampling_config={},
                 redundancy=None,
                 methods="uniform",
                 view_dims=None,
                 estimator_name="LOneOneScore",
                 estimator_config={},
                 build_method="iterative",
                 precision=0.1,
                 priority="random",
                 confusion_matrix=None,
                 n_informative=10,
                 step=1):
        if config_path is not None:
            with open(config_path) as config_file:
                args = yaml.safe_load(config_file)
                self.__init__(**args)
        else:
            self.n_samples = n_samples
            self.n_views = n_views
            self.n_classes = n_classes
            self.n_clusters_per_class = n_clusters_per_class
            self.class_sep = class_sep
            self.class_weights = class_weights
            self.flip_y = flip_y
            self.example_subsampling_method= example_subsampling_method
            self.n_informative_weights = ""
            self.feature_subampling_method = feature_subampling_method
            self.feature_subsampling_config = feature_subsampling_config
            self.example_subsampling_config = example_subsampling_config
            self.redundancy = redundancy
            self.estimator_name = estimator_name
            self.build_method = build_method
            self.precision = precision
            self.priority = priority
            self.n_informative = n_informative
            self.estimator_config = estimator_config
            self.step = step
            if isinstance(methods, list):
                self.methods = methods
            elif isinstance(methods, str):
                self.methods = [methods for _ in range(self.n_views)]
            else:
                raise ValueError("methods should be list or string not {}".format(type(methods)))
            if isinstance(random_state, np.random.RandomState):
                self.random_state = random_state
            elif isinstance(random_state, int):
                self.random_state = np.random.RandomState(random_state)
            else:
                raise ValueError("random_sate must be np.random.RandomState or int")
            if view_dims is None:
                # TODO
                self.view_dims = self.random_state.randint(max(n_informative, lower_dim), high=max(n_informative+1,higher_dim), size=self.n_views)
            else:
                self.view_dims = np.asarray(view_dims)
            if confusion_matrix is None:
                self.input_confusion_matrix = ""
            elif isinstance(confusion_matrix, str):
                self.input_confusion_matrix = np.genfromtxt(input_confusion_matrix, delimiter=",", dtype=float)
            else:
                self.input_confusion_matrix = np.asarray(confusion_matrix)
            # if error_matrix is not None:
            #     self.error_matrix = error_matrix
            # else:
            #     self.error_matrix = self.random_state.uniform(size=(self.n_classes, self.n_views))
            # self.n_informative_divid = n_informative_divid
            # self.d = lower_dim
            # self.D = higher_dim
            # self.standard_deviation = standard_deviation
            # self.Z_factor = Z_factor
            # self.R = R


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
                                        'view'+str(view_index)+"_"+str(self.n_informative_weights[view_index])+'.csv'),
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
            df_dataset.attrs["name"] = "GeneratedView"+str(view_index)+"_"+str(self.n_informative_weights[view_index])

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

    def get_dataset(self):
        self.latent_space, self.y = make_classification(n_samples=self.n_samples,
                                                n_features=self.n_informative,
                                                n_informative=self.n_informative,
                                                n_redundant=0,
                                                n_repeated=0,
                                                n_classes=self.n_classes,
                                                n_clusters_per_class=self.n_clusters_per_class,
                                                weights=self.class_weights,
                                                flip_y=self.flip_y,
                                                class_sep =self.class_sep,
                                                shuffle=False,
                                                random_state=self.random_state)
        self.view_matrices = [self.gen_random_matrix(self.methods[view_index],
                                                     view_index)
                              for view_index in range(self.n_views)]
        self.output_confusion_matrix = np.zeros((self.n_classes, self.n_views))
        for view_index in range(self.n_views):
            for class_index in range(self.n_classes):
                # Restarting with all the available features
                self.available_feature_indices = np.arange(self.n_informative)
                class_example_indices = np.where(self.y==class_index)[0]
                self.get_class_subview(class_example_indices, view_index, class_index)

    def gen_random_matrix(self, method, view_index , low=0, high=1):
        if method == "uniform":
            return self.random_state.uniform(low=low, high=high,
                                             size=(self.n_samples,
                                                   self.view_dims[view_index]))
        else:
            raise ValueError("Random matrix method {} is not defined".format(method))

    def get_class_subview(self, class_examples_indices, view_index, class_index):
        if self.build_method == "iterative":
            n_examples = 1
            n_features = 1
            dist = 100
            min_dist = 100
            min_dist_score = 0
            print("Restarting a view", self.available_feature_indices.shape[0])
            while dist > self.precision:
                if (n_examples >= self.latent_space.shape[0] and n_features >= self.latent_space.shape[1]) or self.available_feature_indices.shape[0]<1:
                    raise ValueError("For view {}, class {}, unable to attain a score of {} with a "
                                     "precision of {}, the best score "
                                     "was {}".format(view_index, class_index, self.input_confusion_matrix[class_index, view_index],
                                                     self.precision,
                                                     min_dist_score))
                chosen_example_indices = self.sub_sample_examples(class_examples_indices,
                                                                  n_examples,
                                                                  **self.example_subsampling_config)
                chosen_features_indices = self.sub_sample_features(n_features,
                                                                   **self.feature_subsampling_config)
                couples = self.make_couples(chosen_example_indices, chosen_features_indices)
                for [row_idx, col_idx] in couples:
                    self.view_matrices[view_index][row_idx, col_idx] = self.latent_space[row_idx, col_idx]
                estimator_value = self.get_estimator(chosen_features_indices,
                                              chosen_example_indices,
                                              self.view_matrices[view_index][class_examples_indices, :])
                print(estimator_value)
                # print("\t Target : {}, Val : {}".format(self.input_confusion_matrix[class_index, view_index], estimator_value))
                # print("\t Examples : {}, Features : {}".format(n_examples, n_features))
                dist = abs(self.input_confusion_matrix[class_index, view_index] - estimator_value)
                if dist < min_dist:
                    min_dist = dist
                    min_dist_score = estimator_value
                if self.priority == "both":
                    n_examples += self.step
                    n_features += self.step
                elif self.priority=="examples":
                    if n_examples < self.latent_space.shape[0]:
                        n_examples += self.step
                    else:
                        n_examples = 1
                        n_features += self.step
                if self.priority == "features":
                    if n_features < self.latent_space.shape[1]:
                        n_features += self.step
                    else:
                        n_features = 1
                        n_examples += self.step
                if self.priority == "random":
                    n_examples += self.random_state.randint(0, self.step+1)
                    n_features += self.random_state.randint(0, self.step+1)
        self.output_confusion_matrix[class_index, view_index] = estimator_value
        self.remove_feature_indices(chosen_features_indices)

    #TODO
    def make_couples(self, row_indices, col_indices):
        couples = [[row, col] for col in col_indices for row in row_indices]
        return couples

    def remove_feature_indices(self, indices):
        if self.redundancy == None:
            self.available_feature_indices = np.array([idx for idx in self.available_feature_indices if idx not in indices])
        elif self.redundancy == "replace":
            pass

    def get_estimator(self, chosen_feature_indices, chosen_example_indices, class_subview):
        if self.estimator_name == "LOneOneScore":
            return LOneOneScore(self.random_state).score(self.latent_space, class_subview, self.y, patch_size=len(chosen_example_indices)*len(chosen_feature_indices))
        elif self.estimator_name == "DTScore":
            return DTScore(self.random_state).score(self.latent_space, class_subview, self.y,**self.estimator_config)
        else:
            raise ValueError("Estimator {} is not defined".format(self.estimator_name))

    def sub_sample(self, indices, quantity, method, beginning=None):
        if quantity > len(indices):
            # TODO Warning
            quantity = len(indices)
        # print(quantity)
        if method == "block":
            if beginning is None and quantity!=len(indices):
                beginning = self.random_state.randint(0, len(
                    indices) - quantity)
            if  beginning is not None and len(indices)-beginning > quantity :
                return indices[beginning:quantity + beginning]
            else:
                #TODO Wargning on the size of the outût
                return indices[-quantity:]
        if method == "choice":
            return self.random_state.choice(indices, quantity, replace=False)

    def sub_sample_features(self, n_features, **kwargs):
        return self.sub_sample(self.available_feature_indices, n_features, self.feature_subampling_method, **kwargs)

    def sub_sample_examples(self, class_examples_indices, n_examples, **kwargs):
        return self.sub_sample(class_examples_indices, n_examples, self.example_subsampling_method, **kwargs)

    def to_hdf5_mc(self, saving_path=".", name="generated_dset"):

        dataset_file = h5py.File(os.path.join(saving_path, name+".hdf5"), 'w')

        labels_dataset = dataset_file.create_dataset("Labels",
                                                     shape=self.y.shape,
                                                     data=self.y)

        labels_names = ["Label_1", "Label_0"]

        labels_dataset.attrs["names"] = [
            label_name.encode() if not isinstance(label_name, bytes)
            else label_name for label_name in labels_names]

        for view_index, data in enumerate(self.view_matrices):
            df_dataset = dataset_file.create_dataset("View" + str(view_index),
                                                     shape=data.shape,
                                                     data=data)

            df_dataset.attrs["sparse"] = False
            df_dataset.attrs["name"] = "GeneratedView"+str(view_index)

        meta_data_grp = dataset_file.create_group("Metadata")

        meta_data_grp.attrs["nbView"] = len(self.view_matrices)
        meta_data_grp.attrs["nbClass"] = np.unique(self.y)
        meta_data_grp.attrs["datasetLength"] = \
        self.view_matrices[0].shape[0]

        meta_data_grp.create_dataset("example_ids", data=np.array(
            ["gen_example_" + str(ex_indx) for ex_indx in
             range(self.view_matrices[0].shape[0])]).astype(
            np.dtype("S100")), dtype=np.dtype("S100"))

        dataset_file.close()
        np.savetxt(os.path.join(saving_path, "output_confusion_matrix.csv"),self.output_confusion_matrix)


class InformationScorer(ABC):

    def __init__(self, random_state):
        self.random_state = random_state

    @abstractmethod
    def score(self, latent_space, class_subview, y):
        pass


class LOneOneScore(InformationScorer):

    def score(self, latent_space, class_subview, y, patch_size=100):
        return 1 - patch_size/(class_subview.shape[0]*latent_space.shape[1])

class DTScore(InformationScorer):

    def __init__(self, random_state):
        super().__init__(random_state)

    def score(self, latent_space, class_subview, y, scoring="zero_one_loss",
              n_splits=7, low=None, high=None, sample_ratio=1, **kwargs):
        dt = DecisionTreeClassifier(**kwargs)
        if low is None:
            low = np.min(class_subview)
        if high is None:
            high = np.max(class_subview)
        dataset = np.concatenate((class_subview,
                                 self.random_state.uniform(low=low, high=high,
                                             size=(int(class_subview.shape[0]*sample_ratio),
                                                   class_subview.shape[1]))),
                                 axis=0)
        y = np.zeros(int(class_subview.shape[0]*(1+sample_ratio)))
        to_detect_indices = np.arange(class_subview.shape[0])
        y[:-int(class_subview.shape[0]*sample_ratio)] = np.ones(int(class_subview.shape[0]*sample_ratio))
        scores = np.zeros(n_splits)
        skf = StratifiedKFold(n_splits=n_splits)
        for fold_index, (train_indices, test_indices) in enumerate(skf.split(dataset, y)):
            intersection = np.intersect1d(test_indices, to_detect_indices, assume_unique=True)
            dt.fit(dataset[train_indices], y[train_indices])
            pred = dt.predict(dataset[intersection])
            if scoring=="zero_one_loss":
                scores[fold_index] = zero_one_loss(y_true=y[intersection],
                                                   y_pred=pred)
            else:
                raise ValueError("{} is not a valid scoring parameter".format(scoring))
        return np.mean(scores)






if __name__=="__main__":
    n_samples = 500  # Number of samples in tha dataset
    n_views = 4  # Number of views in the dataset
    n_classes = 3  # Number of classes in the dataset
    Z_factor = 10  # Z dim = latent_space_dim * z_factor
    R = 0  # Precentage of non-redundant features in the view
    n_clusters_per_class = 1  # Number of clusters for each class
    class_sep = 0.1 # Separation between the different classes
    n_informative = 100 # Divides the number of informative features in the latent space
    flip_y = 0.00  # Ratio of label noise
    random_state = 42
    lower_dim = 50
    higher_dim = 100
    class_weights = None # The proportions of examples in each class
    n_informative_weights = np.array([0.3,0.3,0.7,0.3,0.3])  # The proportion of the number of informative features for each view
    input_confusion_matrix = np.array([np.array([0.4, 0.2, 0.3, 0.1]),
                                       np.array([0.3, 0.3, 0.3, 0.1]),
                                       np.array([0.05, 0.1, 0.3, 0.1])])
    precision = 0.05
    example_subsampling_method = "block"
    feature_subampling_method = "block"
    estimator_name = "DTScore"
    estimator_config = {"max_depth": n_informative}

    path = "/home/baptiste/Documents/Datasets/Generated/confusion/"
    name = "confusion_test"
    if not os.path.exists(path):
        os.mkdir(path)
    # class_seps = [0.1,0.3,0.5,0.6,0.7,0.9,1.0,2.0,10,100]
    class_seps =[0.1]
    scores_sep = np.zeros((len(class_seps), n_classes))
    for sep_index, class_sep in enumerate(class_seps):

        multiview_generator = MultiviewDatasetGenetator(n_samples=n_samples,
                                                        n_views=n_views,
                                                        n_classes=n_classes,
                                                        n_clusters_per_class=n_clusters_per_class,
                                                        class_sep=class_sep,
                                                        n_informative=n_informative,
                                                        lower_dim=lower_dim,
                                                        higher_dim=higher_dim,
                                                        flip_y=flip_y,
                                                        class_weights=class_weights,
                                                        random_state=random_state,
                                                        example_subsampling_method=example_subsampling_method,
                                                        feature_subampling_method=feature_subampling_method,
                                                        confusion_matrix=input_confusion_matrix,
                                                        precision=precision,
                                                        estimator_name=estimator_name,
                                                        )

        def inter(arr1, arr2):
            return np.array([i for i in arr1 if i in arr2])


        ratio = 0.8
        n_splits = 40

        multiview_generator.get_dataset()
        print(" Input : ")
        print( multiview_generator.input_confusion_matrix)
        print("\n Estimated Output : ")
        print(multiview_generator.output_confusion_matrix)
        scores = np.zeros((n_classes, n_views))
        class_indices = [np.array([i for i in np.arange(n_samples) if multiview_generator.y[i]==k]) for k in range(n_classes)]
        mets = np.zeros((n_classes, n_views, n_splits))
        for view_index, view in enumerate(multiview_generator.view_matrices):
            dt = DecisionTreeClassifier(max_depth=view.shape[1], random_state=random_state)
            for _ in range(n_splits):
                train_set = np.random.RandomState(42).choice(np.arange(view.shape[0]), size=int(view.shape[0]*ratio), replace=False)
                test_set = np.array([i for i in np.arange(view.shape[0], dtype=int) if i not in train_set]).astype(int)
                dt.fit(view[train_set, :], multiview_generator.y[train_set])
                preds = np.zeros(n_samples)
                preds[test_set] = dt.predict(view[test_set, :])
                for k in range(n_classes):
                    mets[k, view_index, _] = zero_one_loss(multiview_generator.y[inter(class_indices[k], test_set)], preds[inter(class_indices[k], test_set)])
        print("\nDecision_tree output : ")
        print(np.mean(mets, axis=2))
        print("\n Decision tree on latent : ")
        lat = multiview_generator.latent_space
        sc =  np.zeros((n_classes, n_splits))
        dt = DecisionTreeClassifier(max_depth=lat.shape[1],
                                    random_state=random_state)
        for _ in range(n_splits):
            train_set = np.random.RandomState(42).choice(np.arange(lat.shape[0]),
                                                         size=int(
                                                             lat.shape[0] * ratio),
                                                         replace=False)
            test_set = np.array([i for i in np.arange(lat.shape[0], dtype=int) if
                                 i not in train_set]).astype(int)
            dt.fit(lat[train_set, :], multiview_generator.y[train_set])
            preds = np.zeros(n_samples)
            preds[test_set] = dt.predict(lat[test_set, :])
            for k in range(n_classes):
                sc[k, _] = zero_one_loss(
                    multiview_generator.y[inter(class_indices[k], test_set)],
                    preds[inter(class_indices[k], test_set)])
        scores_sep[sep_index] = np.mean(sc, axis=1)
    print(scores_sep)
    # multiview_generator.to_hdf5_mc(saving_path=path, name=name)



    # def generate(self):
    #     if self.n_views < 2:
    #         raise ValueError("n_views >= 2")
    #     if self.n_classes < 2:
    #         raise ValueError("n_classes >= 2")
    #     if self.Z_factor < 1:
    #         raise ValueError(
    #             "Z_factor >= 1 pour le bon fonctionnement de l'algorithme")
    #     if (self.R < 0) or (self.R > 1):
    #         raise ValueError("0 <= R <= 1")
    #     if self.n_clusters_per_class < 1:
    #         raise ValueError("n_clusters_per_class >= 1")
    #     if self.class_sep < 0:
    #         raise ValueError("class_sep_factor >= 0")
    #     if self.n_informative_divid < 1:
    #         raise ValueError("n_informative_divid >= 1")
    #     if self.d < 1:
    #         raise ValueError("d >= 1")
    #     if (self.d + self.D) / 2 - 3 * self.standard_deviation < 1:
    #         raise ValueError(
    #             "Il faut que (d+D)/2 - 3*standard_deviation >= 1 pour avoir des valeurs positives non nulles lors de l'emploi de la loi normale")
    #     if self.error_matrix.shape != (self.n_classes, self.n_views):
    #         raise "Error matrix must be of shape ({}, {}), it is of shape {}".format(self.n_classes, self.n_views, self.error_matrix.shape)
    #
    #     # n_views dimension of view v values randomly from N((d+D)/2, standard_deviation^2)
    #     d_v = self.random_state.normal(loc=(self.d + self.D) / 2,
    #                            scale=self.standard_deviation,
    #                            size=self.n_views)
    #     d_v = list(d_v)
    #     remove_list, add_list = [], []
    #     for dim_view in d_v:
    #         if dim_view < self.d or dim_view > self.D:  # 1 <= d <= dim_view <= D
    #             remove_list.append(dim_view)
    #             add = -1
    #             while add < self.d or add > self.D:
    #                 add = self.random_state.normal((self.d + self.D) / 2, self.standard_deviation)
    #             add_list.append(add)
    #     d_v = [view for view in d_v if view not in remove_list] + add_list
    #     d_v = [int(view) for view in d_v]  # dimension of views = integer
    #     # d_v = list of views dimension from the highest to the lowest
    #     d_v.sort(reverse=True)
    #     # Dimension of latent space Z (multiplied by Z_factor)
    #     self.dim_Z = self.Z_factor * self.latent_space_dimension(d_v)
    #     # Number of informative features
    #     self.n_informative = round(self.dim_Z / self.n_informative_divid)
    #     # Generation of latent space Z
    #     print("Dim Z : ", self.dim_Z, "N Info : ", self.n_informative, "View_dim", d_v)
    #     self.Z, self.y = make_classification(n_samples=self.n_samples, n_features=self.dim_Z,
    #                                          n_informative=self.n_informative, n_redundant=0,
    #                                          n_repeated=0, n_classes=self.n_classes,
    #                                          n_clusters_per_class=self.n_clusters_per_class,
    #                                          weights=self.class_weights,
    #                                          flip_y=self.flip_y,
    #                                          class_sep=self.n_clusters_per_class * self.class_sep,
    #                                          random_state=self.random_state, shuffle=False)
    #     self.informative_indices = np.arange(self.dim_Z)[:self.n_informative]
    #     I_q = np.arange(self.Z.shape[1])
    #     meta_I_v = []
    #     self.results = []
    #     for view_index in range(n_views):
    #         if self.n_informative_weights is not None and len(self.n_informative_weights)==n_views:
    #             if self.n_informative*self.n_informative_weights[view_index] > d_v[view_index]:
    #                 n_informative_view = int(self.n_informative*self.n_informative_weights[view_index])
    #                 d_v[view_index] = n_informative_view
    #                 I_v = self.random_state.choice(self.informative_indices,
    #                                                size=n_informative_view,
    #                                                replace=False)
    #             else:
    #                 n_informative_view = int(self.n_informative*self.n_informative_weights[view_index])
    #                 informative_indices = self.random_state.choice(self.informative_indices,
    #                                                size=n_informative_view,
    #                                                replace=False)
    #                 I_v = np.concatenate((informative_indices,
    #                                      self.random_state.choice(I_q,
    #                                                               size=d_v[view_index]-n_informative_view,
    #                                                               replace=False)))
    #         else:
    #         # choice d_v[view] numeros of Z columns uniformly from I_q
    #             I_v = self.random_state.choice(I_q, size=d_v[view_index],
    #                                            replace=False)  # tirage dans I_q sans remise de taille d_v[view]
    #         meta_I_v += list(I_v)
    #         # projection of Z along the columns in I_v
    #         X_v = self.projection( I_v)
    #         self.results.append((X_v, I_v))
    #         # remove R*d_v[view] columns numeros of I_v form I_q
    #         elements_to_remove = self.random_state.choice(I_v,
    #                                               size=floor(self.R * d_v[view_index]),
    #                                               replace=False)  # tirage dans I_v sans remise de taille floor(R*d_v[view])
    #         I_q = np.setdiff1d(I_q,
    #                            elements_to_remove)  # I_q less elements from elements_to_remove
    #     print("View_dim", d_v)
    #     self.unsued_dimensions_list = [column for column in I_q if
    #                               column not in meta_I_v]
    #     self.unsued_dimensions_percent = round(
    #         (len(self.unsued_dimensions_list) / self.dim_Z) * 100, 2)
    #
    # def projection(self, chosen_columns_list):
    #     """
    #     Returns the projection of latent_space on the columns of chosen_columns_list (in chosen_columns_list order)
    #
    #     Parameters:
    #     -----------
    #     chosen_columns_list : list
    #
    #     Returns:
    #     --------
    #     an array of dimension (number of rows of latent_space, length of chosen_columns_list)
    #     """
    #     return self.Z[:, chosen_columns_list]
    #
    # def latent_space_dimension(self, views_dimensions_list):
    #     """
    #     Returns the minimal dimension of latent space (enough to build the dataset) for generator_multiviews_dataset compared to views_dimensions_list
    #
    #     Parameters:
    #     -----------
    #     views_dimensions_list : list
    #     R : float
    #
    #     Returns:
    #     --------
    #     an int
    #     """
    #     max_view_dimension = max(views_dimensions_list)
    #     dimension = ceil(self.R * sum(views_dimensions_list))
    #
    #     if dimension < max_view_dimension:
    #         dimension = max_view_dimension
    #
    #     reduced_dimension = dimension
    #     remove_sum = 0
    #
    #     for num_view in range(1, len(views_dimensions_list)):
    #         view_prec = views_dimensions_list[num_view - 1]
    #         view_current = views_dimensions_list[num_view]
    #         remove = floor(self.R * view_prec)
    #         remove_sum += remove
    #         if reduced_dimension - remove < view_current:
    #             dimension += view_current - (reduced_dimension - remove)
    #         reduced_dimension = dimension - remove_sum
    #
    #     return dimension

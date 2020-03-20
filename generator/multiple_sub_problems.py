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


def format_array(input, size):
    if isinstance(input, int) or isinstance(input, float):
        return np.zeros(size)+input
    elif isinstance(input, np.ndarray):
        if size==input.shape[0]:
            return input
        else:
            raise ValueError("Input shape did not match "
                             "size : {} != {}".format(input.shape[0], size))
    else:
        raise ValueError("Must provide scalar or np.ndarray, "
                         "provided {}".format(type(input)))


def init_array_attr(attr, n_repeat, base_val=0):
    if attr is None:
        return np.ones((n_repeat, 1))*base_val
    elif type(attr)==float or type(attr)==int:
        return np.ones((n_repeat, 1))*attr
    elif isinstance(attr, np.ndarray):
        return attr.reshape((n_repeat, 1))
    else:
        raise ValueError("Wring type for attr : {}".format(type(attr)))


class MultiViewSubProblemsGenerator():
    """
    This moltiview generator uses multiple monoview subproblems in which the examples may be misdescribed.
    """

    def __init__(self, random_state=42, n_samples=100, n_classes=4, n_views=4,
                 confusion_matrix=None, class_seps=1.0, n_features=10,
                 n_informative=10, n_redundant=0, n_repeated=0, mult=2,
                 class_weights=None, redundancy=None, complementarity=None,
                 mutual_error=None):

        if isinstance(random_state, int):
            self.rs = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rs = random_state
        else:
            raise ValueError("Random state must be either en int or a "
                             "np.random.RandomState object, "
                             "here it is {}".format(random_state))

        self.class_seps = format_array(class_seps, n_views)
        self.n_features = format_array(n_features, n_views).astype(int)
        self.n_informative = format_array(n_informative, n_views).astype(int)
        self.n_redundant = format_array(n_redundant, n_views).astype(int)
        self.n_repeated = format_array(n_repeated, n_views).astype(int)
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_views = n_views
        self.redundancy = init_array_attr(redundancy, n_classes, base_val=0)
        self.mutual_error = init_array_attr(mutual_error, n_classes, base_val=0)
        self.complementarity = init_array_attr(complementarity, n_classes,
                                               base_val=0)
        if confusion_matrix is None:
            self.confusion_matrix = np.zeros((n_classes, n_views))+0.5
        elif isinstance(confusion_matrix, np.ndarray):
            if confusion_matrix.shape != (n_classes, n_views):
                raise ValueError("Confusion matrix must be of shape "
                                 "(n_classes x n_views), here it is of shape {} "
                                 "and n_classes={}, n_view={}".format(confusion_matrix.shape,
                                                                      n_classes,
                                                                      n_views))
            else:
                self.confusion_matrix = confusion_matrix
        else:
            raise ValueError("Confusion matrix of wrong type : "
                             "{} instead of np.array".format(type(confusion_matrix)))
        self.classes = np.arange(self.n_classes)
        self.mult = mult
        if class_weights is None:
            class_weights = np.ones(n_classes)
        self.class_weights = class_weights/np.sum(class_weights)
        self.n_examples_per_class = (self.class_weights * n_samples).astype(int)
        self.n_well_described = [(self.n_examples_per_class[class_index]*(1-confusion)).astype(int)
                                 for class_index, confusion in enumerate(self.confusion_matrix)]
        self.n_misdescribed = [(self.n_examples_per_class[class_index]-self.n_well_described[class_index])
                                 for class_index in range(self.n_classes)]
        self.n_samples = np.sum(self.n_examples_per_class)
        example_indices = np.arange(np.sum(self.n_examples_per_class))
        self.rs.shuffle(example_indices)
        self.class_example_indices = [example_indices[sum(self.n_examples_per_class[:ind]):
                                                      sum(self.n_examples_per_class[:ind+1])]
                                      for ind in range(self.n_classes)]
        self.well_described = [[_ for _ in range(self.n_views)] for _ in range(self.n_classes)]
        self.misdescribed = [[_ for _ in range(self.n_views)] for _ in range(self.n_classes)]
        self.redundant_indices = [_ for _ in range(self.n_classes)]
        self.mutual_error_indices = [_ for _ in range(self.n_classes)]
        self.complementarity_examples = [_ for _ in range(self.n_classes)]
        self.good_views_indices = [_ for _ in range(self.n_classes)]
        self.bad_views_indices = [_ for _ in range(self.n_classes)]
        self.available_init_indices = self.class_example_indices.copy()

    def generate_one_sub_problem(self, view_index, n_clusters_per_class=1,
                                 flip_y=0, hypercube=True,
                                 shift=0.0, scale=1.0):

        X, y = make_classification(n_samples=self.n_samples*self.mult,
                                   n_features=self.n_features[view_index],
                                   n_informative=self.n_informative[view_index],
                                   n_redundant=self.n_redundant[view_index],
                                   n_repeated=self.n_repeated[view_index],
                                   n_classes=self.n_classes,
                                   n_clusters_per_class=n_clusters_per_class,
                                   weights=self.class_weights,
                                   flip_y=flip_y,
                                   class_sep=self.class_seps[view_index],
                                   hypercube=hypercube,
                                   shift=shift,
                                   scale=scale,
                                   shuffle=False,
                                   random_state=self.rs)
        X = self.adding_error_introduction(X, y, view_index)
        return X

    def gen_redundancy(self):
        """
        Randomly selects the examples that will participate to redundancy
        (well described by all the views)
        """
        if (np.repeat(self.redundancy, self.n_views, axis=1) > 1-self.confusion_matrix).any():
            raise ValueError("Redundancy ({}) must be at least equal to the lowest accuracy rate of all the confusion matrix ({}).".format(self.redundancy, np.min(1-self.confusion_matrix, axis=1)))
        else:
            for class_index, redundancy in enumerate(self.redundancy):
                self.redundant_indices[class_index] = self.rs.choice(self.available_init_indices[class_index],
                                                        size=int(self.n_examples_per_class[class_index]*redundancy))
                self.remove_available(self.available_init_indices, self.redundant_indices[class_index], class_index)

    def gen_mutual_error(self):
        """
        Randomly selects the examples that will participate to mutual error
        (mis-described by all the views)
        """
        if (np.repeat(self.mutual_error, self.n_views, axis=1)>self.confusion_matrix).any():
            raise ValueError(
                "Mutual error ({}) must be at least equal to the lowest error rate of all the confusion matrix ({}).".format(
                    self.mutual_error, np.min(1 - self.confusion_matrix, axis=1)))
        else:
            for class_index, mutual_error in enumerate(self.mutual_error):
                self.mutual_error_indices[class_index] = self.rs.choice(self.available_init_indices[class_index],
                                                    size=int(self.n_examples_per_class[class_index]*mutual_error),
                                                                        replace=False)
                self.remove_available(self.available_init_indices,
                                      self.mutual_error_indices[class_index],
                                          class_index)

    def gen_complementarity(self):
        """
        Randomly selects the examples that will participate to complementarity
        (well described by a fraction of the views)
        """
        # TODO complementarity level
        if ((self.complementarity*self.n_examples_per_class)[0]>np.array([len(inds) for inds in self.available_init_indices])).any():
            raise ValueError("Complementarity ({}) must be at least equal to the lowest accuracy rate of all the confusion matrix ({}).".format(self.redundancy, np.min(1-self.confusion_matrix, axis=1)))
        else:
            for class_index, complementarity in enumerate(self.complementarity):
                self.complementarity_examples[class_index]  = self.rs.choice(self.available_init_indices[class_index],
                                                  size=int(self.n_examples_per_class[class_index]*complementarity),
                                                               replace=False)
                self.good_views_indices[class_index] = [self.rs.choice(np.arange(self.n_views),
                                                                       size=self.rs.randint(1, self.n_views),
                                                                       replace=False)
                                                        for _ in self.complementarity_examples[class_index]]
                self.bad_views_indices[class_index] = [np.array([ind
                                                                 for ind
                                                                 in range(self.n_views)
                                                                 if ind not in self.good_views_indices[class_index][ex_ind]])
                                                       for ex_ind, _ in enumerate(self.complementarity_examples[class_index]) ]
                self.remove_available(self.available_init_indices, self.complementarity_examples[class_index],
                                      class_index)

    def gen_example_indices(self, ):
        """
        Selects examples accordin to their role (redundancy, ....) and then
        affects more error if needed according to the input confusion matrix)

        """
        self.gen_redundancy()
        self.gen_mutual_error()
        self.gen_complementarity()

        for class_index in range(self.n_classes):
            for view_index, view_confusion in enumerate(np.transpose(self.confusion_matrix)):
                # Mutual error examples are misdescribed in every views
                self.misdescribed[class_index][view_index] = list(self.mutual_error_indices[class_index])
                # Redundant examples are well described in every view
                self.well_described[class_index][view_index] = list(self.redundant_indices[class_index])
                # Complementar examples are well described in certain views
                # and mis described in the others.
                self.well_described[class_index][view_index] += [complem_ind
                                                                 for ind, complem_ind
                                                                 in enumerate(self.complementarity_examples[class_index])
                                                                 if view_index in self.good_views_indices[class_index][ind]]
                self.misdescribed[class_index][view_index]+=[complem_ind
                                                             for ind, complem_ind
                                                             in enumerate(self.complementarity_examples[class_index])
                                                             if view_index in self.bad_views_indices[class_index][ind]]

                # Getting the number of examples that the view must
                # describe well for this class :
                n_good_descriptions_to_get = int(self.n_well_described[class_index][view_index]-
                                                 len(self.well_described[class_index][view_index]))
                if n_good_descriptions_to_get < 0:
                    raise ValueError("For view {}, class {}, the error matrix "
                                     "is not compatible with the three "
                                     "parameters, either lower the "
                                     "error (now:{}), or lower redundancy "
                                     "(now: {}) and/or complementarity "
                                     "(now: {})".format(view_index, class_index,
                                                        self.confusion_matrix[class_index, view_index],
                                                        self.redundancy[0,0],
                                                        self.complementarity[0,0]))
                if n_good_descriptions_to_get > len(self.available_init_indices[class_index]):
                    raise ValueError("For view {}, class {}, the error matrix "
                                     "is not compatible with the three "
                                     "parameters, either increase the "
                                     "error (now:{}), or lower redundancy "
                                     "(now: {}) and/or complementarity "
                                     "(now: {})".format(view_index, class_index,
                                                        self.confusion_matrix[class_index, view_index],
                                                        self.redundancy,
                                                        self.complementarity))

                # Filling with the needed well described examples
                well_described = list(self.rs.choice(self.available_init_indices[class_index],
                                                     size=n_good_descriptions_to_get,
                                                     replace=False))
                # And misdescribed couterparts
                misdescribed = [ind for ind
                                in self.available_init_indices[class_index]
                                if ind not in well_described]

                #Compiling the different tables
                self.well_described[class_index][view_index]+=well_described
                self.well_described[class_index][view_index] = np.array(self.well_described[class_index][view_index])
                self.misdescribed[class_index][view_index]+=misdescribed
                self.misdescribed[class_index][view_index] = np.array(
                    self.misdescribed[class_index][view_index])

    def remove_available(self, available_indices, to_remove, class_index):
        """
        Removes indices from the available ones array
        """
        available_indices[class_index] = [ind
                                          for ind
                                          in available_indices[class_index]
                                          if ind not in to_remove]
        return available_indices

    def get_repartitions(self, view_index, class_index):
        n_misdescribed = self.misdescribed[class_index][view_index].shape[0]
        updated_weights = np.array([weight if ind!=class_index else 0
                                    for ind, weight
                                    in enumerate(self.class_weights)])
        updated_weights /= np.sum(updated_weights)
        repartitions = [int(n_misdescribed*updated_weights[class_ind])
                        for class_ind in range(self.n_classes)]
        if sum(repartitions) != n_misdescribed:
            avail_classes = [ind for ind in range(self.n_classes)
                                 if ind!=class_index]
            for ind in self.rs.choice(avail_classes,
                                      size=n_misdescribed - sum(repartitions),
                                      replace=False):
                repartitions[ind] += 1
        return repartitions

    def adding_error_introduction(self, latent_space, y, view_index):

        # Initializing the view to zeros
        X = np.zeros((self.n_samples, latent_space.shape[1]))

        # Getting the indices of the examples in each class
        available_indices = [list(np.where(y == class_index)[0])
                                 for class_index in range(self.n_classes)]
        for class_index in range(self.n_classes):
            n_well_described = self.well_described[class_index][view_index].shape[0]
            if n_well_described > 0:
                well_described_latent = self.rs.choice(available_indices[class_index],
                                                       size=n_well_described,
                                                       replace=False)
                available_indices = self.remove_available(available_indices, well_described_latent, class_index)
                X[self.well_described[class_index][view_index],:] = latent_space[well_described_latent,:]
            if self.misdescribed[class_index][view_index].size != 0:
                repartitions = self.get_repartitions(view_index, class_index)
                mis_described_latent = []
                print(repartitions)
                for class_ind, repartition in enumerate(repartitions):
                    mis_described_latent += list(self.rs.choice(available_indices[class_ind],
                                                           size=repartition,
                                                           replace=False))
                    available_indices = self.remove_available(available_indices,
                                                              mis_described_latent,
                                                              class_ind)
                print(len(self.misdescribed[class_index][view_index]), len(mis_described_latent))

                X[self.misdescribed[class_index][view_index],:] = latent_space[mis_described_latent, :]
        return X

    def generate_multi_view_dataset(self, ):
        self.y = None
        self.view_data=[]
        self.y = np.zeros((self.n_samples,))
        for class_index, indices in enumerate(self.class_example_indices):
            self.y[indices] = class_index
        self.gen_example_indices()
        for view_index in range(self.n_views):
            X = self.generate_one_sub_problem(view_index)
            self.view_data.append(X)
        self.to_hdf5_mc()
        return

    def to_hdf5_mc(self, saving_path=".", name="generated_dset"):

        dataset_file = h5py.File(os.path.join(saving_path, name+".hdf5"), 'w')

        labels_dataset = dataset_file.create_dataset("Labels",
                                                     shape=self.y.shape,
                                                     data=self.y)

        labels_names = ["label_"+str(i) for i in range(self.n_classes)]

        labels_dataset.attrs["names"] = [
            label_name.encode() if not isinstance(label_name, bytes)
            else label_name for label_name in labels_names]

        for view_index, data in enumerate(self.view_data):
            df_dataset = dataset_file.create_dataset("View" + str(view_index),
                                                     shape=data.shape,
                                                     data=data)

            df_dataset.attrs["sparse"] = False
            df_dataset.attrs["name"] = "GeneratedView"+str(view_index)

        meta_data_grp = dataset_file.create_group("Metadata")

        meta_data_grp.attrs["nbView"] = self.n_views
        meta_data_grp.attrs["nbClass"] = np.unique(self.y)
        meta_data_grp.attrs["datasetLength"] = \
        self.view_data[0].shape[0]

        meta_data_grp.create_dataset("example_ids", data=np.array(
            ["gen_example_" + str(ex_indx) for ex_indx in
             range(self.view_data[0].shape[0])]).astype(
            np.dtype("S100")), dtype=np.dtype("S100"))

        dataset_file.close()


    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.model_selection import StratifiedKFold
    # from sklearn.metrics import confusion_matrix
    # dt = DecisionTreeClassifier()
    # folds_gene = StratifiedKFold(n_folds, random_state=42, shuffle=True)
    # folds = folds_gene.split(np.arange(gene.y.shape[0]), gene.y)
    # folds = [[list(train), list(test)] for train, test in folds]
    # confusion_mat = np.zeros((n_folds, n_views, n_classes, n_classes))
    # n_sample_per_class = np.zeros((n_views, n_classes, n_folds))
    # for view_index in range(n_views):
    #     for fold_index, [train, test] in enumerate(folds):
    #         dt.fit(gene.view_data[view_index][train, :], gene.y[train])
    #         pred = dt.predict(gene.view_data[view_index][test, :])
    #         confusion_mat[fold_index, view_index, :, :] = confusion_matrix(gene.y[test], pred)
    #         for class_index in range(n_classes):
    #             n_sample_per_class[view_index, class_index, fold_index] = np.where(gene.y[test]==class_index)[0].shape[0]
    # confusion_mat = np.mean(confusion_mat, axis=0)
    # n_sample_per_class = np.mean(n_sample_per_class, axis=2)
    # confusion_output = np.zeros((n_classes, n_views))
    # for class_index in range(n_classes):
    #     for view_index in range(n_views):
    #         confusion_output[class_index, view_index] = 1-confusion_mat[view_index, class_index, class_index]/n_sample_per_class[view_index, class_index]
    # print(confusion_output)
    # fig = make_subplots(rows=2, cols=2, subplot_titles=[
    #     "View {}, Confusion : <br>In:{}<br>Out:{}".format(view_index,
    #                                                np.round(conf[:, view_index], 3),
    #                                                np.round(confusion_output[:, view_index], 3)) for
    #     view_index
    #     in range(n_views)],
    #                     specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, ],
    #                            [{'type': 'scatter3d'},
    #                             {'type': 'scatter3d'}, ]])
    # row = 1
    # col = 1
    # for view_index in range(n_views):
    #     for lab_index in range(n_classes):
    #         concerned_examples = np.where(gene.y == lab_index)[0]
    #         fig.add_trace(
    #             go.Scatter3d(
    #                 x=gene.view_data[view_index][concerned_examples, 0],
    #                 y=gene.view_data[view_index][concerned_examples, 1],
    #                 z=gene.view_data[view_index][concerned_examples, 2],
    #                 mode='markers', marker=dict(
    #                     size=1,  # set color to an array/list of desired values
    #                     color=DEFAULT_PLOTLY_COLORS[lab_index],
    #                     opacity=0.8
    #                 ), name="Class {}".format(lab_index)), row=row, col=col)
    #         # fig.update_layout(
    #         #             scene=dict(
    #         #             xaxis=dict(nticks=4, range=[low_range, high_range], ),
    #         #             yaxis=dict(nticks=4, range=[low_range, high_range], ),
    #         #             zaxis=dict(nticks=4, range=[low_range, high_range], ), ),)
    #     col += 1
    #     if col == 3:
    #         col = 1
    #         row += 1
    #         # fig.update_xaxes(range=[-class_sep-0.1*class_sep, +class_sep+margin_ratio*class_sep], row=row, col=col)
    #         # fig.update_yaxes(
    #         #     range=[-class_sep - 0.1 * class_sep, +class_sep + margin_ratio * class_sep],
    #         #     row=row, col=col)
    #         # fig.update_zaxes(
    #         #     range=[-class_sep - 0.1 * class_sep, +class_sep + margin_ratio * class_sep],
    #         #     row=row, col=col)
    # plotly.offline.plot(fig, filename="center_blob.html")


    #
    #
    # def replacement_error_introduction(self, latent_space, y, view_index):
    #     """
    #     Switch examples places to introduce error in the dataset
    #     :param latent_space:
    #     :return:
    #     """
    #     # Initializing the view to zeros
    #     X = np.zeros((self.n_samples, latent_space.shape[1]))
    #
    #     # Getting the indices of the examples in each class
    #     class_indices_subsets = [list(np.where(y == class_index)[0])
    #                              for class_index in range(self.n_classes)]
    #
    #     # Isolating the examples that will be switched with other blobs, and
    #     # the one that will stay in the right blob
    #     switched_inds_list = []
    #     kept_inds_list = []
    #     for class_index in range(n_classes):
    #         n_mislabelled = int(self.n_examples_per_class *
    #                             self.confusion_matrix[class_index, view_index])
    #         switched_inds = self.rs.choice(class_indices_subsets[class_index],
    #                            size=n_mislabelled, replace=False)
    #         switched_inds_list.append(switched_inds)
    #         kept_inds = [ind for ind in range(n_samples)
    #                         if ind not in switched_inds]
    #         kept_inds_list.append(kept_inds)
    #         # Filling the view with the examples that stay in their blobs
    #         X[kept_inds, :] = latent_space[kept_inds, :]
    #
    #     # The available spots for switching
    #     available_spots_list = switched_inds_list.copy()
    #
    #     # The number of examples that must be switched for each class
    #     n_switched_list = [len(switched_inds)
    #                     for switched_inds
    #                     in switched_inds_list]
    #     if sum(n_switched_list)!=0:
    #         # For each class, switch all the examples to the other classes :
    #         for class_index, switched_inds in enumerate(switched_inds_list):
    #             n_switched = len(switched_inds)
    #
    #             # Find for each class how many examples are going to be switched with it
    #             n_displaced_list = [int(n_switched * n_switched_other /
    #                                     sum(n_switched_list))
    #                                 if class_ind != class_index
    #                                 else 0
    #                                 for class_ind, n_switched_other
    #                                 in enumerate(n_switched_list)]
    #
    #             # Get the indices that will be switched to each class
    #             ind = 0
    #             displaced_inds_list = []
    #             for n_displaced in n_displaced_list:
    #                 displaced_inds_list.append(switched_inds[ind:ind+n_displaced])
    #                 ind += n_displaced
    #
    #             # Get where they are going to be displaced
    #             destinations_list = [self.rs.choice(available_spots,
    #                                            size=n_displaced_list[class_ind],
    #                                            replace=False)
    #                             for class_ind, available_spots
    #                             in enumerate(available_spots_list)]
    #
    #             # Remove the selected spots from the available ones
    #             available_spots_list = [
    #              [ind for ind in available_spots if ind not in destinations]
    #                  for available_spots, destinations
    #                  in zip(available_spots_list, destinations_list)]
    #
    #             # Fill X with the switched examples
    #             for destinations, displaced_inds in zip(destinations_list,
    #                                                     displaced_inds_list):
    #
    #                 X[destinations, :] = latent_space[displaced_inds, :]
    #                 X[displaced_inds, :] = latent_space[destinations, :]
    #     return X
    #
    #
    # def mislabel(self, latent_space, y, view_index):
    #     used_examples = []
    #     X = np.zeros((self.n_samples, latent_space.shape[1]))
    #     classes_subsets = []
    #     for class_index in range(self.n_classes):
    #         example_indices = np.where(y==class_index)[0]
    #         classes_subsets.append(list(example_indices))
    #
    #     available_spots_list = []
    #     for class_index in range(n_classes):
    #         n_mislabelled = int(self.n_examples_per_class *
    #                             self.confusion_matrix[class_index, view_index])
    #         available_spots_list.append(self.rs.choice(classes_subsets[class_index],
    #                                size=n_mislabelled, replace=False))
    #     mislabelled_example_indices_list = available_spots_list.copy()
    #     if self.mislabelling_method == "replacement":
    #         n_spots_list = [len(mislabelled_examples)
    #                         for mislabelled_examples
    #                         in mislabelled_example_indices_list]
    #         for class_index, mislabelled_example_indices in mislabelled_example_indices_list:
    #             n_mislabelled = len(mislabelled_example_indices)
    #             n_displaced_list = [int(n_mislabelled*n_spots/sum(n_spots_list))
    #                                if class_ind != class_index
    #                                else 0
    #                                for class_ind, n_spots
    #                                in enumerate(n_spots_list)]
    #             ind = 0
    #             displaced_inds_list = []
    #             for n_displaced in n_displaced_list:
    #                 displaced_inds_list.append(mislabelled_example_indices[ind:ind+n_displaced])
    #                 ind += n_displaced
    #             destinations = [self.rs.choice(available_spots,
    #                                            size=n_displaced_list[class_ind],
    #                                            replace=False)
    #                             for class_ind, available_spots
    #                             in enumerate(available_spots_list)]
    #             available_spots_list = remove_spots(available_spots_list, class_index, destinations)
    #             for displaced_inds, destination in zip(displaced_inds_list, destinations):
    #                 examples_dest = X[destination, :]
    #                 X[destination, :] = X[mislabelled_example_indices[displaced_inds], :]
    #                 X[mislabelled_example_indices[displaced_inds], :] = examples_dest
    #
    #
    #
    #
    #     for class_index in range(self.n_classes):
    #         n_mislabelled = int(self.n_examples_per_class*
    #                             self.confusion_matrix[class_index,view_index])
    #
    #         # mislabelled_examples = self.random_state.choice(classes_subsets[class_index],
    #         #                                                 size=n_mislabelled,
    #         #                                                 replace=False)
    #         # Remove the misslabelled examples from the class indices
    #         # classes_subsets[class_index] = [example_index
    #         #                                 for example_index in classes_subsets[class_index]
    #         #                                 if example_index not in mislabelled_examples]
    #         if self.mislabelling_method == "random":
    #             available_examples = sum([class_subset
    #                                            for class_ind, class_subset
    #                                            in enumerate(classes_subsets)
    #                                            if class_ind != class_index], [])
    #             available_examples = [ind for ind in available_examples if ind not in used_examples]
    #             mislabelled_examples_indices = self.rs.choice(available_examples,
    #                                                           size=n_mislabelled,
    #                                                           replace=False)
    #             used_examples += list(mislabelled_examples_indices)
    #             available_examples = [ind for ind in classes_subsets[class_index] if ind not in used_examples]
    #             good_examples_indices = self.rs.choice(available_examples,
    #                                                     size=self.n_examples_per_class-n_mislabelled)
    #             example_indices = np.concatenate((mislabelled_examples_indices,
    #                                       good_examples_indices))
    #             used_examples += list(good_examples_indices)
    #             X[class_index * self.n_examples_per_class:
    #               (class_index + 1) * self.n_examples_per_class] = latent_space[
    #                                                                example_indices,
    #                                                                :]
    #         elif self.mislabelling_method=="center_blob":
    #             mislabelled_examples_indices = self.rs.choice(classes_subsets[class_index],
    #                                                           size=n_mislabelled,
    #                                                           replace=False)
    #             good_examples_indices = np.array([ind
    #                                               for ind
    #                                               in classes_subsets[class_index]
    #                                               if ind
    #                                               not in mislabelled_examples_indices])
    #             print(n_mislabelled, good_examples_indices.shape, self.n_examples_per_class)
    #             print(X.shape)
    #             X[class_index * self.n_examples_per_class:
    #               (class_index ) * self.n_examples_per_class+n_mislabelled,
    #             :] =  self.rs.uniform(low=0, high=0.1,
    #                                   size=(n_mislabelled, latent_space.shape[1]))
    #             X[(class_index ) * self.n_examples_per_class+n_mislabelled:
    #               (class_index+1)* self.n_examples_per_class, :] = latent_space[good_examples_indices, :]
    #         else:
    #             raise ValueError('{} is not implemented as a mislabelling method'.format(self.mislabelling_method))
    #
    #         # print(classes_mislabelled_examples)
    #     #     for mislabelled_example, mislabelled_class in zip(
    #     #             mislabelled_examples, classes_mislabelled_examples):
    #     #         classes_subsets[mislabelled_class].append(mislabelled_example)
    #     # example_indices = sum(classes_subsets, [])
    #     # print(classes_subsets)
    #     return X

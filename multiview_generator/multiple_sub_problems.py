import os

import h5py
import numpy as np
import yaml
from sklearn.datasets import make_classification, make_gaussian_quantiles
from tabulate import tabulate
import pandas as pd
import inspect
from datetime import datetime

from .utils import format_array, get_config_from_file, \
    init_random_state, init_error_matrix, init_list
from .base_strs import *


class MultiViewSubProblemsGenerator:
    r"""
    This engine generates one monoview sub-problem for each view with independant data.
    If then switch descriptions between the samples to create error and difficulty in the dataset

    :param random_state: The random state or seed.
    :param n_samples: The number of samples that the dataset will contain
    :param n_classes: The number of classes in which the samples will be labelled
    :param n_views: The number of views describing the samples
    :param error_matrix: The error matrix giving in row i column j the error of the Bayes classifier on Class i for View j
    :param latent_size_multiplicator: The mutlitplicator for the latent space generated for each view. It is useful mainly if the dataset is un-balanced.
    :param n_features: The number of features describing the samples for each view (can specify an int or array-like of length ``n_views``)
    :param class_weights: The proposrtion of the dataset that will be labelled in each class. Must specify an array-like of size n_classes ([0.1,0.45,0.45] will output a dataset with with 10% of the samples in the first class and 45% in the two others.)
    :param redundancy: The proportion of the samples that will be well-decribed by all the views.
    :param complementarity: The proportion of samples that will be well-decribed only by some views
    :param complementarity_level: The proportion of views that will have a good description of the complementray samples
    :param mutual_error: The propotion of samples that will be mis-described by all the views
    :param name: The name of the dataset (will be used to name the file)
    :param config_file: The path to the yaml config file. If provided, the config fil entries will overwrite the one passes as arguments.
    :param sub_problem_type: The type of monoview-generator used in each view (make_classification, make_gaussian_quantiles, ...). If only a string is provided, the same generator will be used for all the views.
    :param sub_problem_configurations: The configuration for each of the generators. If only a dict is provided, the same configuration will be used for all the views.

    :type random_state: int or np.random.RandomState
    :type n_samples: int
    :type n_classes: int
    :type n_views: int
    :type error_matrix: np.ndarray
    :type latent_size_multiplicator: float
    :type n_features: int or array-like
    :type class_weights: float or array-like
    :type redundancy: float
    :type complementarity: float
    :type complementarity_level: float
    :type mutual_error: float
    :type name: str
    :type config_file: str
    :type sub_problem_type: str or list
    :type sub_problem_configurations: None, dict or list
    """

    def __init__(self, random_state=42, n_samples=100, n_classes=4, n_views=4,
                 error_matrix=None, latent_size_multiplicator=2, n_features=3,
                 class_weights=1.0, redundancy=0.0, complementarity=0.0,
                 complementarity_level=0.66,
                 mutual_error=0.0, name="generated_dataset", config_file=None,
                 sub_problem_type="base", sub_problem_configurations=None,
                 **kwargs):

        if config_file is not None:
            args = get_config_from_file(config_file)
            self.__init__(**args)
        else:
            self.rs = init_random_state(random_state)
            self.n_samples = n_samples
            self.n_classes = n_classes
            self.n_views = n_views
            self.name = name
            self.n_features = format_array(n_features, n_views, type_needed=int)
            self.redundancy = format_array(redundancy, n_classes,
                                           type_needed=float).reshape(
                (n_classes, 1))
            self.mutual_error = format_array(mutual_error, n_classes,
                                             type_needed=float).reshape(
                (n_classes, 1))
            self.complementarity = format_array(complementarity, n_classes,
                                                type_needed=float).reshape(
                (n_classes, 1))
            self.complementarity_level = format_array(complementarity_level, n_classes, type_needed=float).reshape(((n_classes, 1)))
            self.latent_size_mult = latent_size_multiplicator
            self._init_sub_problem_config(sub_problem_configurations,
                                          sub_problem_type)
            self.error_matrix = init_error_matrix(error_matrix, n_classes,
                                                  n_views)
            self.classes = np.arange(self.n_classes)
            self.class_weights = format_array(class_weights, n_classes,
                                              type_needed=float) / np.sum(
                class_weights)
            self._init_base_arguments()

    def generate_multi_view_dataset(self, ):
        """
        This is the main method. It will generate a multiview dataset according
        to the configuration.
        To do so,

        * it generates the labels of the multiview dataset,
        * then it generates all the subsets of examples (redundant, ...)
        * finally, for each view it generates a monview dataset according to the configuration


        :return: view_data a list contianing the views np.ndarrays and y, the label array.
        """

        self.y = np.zeros((self.n_samples,))
        for class_index, indices in enumerate(self.class_example_indices):
            self.y[indices] = class_index

        self.view_data = []
        self._gen_example_indices()
        for view_index in range(self.n_views):
            X = self._generate_one_sub_problem(view_index)
            self.view_data.append(X)
        return self.view_data, self.y

    def to_hdf5_mc(self, saving_path="."):
        """
        This is used to save the dataset in an HDF5 file, compatible with
        :summit:`SuMMIT <>`

        :param saving_path: where to save the dataset, the file will be names after the self.name attribute.
        :type saving_path: str
        :return: None
        """

        dataset_file = h5py.File(os.path.join(saving_path, self.name + ".hdf5"),
                                 'w')

        labels_dataset = dataset_file.create_dataset("Labels",
                                                     shape=self.y.shape,
                                                     data=self.y)

        labels_names = ["label_" + str(i + 1) for i in range(self.n_classes)]

        labels_dataset.attrs["names"] = [
            label_name.encode() if not isinstance(label_name, bytes)
            else label_name for label_name in labels_names]

        for view_index, data in enumerate(self.view_data):
            df_dataset = dataset_file.create_dataset("View" + str(view_index),
                                                     shape=data.shape,
                                                     data=data)

            df_dataset.attrs["sparse"] = False
            df_dataset.attrs["name"] = "generated_view_" + str(view_index + 1)

        meta_data_grp = dataset_file.create_group("Metadata")

        meta_data_grp.attrs["nbView"] = self.n_views
        meta_data_grp.attrs["nbClass"] = np.unique(self.y)
        meta_data_grp.attrs["datasetLength"] = self.view_data[0].shape[0]

        self.gen_report(save=False)
        meta_data_grp.attrs["description"] = self.report

        meta_data_grp.create_dataset("sample_ids", data=np.array(
            self.example_ids).astype(
            np.dtype("S100")), dtype=np.dtype("S100"))

        dataset_file.close()

    def gen_report(self, output_path='.', file_type="md", save=True):
        """
        Generates a markdown report based on the configuration.
        If ``save`` is True, it will be saved in ``output_path`` as <self.name>.<``file_type``> .

        :param output_path: path to store the text report.
        :type output_path: str
        :param file_type: Type of file in which the report is saved (currently supported : "md" or "txt")
        :type file_type: str
        :param save: Whether to save the string in a file or not.
        :type save: bool
        :return: The report string
        """
        report_string = "# Generated dataset description\n\n"

        report_string+= "The dataset named `{}` has been generated by [{}]({}) " \
                        "and is comprised of \n\n* {} examples, splitted in " \
                        "\n* {} classes, described by \n* {} views.\n\n".format(self.name, GENE, LINK, self.n_samples, self.n_classes, self.n_views)

        error_df = pd.DataFrame(self.error_matrix,
                                index=["Class "+str(i+1)
                                       for i in range(self.n_classes)],
                                columns=['View '+str(i+1) for i in range(self.n_views)])

        report_string += "The input error matrix is \n \n"+tabulate(error_df,
                                                                 headers='keys',
                                                                 tablefmt='github')

        report_string += "\n\n The classes are balanced as : \n\n* Class "
        report_string += '\n* Class '.join(["{} : {} examples ({}% of the dataset)".format(i+1,
                                                                                         n_ex,
                                                                                         int(ratio*100))
                                          for i, (n_ex, ratio)
                                          in enumerate(zip(self.n_examples_per_class,
                                                           self.class_weights))])

        report_string += "\n\n The views have \n\n* {}% redundancy, \n* {}% mutual error" \
                         " and \n* {}% complementarity,\n\nthe remaining examples are " \
                         "randomly mis-labelled to fit the input error matrix.\n\n".format(self.redundancy[0,0]*100, self.mutual_error[0,0]*100, self.complementarity[0,0]*100)

        report_string+="## Views description"
        for view_index in range(self.n_views):
            report_string += self.gen_view_report(view_index)

        report_string += "\n\nThis report has been automatically generated on {}".format(datetime.now().strftime("%B %d, %Y at %H:%M:%S"))
        if save:
            with open(os.path.join(output_path, "report."+file_type), "w") as output:
                output.write(report_string)
        self.report = report_string
        return report_string

    def gen_view_report(self, view_index):
        view_string = "\n\n### View "+str(view_index+1)
        view_string+="\n\nThis view is generated with {}, with the following configuration : \n```yaml\n".format(self._get_generator_report(view_index))
        view_string += yaml.dump(self.sub_problem_configurations[view_index], line_break="\n",default_flow_style=False)
        view_string+="```"
        return view_string

    def _get_generator_report(self, view_index, doc_type=".md"):
        if self.sub_problem_types[view_index] in ["make_classification", "base"]:
            return "[`make_classification`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)"
        elif self.sub_problem_types[view_index]in ["gaussian", "make_gaussian_quantiles"]:
            return "[`make_gaussian_quantiles`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html#sklearn.datasets.make_gaussian_quantiles)"

    def _init_sub_problem_config(self, sub_problem_configs, sub_problem_type):
        """
        plif
        :param sub_problem_configs:
        :param sub_problem_type:
        :return:
        """
        if sub_problem_type is None:
            self.sub_problem_types = ["base" for _ in range(self.n_views)]
        else:
            self.sub_problem_types = init_list(sub_problem_type,
                                               size=self.n_views,
                                               type_needed=str)

        if sub_problem_configs is None:
            self.sub_problem_configurations = [
                {"n_informative": int(self.n_features[view_index]),
                 "n_redundant": 0,
                 "n_repeated": 0,
                 "n_clusters_per_class": 1,
                 "class_sep": 10.0, }
                if self.sub_problem_types[view_index] == "base"
                else {} for view_index in range(self.n_views)]
        else:
            self.sub_problem_configurations = init_list(sub_problem_configs,
                                                        size=self.n_views,
                                                        type_needed=dict)

    def _init_base_arguments(self):
        self.n_examples_per_class = (
                self.class_weights * self.n_samples).astype(int)
        self.n_well_described = [
            (self.n_examples_per_class[class_index] * (1 - confusion)).astype(
                int)
            for class_index, confusion in enumerate(self.error_matrix)]
        self.n_misdescribed = [(self.n_examples_per_class[class_index] -
                                self.n_well_described[class_index])
                               for class_index in range(self.n_classes)]
        self.n_samples = np.sum(self.n_examples_per_class)
        example_indices = np.arange(int(np.sum(self.n_examples_per_class)))
        self.rs.shuffle(example_indices)
        self.class_example_indices = [
            example_indices[sum(self.n_examples_per_class[:ind]):
                            sum(self.n_examples_per_class[:ind + 1])]
            for ind in range(self.n_classes)]
        self.well_described = [[_ for _ in range(self.n_views)] for _ in
                               range(self.n_classes)]
        self.misdescribed = [[_ for _ in range(self.n_views)] for _ in
                             range(self.n_classes)]
        self.redundancy_indices = [_ for _ in range(self.n_classes)]
        self.mutual_error_indices = [_ for _ in range(self.n_classes)]
        self.complementarity_examples = [_ for _ in range(self.n_classes)]
        self.good_views_indices = [_ for _ in range(self.n_classes)]
        self.bad_views_indices = [_ for _ in range(self.n_classes)]
        self.available_init_indices = self.class_example_indices.copy()
        self.example_ids = ["example_{}".format(ind)
                            for ind
                            in range(int(np.sum(self.n_examples_per_class)))]

    def _generate_one_sub_problem(self, view_index, flip_y=0, hypercube=True,
                                  shift=0.0, scale=1.0, ):
        if self.sub_problem_types[view_index] in ["make_classification",
                                                  "base"]:
            X, y = make_classification(
                n_samples=self.n_samples * self.latent_size_mult,
                n_classes=self.n_classes,
                n_features=self.n_features[view_index],
                weights=self.class_weights,
                flip_y=flip_y,
                shift=shift,
                scale=scale,
                shuffle=False,
                random_state=self.rs,
                **self.sub_problem_configurations[view_index])
            self.sub_problem_configurations[view_index] = {
                "n_features": int(self.n_features[view_index]),
                "flip_y": flip_y,
                "hypercube": hypercube,
                "shift": shift,
                "scale": scale,
                "shuffle": False,
            ** self.sub_problem_configurations[view_index]
            }
        elif self.sub_problem_types[view_index] in ["gaussian", "make_gaussian_quantiles"]:
            n_samples_compensated = self._get_n_samples()
            X, y = make_gaussian_quantiles(n_samples=n_samples_compensated,
                                           n_features=self.n_features[
                                               view_index],
                                           n_classes=self.n_classes,
                                           shuffle=False,
                                           random_state=self.rs,
                                           cov=3.,
                                           **self.sub_problem_configurations[
                                               view_index])

            signature = inspect.signature(make_gaussian_quantiles)
            base_config = {k: v.default for k, v in signature.parameters.items()
                           if v.default is not inspect.Parameter.empty}
            config = dict((key, self.sub_problem_configurations[key])
                          if key in self.sub_problem_configurations
                          else (key, val)
                          for key, val in base_config.items() if key not in ["n_samples", "n_classes", "random_state"])
            config['n_features'] = int(self.n_features[view_index])
            self.sub_problem_configurations[view_index] = config
        else:
            raise ValueError(
                "{} is not a valid sub_problem_type".format(
                    self.sub_problem_types[view_index]))
        X = self._add_error(X, y, view_index)
        return X

    def _add_error(self, latent_space, y, view_index):

        # Initializing the view to zeros
        X = np.zeros((self.n_samples, latent_space.shape[1]))

        # Getting the indices of the examples that are available in each class
        available_indices = [list(np.where(y == class_index)[0])
                             for class_index in range(self.n_classes)]
        for class_index in range(self.n_classes):
            n_well_described = \
                self.well_described[class_index][view_index].shape[0]
            if n_well_described > 0:
                well_described_latent = self.rs.choice(
                    available_indices[class_index],
                    size=n_well_described,
                    replace=False)
                available_indices = self._remove_available(available_indices,
                                                           well_described_latent,
                                                           class_index)
                X[self.well_described[class_index][view_index],
                :] = latent_space[well_described_latent, :]
            if self.misdescribed[class_index][view_index].size != 0:
                repartitions = self._get_repartitions(view_index, class_index)
                mis_described_latent = []
                for class_ind, repartition in enumerate(repartitions):
                    mis_described_latent += list(
                        self.rs.choice(available_indices[class_ind],
                                       size=repartition,
                                       replace=False))
                    available_indices = self._remove_available(
                        available_indices,
                        mis_described_latent,
                        class_ind)

                X[self.misdescribed[class_index][view_index], :] = latent_space[
                                                                   mis_described_latent,
                                                                   :]
        return X

    def _get_n_samples(self):
        max_number_of_samples = np.max(self.n_examples_per_class)
        if self.n_samples * self.latent_size_mult / self.n_classes < max_number_of_samples:
            compensator = (max_number_of_samples * self.n_classes) / (
                    self.n_samples * self.latent_size_mult)
            return self.n_samples * self.latent_size_mult * compensator
        else:
            return self.n_samples * self.latent_size_mult

    def _gen_indices_for(self, name="redundancy", error_mat_fun=lambda x: 1 - x):
        quantities = getattr(self, name)
        indices = getattr(self, name + "_indices")
        if (np.repeat(quantities, self.n_views,
                      axis=1) > error_mat_fun(self.error_matrix)).any():
            raise ValueError(
                "{} ({}) must be at least equal to the lowest accuracy rate "
                "of all the confusion matrix ({}).".format(name,
                    quantities, np.min(error_mat_fun(self.error_matrix), axis=1)))
        else:
            for class_index, quantity in enumerate(quantities):
                indices[class_index] = self.rs.choice(
                    self.available_init_indices[class_index],
                    size=int(
                        self.n_examples_per_class[class_index] * quantity),
                    replace=False)
                self._update_example_indices(
                    indices[class_index], name,
                    class_index)
                self._remove_available(self.available_init_indices,
                                       indices[class_index],
                                       class_index)

    def _update_example_indices(self, target, target_name, class_ind):
        for ind, target_ind in enumerate(target):
            self.example_ids[target_ind] = target_name + "_{}_{}".format(ind,
                                                                         class_ind)

    def _gen_complementarity(self,):
        """
        Randomly selects the examples that will participate to complementarity
        (well described by a fraction of the views)
        """
        if ((self.complementarity * self.n_examples_per_class)[0] > np.array(
                [len(inds) for inds in self.available_init_indices])).any():
            raise ValueError(
                "Complementarity ({}) + Redundnacy ({}) + Mutual Error ({}) must be < 1 (here : {})".format(
                    self.complementarity, self.redundancy, self.mutual_error,
                self.complementarity + self.redundancy + self.mutual_error))
        else:
            for class_index, complementarity in enumerate(self.complementarity):
                n_comp = int(self.complementarity_level[class_index]*self.n_views)
                self.complementarity_examples[class_index] = self.rs.choice(
                    self.available_init_indices[class_index],
                    size=int(self.n_examples_per_class[
                                 class_index] * complementarity),
                    replace=False)
                self._update_example_indices(
                    self.complementarity_examples[class_index],
                    'Complementary', class_index)
                self.good_views_indices[class_index] = [
                    self.rs.choice(np.arange(self.n_views),
                                   size=n_comp,
                                   replace=False)
                    for _ in self.complementarity_examples[class_index]]
                self.bad_views_indices[class_index] = [np.array([ind
                                                                 for ind
                                                                 in range(
                        self.n_views)
                                                                 if ind not in
                                                                 self.good_views_indices[
                                                                     class_index][
                                                                     ex_ind]])
                                                       for ex_ind, _ in
                                                       enumerate(
                                                           self.complementarity_examples[
                                                               class_index])]
                self._remove_available(self.available_init_indices,
                                       self.complementarity_examples[
                                           class_index],
                                       class_index)

    def _gen_example_indices(self, ):
        """
        Selects examples accordin to their role (redundancy, ....) and then
        affects more error if needed according to the input confusion matrix)

        """
        self._gen_indices_for(name="redundancy", error_mat_fun=lambda x: 1-x)
        self._gen_indices_for(name="mutual_error", error_mat_fun=lambda x: x)
        self._gen_complementarity()
        for class_index in range(self.n_classes):
            for view_index, view_confusion in enumerate(
                    np.transpose(self.error_matrix)):
                # Mutual error examples are misdescribed in every views
                self.misdescribed[class_index][view_index] = list(
                    self.mutual_error_indices[class_index])
                # Redundant examples are well described in every view
                self.well_described[class_index][view_index] = list(
                    self.redundancy_indices[class_index])
                # Complementary examples are well described in certain views
                # and mis described in the others.
                self.well_described[class_index][view_index] += [complem_ind
                                                                 for
                                                                 ind, complem_ind
                                                                 in enumerate(
                        self.complementarity_examples[class_index])
                                                                 if
                                                                 view_index in
                                                                 self.good_views_indices[
                                                                     class_index][
                                                                     ind]]
                self.misdescribed[class_index][view_index] += [complem_ind
                                                               for
                                                               ind, complem_ind
                                                               in enumerate(
                        self.complementarity_examples[class_index])
                                                               if view_index in
                                                               self.bad_views_indices[
                                                                   class_index][
                                                                   ind]]

                # Getting the number of examples that the view must
                # describe well for this class :
                n_good_descriptions_to_get = int(
                    self.n_well_described[class_index][view_index] -
                    len(self.well_described[class_index][view_index]))
                if n_good_descriptions_to_get < 0:
                    raise ValueError("For view {}, class {}, the error matrix "
                                     "is not compatible with the three "
                                     "parameters, either lower the "
                                     "error (now:{}), or lower redundancy "
                                     "(now: {}) and/or complementarity "
                                     "(now: {})".format(view_index, class_index,
                                                        self.error_matrix[
                                                            class_index, view_index],
                                                        self.redundancy[0, 0],
                                                        self.complementarity[
                                                            0, 0]))
                if n_good_descriptions_to_get > len(
                        self.available_init_indices[class_index]):
                    raise ValueError("For view {}, class {}, the error matrix "
                                     "is not compatible with the three "
                                     "parameters, either increase the "
                                     "error (now:{}), or lower redundancy "
                                     "(now: {}) and/or complementarity "
                                     "(now: {})".format(view_index, class_index,
                                                        self.error_matrix[
                                                            class_index, view_index],
                                                        self.redundancy,
                                                        self.complementarity))

                # Filling with the needed well described examples
                well_described = list(
                    self.rs.choice(self.available_init_indices[class_index],
                                   size=n_good_descriptions_to_get,
                                   replace=False))
                # And misdescribed couterparts
                misdescribed = [ind for ind
                                in self.available_init_indices[class_index]
                                if ind not in well_described]

                # Compiling the different tables
                self.well_described[class_index][view_index] += well_described
                self.well_described[class_index][view_index] = np.array(
                    self.well_described[class_index][view_index])
                self.misdescribed[class_index][view_index] += misdescribed
                self.misdescribed[class_index][view_index] = np.array(
                    self.misdescribed[class_index][view_index])

    def _remove_available(self, available_indices, to_remove, class_index):
        """
        Removes indices from the available ones array
        """
        available_indices[class_index] = [ind
                                          for ind
                                          in available_indices[class_index]
                                          if ind not in to_remove]
        return available_indices

    def _get_repartitions(self, view_index, class_index):
        n_misdescribed = self.misdescribed[class_index][view_index].shape[0]
        updated_weights = np.array([weight if ind != class_index else 0
                                    for ind, weight
                                    in enumerate(self.class_weights)])
        updated_weights /= np.sum(updated_weights)
        repartitions = [int(n_misdescribed * updated_weights[class_ind])
                        for class_ind in range(self.n_classes)]
        if sum(repartitions) != n_misdescribed:
            avail_classes = [ind for ind in range(self.n_classes)
                             if ind != class_index]
            for ind in self.rs.choice(avail_classes,
                                      size=n_misdescribed - sum(repartitions),
                                      replace=False):
                repartitions[ind] += 1
        return repartitions

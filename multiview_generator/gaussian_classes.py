import numpy as np
import itertools
import math
from scipy.special import erfinv

from .utils import format_array, get_config_from_file, \
    init_random_state, init_error_matrix, init_list
from .base_strs import *
from .base import MultiViewSubProblemsGenerator
from multiview_generator import sub_problems


class MultiViewGaussianSubProblemsGenerator(MultiViewSubProblemsGenerator):

    def __init__(self, random_state=42, n_samples=100, n_classes=4, n_views=4,
                 error_matrix=None, n_features=3,
                 class_weights=1.0, redundancy=0.05, complementarity=0.05,
                 complementarity_level=3,
                 mutual_error=0.01, name="generated_dataset", config_file=None,
                 sub_problem_type="base", sub_problem_configurations=None,
                 sub_problem_generators="StumpsGenerator", random_vertices=False,
                 min_rndm_val=-1, max_rndm_val=1, **kwargs):
        """
        :param random_state: int or np.random.RandomState object to fix the
        random seed
        :param n_samples: int representing the number of samples in the dataset
        (the real number of samples can be different in the output dataset, as
        it will depend on the class distribution of the samples)
        :param n_classes: int the number of classes in the dataset
        :param n_views: int the number of views in the dataset
        :param error_matrix: the error matrix of size n_classes x n_views
        :param n_features: list of int containing the number fo features for
        each view
        :param class_weights: list of floats containing the proportion of
        samples in each class.
        :param redundancy: float controlling the ratio of redundant samples
        :param complementarity: float controlling the ratio of complementary
        samples
        :param complementarity_level: float controlling the ratio of views
        having a good description of the complementary samples.
        :param mutual_error: float controlling the ratio of complementary
        samples
        :param name: string naming the generated dataset
        :param config_file: string path pointing to a yaml config file
        :param sub_problem_type: list of string containing the class names for
        each sub problem type
        :param sub_problem_configurations: list of dict containing the specific
        configuration for each sub-problem generator
        :param kwargs: additional arguments
        """

        MultiViewSubProblemsGenerator.__init__(self, random_state=random_state,
                                               n_samples=n_samples,
                                               n_classes=n_classes,
                                               n_views=n_views,
                                               error_matrix=error_matrix,
                                               n_features=n_features,
                                               class_weights=class_weights,
                                               redundancy=redundancy,
                                               complementarity=complementarity,
                                               complementarity_level=complementarity_level,
                                               mutual_error=mutual_error,
                                               name=name,
                                               config_file=config_file,
                                               sub_problem_type=sub_problem_type,
                                               sub_problem_configurations=sub_problem_configurations,
                                               min_rndm_val=min_rndm_val,
                                               max_rndm_val=max_rndm_val,
                                               **kwargs)
        self.random_vertices = format_array(random_vertices, n_views, bool)
        self.sub_problem_generators = format_array(sub_problem_generators, n_views, str)

    def generate_multi_view_dataset(self, ):
        """
        This is the main method. It will generate a multiview dataset according to the configuration.
        To do so,

        * it generates the labels of the multiview dataset,
        * then it assigns all the subsets of samples (redundant, ...)
        * finally, for each view it generates a monoview dataset according to the configuration


        :return: view_data a list containing the views np.ndarrays and y, the label array.
        """

        # Generate the labels
        self.error_2D = np.ones((self.n_samples, self.n_views))
        # Generate the sample descriptions according to the error matrix
        self._sub_problem_generators = [_ for _ in range(self.n_views)]
        for view_index in range(self.n_views):
            sub_problem_generator = getattr(sub_problems,
                                            self.sub_problem_generators[view_index])(
                n_classes=self.n_classes,
                n_features=self.n_features[view_index],
                random_vertices=self.random_vertices[view_index],
                errors=self.error_matrix[:,view_index],
                random_state=self.rs,
                n_samples_per_class=self.n_samples_per_class,
                **self.sub_problem_configurations[view_index])
            vec = sub_problem_generator.gen_data()
            self._sub_problem_generators[view_index] = sub_problem_generator
            self.view_names[view_index] = "view_{}_{}".format(view_index, sub_problem_generator.view_name)
            self.bayes_error[view_index, :] = sub_problem_generator.bayes_error/self.n_samples_per_class
            self.generated_data[view_index, :, :,:self.n_features[view_index]] = vec
            self.selected_vertices[view_index] = sub_problem_generator.selected_vertices
            self.descriptions[view_index, :,:] = sub_problem_generator.descriptions

        self.y = []
        for ind, n_samples_ in enumerate(self.n_samples_per_class):
            self.y += [ind for _ in range(n_samples_)]
        self.y = np.array(self.y, dtype=int)


        self.sample_ids = ["{}_l_{}".format(ind, self.y[ind]) for ind in
                           range(self.n_samples)]

        self.dataset = [np.zeros((self.n_total_samples,
                                  self.n_features[view_index]))
                        for view_index in range(self.n_views)]

        self.assign_mutual_error()
        self.assign_complementarity()
        self.assign_redundancy()

        self.get_distance()
        return self.dataset, self.y

    def assign_mutual_error(self):
        """
        Method assigning the mis-describing views to the mutual error samples.
        """
        for class_ind in range(self.n_classes):
            mutual_start = np.sum(self.n_samples_per_class[:class_ind])
            mutual_end = np.sum(self.n_samples_per_class[:class_ind])+self.mutual_error_per_class[class_ind]
            for view_index in range(self.n_views):
                if len(np.where(self.descriptions[view_index, class_ind, :]==-1)[0])<self.mutual_error_per_class[class_ind]:
                    raise ValueError('For class {}, view {}, the amount of '
                                     'available mis-described samples is {}, '
                                     'and for mutual error to be assigned MAGE '
                                     'needs {}, please reduce the amount of '
                                     'mutual error or increase the error in '
                                     'class {}, view {}'.format(class_ind,
                                                                view_index,
                                                                len(np.where(self.descriptions[view_index, class_ind, :]==-1)[0]),
                                                                self.mutual_error_per_class[class_ind],
                                                                class_ind,
                                                                view_index))
                mis_described_random_ind = self.rs.choice(np.where(self.descriptions[view_index, class_ind, :]==-1)[0], self.mutual_error_per_class[class_ind], replace=False)
                self.dataset[view_index][mutual_start:mutual_end, :] = self.generated_data[view_index, class_ind, mis_described_random_ind, :self.n_features[view_index]]
                self.error_2D[mutual_start:mutual_end, view_index] = 0
                self.descriptions[view_index, class_ind, mis_described_random_ind] = 0
            for sample_ind in np.arange(start=mutual_start, stop=mutual_end):
                self.sample_ids[sample_ind] = self.sample_ids[sample_ind]+"_m"

    def assign_complementarity(self):
        """
        Method assigning mis-described and well-described views to build
        complementary samples
        """
        self.complementarity_ratio = 0
        for class_ind in range(self.n_classes):
            complem_level = int(self.complementarity_level[class_ind].item())
            complem_start = np.sum(self.n_samples_per_class[:class_ind])+self.mutual_error_per_class[class_ind]
            complem_ind = 0
            while complem_level != 0:
                avail_errors = np.array([len(np.where(self.descriptions[view_index, class_ind, :] ==-1)[0]) for view_index in range(self.n_views)])
                avail_success = np.array([len(np.where(self.descriptions[view_index, class_ind, :] == 1)[0]) for view_index in range(self.n_views)])

                cond=True

                while cond:
                    if np.sum(avail_errors) == 0 or np.sum(avail_success) < self.n_views - complem_level:
                        cond = False
                        break
                    elif len(np.where(avail_errors > 0)[0]) < complem_level:
                        cond = False
                        break
                    self.sample_ids[complem_start+complem_ind] += "_c"
                    self.complementarity_ratio += 1/self.n_samples
                    sorted_inds = np.argsort(-avail_errors)
                    selected_failed_views = sorted_inds[:complem_level]
                    sorted_inds = np.array([i for i in np.argsort(-avail_success) if
                                            i not in selected_failed_views])
                    selected_succeeded_views = sorted_inds[
                                               :self.n_views - complem_level]
                    for view_index in range(self.n_views):
                        if view_index in selected_failed_views:
                            self.error_2D[complem_start+complem_ind, view_index] = 0
                            chosen_ind = int(self.rs.choice(np.where(self.descriptions[view_index, class_ind, :]==-1)[0],size=1, replace=False)[0])
                            self.dataset[view_index][complem_start+complem_ind, :] = self.generated_data[view_index, class_ind, chosen_ind, :self.n_features[view_index]]
                            self.descriptions[view_index, class_ind, chosen_ind] = 0
                            self.sample_ids[complem_start+complem_ind] += "_{}".format(view_index)
                            avail_errors[view_index]-=1
                        elif view_index in selected_succeeded_views:
                            chosen_ind = int(self.rs.choice(np.where(self.descriptions[view_index, class_ind, :]==1)[0],size=1, replace=False)[0])
                            self.dataset[view_index][complem_start + complem_ind,:] = self.generated_data[view_index, class_ind, chosen_ind, :self.n_features[view_index]]
                            self.descriptions[view_index, class_ind, chosen_ind] = 0
                            avail_success[view_index] -= 1
                    complem_ind += 1
                complem_level -= 1
            self.n_complem[class_ind] = complem_ind

    def assign_redundancy(self):
        """
        Method assigning the well-describing views to the redundant samples.
        """
        self.real_redundancy_level=0
        for class_ind in range(self.n_classes):
            redun_start = int(np.sum(self.n_samples_per_class[:class_ind])+self.mutual_error_per_class[class_ind]+self.n_complem[class_ind])
            redun_end = np.sum(self.n_samples_per_class[:class_ind+1])
            for view_index in range(self.n_views):
                if len(np.where(self.descriptions[view_index, class_ind, :] == 1)[0]) < redun_end - redun_start and len(np.where(self.descriptions[view_index, class_ind, :] == -1)[0])>0:
                    raise ValueError("For class {}, view {}, reduce the error "
                                     "(now: {}), or increase the complemetarity "
                                     "level (now: {}), there is not enough good "
                                     "descriptions with the current "
                                     "configuration".format(class_ind,
                                                            view_index,
                                                            self.error_matrix[class_ind,
                                                                              view_index],
                                                            self.complementarity_level[class_ind]))
                remaining_good_desc = np.where(self.descriptions[view_index, class_ind, :] == 1)[0]
                self.dataset[view_index][redun_start:redun_end,:] = self.generated_data[view_index, class_ind,remaining_good_desc, :self.n_features[view_index]]
                self.descriptions[view_index, class_ind, remaining_good_desc] = 0
            for sample_ind in np.arange(start=redun_start, stop=redun_end):
                self.sample_ids[sample_ind] = self.sample_ids[sample_ind] + "_r"
                self.real_redundancy_level+=1/self.n_samples

    def get_distance(self):
        """
        Method that records the distance of each description to the ideal
        decision limit, will be used later to quantify more precisely the
        quality of a description.
        """
        self.distances = np.zeros((self.n_views, self.n_samples))
        for view_index, view_data in enumerate(self.dataset):
            for sample_ind, data in enumerate(view_data):
                # The closest dimension to the limit
                dist = np.min(np.abs(data))
                # dist = np.linalg.norm(data-self.selected_vertices[view_index][self.y[sample_ind]])
                self.sample_ids[sample_ind] += "-{}_{}".format(view_index, round(dist, 2))
                self.distances[view_index,sample_ind] = dist

    def _get_generator_report(self, view_index, doc_type=".md"):
        return "home made gaussian generator"

    def _init_sub_problem_config(self, sub_problem_configs, sub_problem_type):
        """
        Initialize the sub problem configurations.

        :param sub_problem_configs:
        :param sub_problem_type:
        :return:
        """

        if sub_problem_configs is None:
            self.sub_problem_configurations = [
                {"n_clusters_per_class": 1,
                 "class_sep": 1.0, }
                for _ in range(self.n_views)]
        else:
            self.sub_problem_configurations = init_list(sub_problem_configs,
                                                        size=self.n_views,
                                                        type_needed=dict)

    def _init_base_arguments(self):
        self.n_samples_per_class = (
                self.class_weights * self.n_samples).astype(int)
        self.n_max_samples = np.max(self.n_samples_per_class)
        self.n_samples = np.sum(self.n_samples_per_class)
        self.n_complem  =np.zeros(self.n_classes)
        self.n_max_features = np.max(self.n_features)
        self.generated_data = self.rs.uniform(low=-self.min_rndm_val,
                                              high=self.max_rndm_val,
                                              size=(self.n_views, self.n_classes,
                                                    self.n_max_samples,
                                                    self.n_max_features))
        self.descriptions = np.zeros((self.n_views, self.n_classes,
                                      self.n_max_samples,))
        self.n_total_samples = np.sum(self.n_samples_per_class)
        sample_indices = np.arange(int(np.sum(self.n_samples_per_class)))
        self.rs.shuffle(sample_indices)
        self.class_sample_indices = [
            sample_indices[sum(self.n_samples_per_class[:ind]):
                            sum(self.n_samples_per_class[:ind + 1])]
            for ind in range(self.n_classes)]
        self.well_described = [[_ for _ in range(self.n_views)] for _ in
                               range(self.n_classes)]
        self.misdescribed = [[_ for _ in range(self.n_views)] for _ in
                             range(self.n_classes)]
        self.redundancy_indices = [_ for _ in range(self.n_classes)]
        self.mutual_error_indices = [_ for _ in range(self.n_classes)]
        self.complementarity_samples = [_ for _ in range(self.n_classes)]
        self.good_views_indices = [_ for _ in range(self.n_classes)]
        self.bad_views_indices = [_ for _ in range(self.n_classes)]
        self.available_init_indices = self.class_sample_indices.copy()
        self.sample_ids = ["sample_{}".format(ind)
                            for ind
                            in range(int(np.sum(self.n_samples_per_class)))]
        self.bayes_error = np.zeros((self.n_views, self.n_classes))
        self.sub_problems = [[] for _ in range(self.n_views)]
        self.mutual_error_per_class = np.array(
            [int(float(self.mutual_error[class_ind].item()) * n_sample_) for class_ind, n_sample_ in
             enumerate(self.n_samples_per_class)])
        self.redundancy_per_class = np.array(
            [int(self.redundancy[class_ind].item() * n_sample_) for class_ind, n_sample_ in enumerate(self.n_samples_per_class)])
        self.view_data = [np.zeros((self.n_samples, self.n_features[view_ind])) for view_ind in range(self.n_views)]
        self.all_mis_described = [[] for _ in range(self.n_views)]
        self.all_well_described = [[] for _ in range(self.n_views)]
        self.selected_vertices = [_ for _ in range(self.n_views)]
        self.avail_well_described = [[] for _ in range(self.n_views)]
        self.avail_mis_described = [[] for _ in range(self.n_views)]
        self.mutual_error_indices = [[] for _ in range(self.n_views)]
        self.redundancy_indices = [[] for _ in range(self.n_views)]
        self.complementarity_indices = [[[] for _ in range(self.n_classes)] for _
                                   in
                                   range(self.n_views)]
        self.complem_names = [[] for _ in range(self.n_classes)]
        self.complem_error = [[] for _ in range(self.n_classes)]
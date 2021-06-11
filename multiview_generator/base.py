import os

import h5py
import numpy as np
import yaml
from sklearn.datasets import make_classification, make_gaussian_quantiles
from tabulate import tabulate
import pandas as pd
import inspect
from datetime import datetime
import plotly
import math
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier

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
    :param n_features: The number of features describing the samples for each view (can specify an int or array-like of length ``n_views``)
    :param class_weights: The proportion of the dataset that will be labelled in each class. Must specify an array-like of size n_classes ([0.1,0.45,0.45] will output a dataset with with 10% of the samples in the first class and 45% in the two others.)
    :param redundancy: The proportion of the samples that will be well-decribed by all the views.
    # :param complementarity: The proportion of samples that will be well-decribed only by some views
    :param complementarity_level: The number of views that will have a bad description of the complementray samples
    :param mutual_error: The proportion of samples that will be mis-described by all the views
    :param name: The name of the dataset (will be used to name the file)
    :param config_file: The path to the yaml config file. If provided, the config fil entries will overwrite the one passed as arguments.

    :type random_state: int or np.random.RandomState
    :type n_samples: int
    :type n_classes: int
    :type n_views: int
    :type error_matrix: np.ndarray
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
                 error_matrix=None, n_features=3,
                 class_weights=1.0, redundancy=0.0, complementarity=0.0,
                 complementarity_level=3,
                 mutual_error=0.0, name="generated_dataset", config_file=None,
                 sub_problem_type="base", sub_problem_configurations=None,
                 min_rndm_val=-1, max_rndm_val=1, **kwargs):

        if config_file is not None:
            args = get_config_from_file(config_file)
            self.__init__(**args)
        else:
            self.view_names = ["generated_view_{}".format(view_index) for view_index in range(n_views)]
            self.rs = init_random_state(random_state)
            self.n_samples = n_samples
            self.n_classes = n_classes
            self.n_views = n_views
            self.name = name
            self.min_rndm_val = min_rndm_val
            self.max_rndm_val = max_rndm_val
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
            self.complementarity_level = format_array(complementarity_level, n_classes, type_needed=int).reshape(((n_classes, 1)))
            self._init_sub_problem_config(sub_problem_configurations,
                                          sub_problem_type)
            self.error_matrix = init_error_matrix(error_matrix, n_classes,
                                                  n_views)
            self.classes = np.arange(self.n_classes)
            self.class_weights = format_array(class_weights, n_classes,
                                              type_needed=float)
            self.class_weights /= np.sum(self.class_weights)
            self._init_base_arguments()

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

        for view_index, data in enumerate(self.dataset):
            df_dataset = dataset_file.create_dataset("View" + str(view_index),
                                                     shape=data.shape,
                                                     data=data)

            df_dataset.attrs["sparse"] = False
            df_dataset.attrs["name"] = self.view_names[view_index]

        meta_data_grp = dataset_file.create_group("Metadata")

        meta_data_grp.attrs["nbView"] = self.n_views
        meta_data_grp.attrs["nbClass"] = np.unique(self.y)
        meta_data_grp.attrs["datasetLength"] = self.dataset[0].shape[0]

        self.gen_report(save=False)
        meta_data_grp.attrs["description"] = self.report

        meta_data_grp.create_dataset("sample_ids", data=np.array(
            self.sample_ids).astype(
            np.dtype("S100")), dtype=np.dtype("S100"))

        dataset_file.close()

    def gen_report(self, output_path='.', file_type="md", save=True, n_cv=5):
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
                        "and is comprised of \n\n* {} samples, splitted in " \
                        "\n* {} classes, described by \n* {} views.\n\n".format(self.name, GENE, LINK, self.n_samples, self.n_classes, self.n_views)

        error_df = pd.DataFrame(self.error_matrix,
                                index=["Class "+str(i+1)
                                       for i in range(self.n_classes)],
                                columns=['View '+str(i+1) for i in range(self.n_views)])

        report_string += "The input error matrix is \n \n"+tabulate(error_df,
                                                                 headers='keys',
                                                                 tablefmt='github')

        report_string += "\n\n The classes are balanced as : \n\n* Class "
        report_string += '\n* Class '.join(["{} : {} samples ({}% of the dataset)".format(i+1,
                                                                                         n_ex,
                                                                                         int(ratio*100))
                                          for i, (n_ex, ratio)
                                          in enumerate(zip(self.n_samples_per_class,
                                                           self.class_weights))])

        report_string += "\n\n The views have \n\n* {}% redundancy, \n* {}% mutual error" \
                         " and \n* {}% complementarity with a level of {}.\n\n".format(round(self.real_redundancy_level*100, 2), self.mutual_error[0,0]*100, round(self.complementarity_ratio*100, 2), self.complementarity_level)

        report_string+="## Views description"
        for view_index in range(self.n_views):
            report_string += self.gen_view_report(view_index)

        report_string += "\n\n## Statistical analysis"

        bayes_error = pd.DataFrame(self.bayes_error,
                     columns=["Class " + str(i + 1)
                            for i in range(self.n_classes)],
                     index=['View ' + str(i + 1) for i in
                              range(self.n_views)])
        report_string += "\n\nBayes error matrix : \n\n"+tabulate(bayes_error, headers='keys',
                                                                 tablefmt='github')

        max_depth = math.ceil(math.log(self.n_classes, 2))
        report_string += "\n\n The error, as computed by the 'empirical bayes' classifier of each view : \n\n".format(max_depth)

        self._gen_dt_error_mat(n_cv)

        dt_error = pd.DataFrame(np.transpose(self.dt_error),
                                columns=["Class " + str(i + 1)
                                       for i in range(self.n_classes)],
                                index=['View ' + str(i + 1) for i in
                                         range(self.n_views)])

        report_string += tabulate(dt_error, headers='keys', tablefmt='github')

        if save:
            self._plot_2d_error(output_path, error=self.error_2D, file_name="report_bayesian_error_2D.html")
            self._plot_2d_error(output_path, error=self.error_2D_dt, file_name="report_dt_error_2D.html")

        report_string += "\n\nThis report has been automatically generated on {}".format(datetime.now().strftime("%B %d, %Y at %H:%M:%S"))
        if save:
            with open(os.path.join(output_path, "report_"+self.name+"."+file_type), "w") as output:
                output.write(report_string)
        self.report = report_string
        return report_string

    def _plot_2d_error(self, output_path, error=None, file_name=""):
        label_index_list = np.concatenate([np.where(self.y == i)[0] for i in
                                           np.unique(
                                               self.y)])
        hover_text = [[self.sample_ids[sample_index] + " labelled " + str(
            self.y[sample_index])
                       for view_index in range(self.n_views)]
                      for sample_index in range(self.n_samples)]
        fig = plotly.graph_objs.Figure()
        fig.add_trace(plotly.graph_objs.Heatmap(
            x=["View {}".format(view_index) for view_index in range(self.n_views)],
            y=[self.sample_ids[label_ind] for label_ind in label_index_list],
            z=error[label_index_list, :],
            text=[hover_text[label_ind] for label_ind in label_index_list],
            hoverinfo=["y", "x", "text"],
            colorscale="Greys",
            colorbar=dict(tickvals=[0, 1],
                          ticktext=["Misdescribed", "Well described"]),
            reversescale=True), )
        fig.update_yaxes(title_text="Samples", showticklabels=True)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showticklabels=True, )
        plotly.offline.plot(fig, filename=os.path.join(output_path, self.name + file_name),
                            auto_open=False)

    def _gen_dt_error_mat(self, n_cv=10):
        # TODO : Seems to rely on random state, but unsure
        self.dt_error = np.zeros((self.n_classes, self.n_views))
        self.error_2D_dt = np.zeros((self.n_samples, self.n_views,))
        self.dt_preds = np.zeros((self.n_samples, self.n_views,))
        classifiers = [generator.get_bayes_classifier() for generator in self._sub_problem_generators]

        for view_index, view_data in enumerate(self.dataset):
            pred = cross_val_predict(classifiers[view_index], view_data, self.y, cv=n_cv, )
            self.dt_preds[:,view_index] = pred
            self.error_2D_dt[:, view_index] = np.equal(self.y, pred).astype(int)
            label_indices = [np.where(self.y == i)[0] for i in
                             range(self.n_classes)]
            loss = [zero_one_loss(pred[label_indice], self.y[label_indice]) for
                    label_indice in label_indices]
            self.dt_error[:, view_index] = np.array(loss)

    # def _find_rows_cols(self):
    #     rows=1
    #     cols=1
    #     if self.n_views == 4:
    #         rows = 2
    #         cols = 2
    #     if self.n_views>1:
    #         for i in range(self.n_views):
    #             if rows*cols < i+1:
    #                 if cols < 4*rows:
    #                     cols+=1
    #                 else:
    #                     rows+=1
    #     return rows, cols

    # def _get_pca(self, n_components=2, output_path='.'):
    #     pca = PCA(n_components=n_components)
    #     import plotly.graph_objects as go
    #     from plotly.subplots import make_subplots
    #     rows, cols = self._find_rows_cols()
    #     fig = make_subplots(rows=rows, cols=cols,
    #                         subplot_titles=["View{}".format(view_index)
    #                                         for view_index
    #                                         in range(self.n_views)],
    #                         specs=[[{'type': 'scatter'} for _ in range(cols) ]
    #                                for _ in range(rows)])
    #     row = 1
    #     col = 1
    #     import plotly.express as px
    #     for view_index, view_data in enumerate(self.dataset):
    #         if self.n_features[view_index]>n_components:
    #             pca.fit(view_data)
    #             reducted_data = pca.transform(view_data)
    #         elif self.n_features[view_index] ==1:
    #             reducted_data = np.transpose(np.array([view_data, view_data]))[0, :, :]
    #         else:
    #             reducted_data = view_data
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=reducted_data[:, 0],
    #                 y=reducted_data[:, 1],
    #                 text=self.sample_ids,
    #                 mode='markers', marker=dict(
    #                     size=3,  # set color to an array/list of desired values
    #                     color=self.y,
    #                     colorscale=["red", "blue", "black", "green", "orange", "purple"],
    #                     opacity=0.8
    #                 ), ),
    #             row=row, col=col)
    #         col += 1
    #         if col > cols:
    #             col = 1
    #             row += 1
    #     fig.update_shapes(dict(xref='x', yref='y'))
    #     plotly.offline.plot(fig, filename=os.path.join(output_path, self.name+"_fig_pca.html"), auto_open=False)

    def gen_view_report(self, view_index):
        view_string = "\n\n### View "+str(view_index+1)
        view_string+=self._sub_problem_generators[view_index].gen_report()
        return view_string

    # def _get_generator_report(self, view_index, doc_type=".md"):
    #     if self.sub_problem_types[view_index] in ["make_classification", "base"]:
    #         return "[`make_classification`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)"
    #     elif self.sub_problem_types[view_index]in ["gaussian", "make_gaussian_quantiles"]:
    #         return "[`make_gaussian_quantiles`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html#sklearn.datasets.make_gaussian_quantiles)"

    def _init_base_arguments(self):
        self.n_samples_per_class = (
                self.class_weights * self.n_samples).astype(int)
        self.n_well_described = [
            (self.n_samples_per_class[class_index] * (1 - confusion)).astype(
                int)
            for class_index, confusion in enumerate(self.error_matrix)]
        self.n_misdescribed = [(self.n_samples_per_class[class_index] -
                                self.n_well_described[class_index])
                               for class_index in range(self.n_classes)]
        self.n_samples = np.sum(self.n_samples_per_class)
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
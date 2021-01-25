import numpy as np
import itertools
import math
from scipy.special import erfinv
import yaml


class BaseSubProblem():

    def __init__(self, n_classes=2, n_features=2, random_vertices=True, errors=np.array([0.5,0.5]), random_state=np.random.RandomState(42), n_samples_per_class=np.array([100,100]), **configuration):
        self.n_classes = n_classes
        self.random_vertices = random_vertices
        self.errors = errors
        self.n_features = n_features
        self.rs = random_state
        self.n_samples_per_class = n_samples_per_class
        self.bayes_error = np.zeros(self.n_classes)
        self.descriptions = np.zeros((self.n_classes, np.max(self.n_samples_per_class)))
        self.config = configuration
        self.view_name = "generated"

    def gen_report(self):
        view_string = "\n\nThis view is generated with {}, with the following configuration : \n```yaml\n".format(
            self.__class__.__name__)
        view_string += yaml.dump(self.config,
                                 line_break="\n", default_flow_style=False)
        view_string += "n_features: {}\n".format(self.n_features)
        view_string += "```"
        return view_string


class StumpsGenerator(BaseSubProblem):

    def gen_data(self):
        """
        Generates the samples according to gaussian distributions with scales
        computed with the given error and class separation

        :param view_index:
        :return:
        """
        self.n_relevant_features = math.ceil(math.log2(self.n_classes))
        self.view_name = "stumps"
        class_sep = self.config["class_sep"]
        vertices = np.array(
            [np.array([coord for coord in coords]) for coords in
             itertools.product(
                 *zip([-1 for _ in range(self.n_relevant_features)],
                      [1 for _ in range(self.n_relevant_features)]))])

        if self.random_vertices == True:
            selected_vertices = self.rs.choice(np.arange(len(vertices)),
                                               self.n_classes,
                                               replace=False)
        else:
            selected_vertices = np.arange(self.n_classes)
        self.selected_vertices = vertices[selected_vertices,
                                 :] * class_sep
        vec = np.zeros((self.n_classes, max(self.n_samples_per_class),
                        self.n_relevant_features))
        for class_ind, center_coord in enumerate(
                self.selected_vertices):
            error = self.errors[class_ind]

            scale = (class_sep / math.sqrt(2)) * (1 / (
                erfinv(2 * (1 - error) ** (
                        1 / self.n_relevant_features) - 1)))
            cov = np.identity(self.n_relevant_features) * scale **2
            vec[class_ind, :,:] = self.rs.multivariate_normal(center_coord, cov,
                                              self.n_samples_per_class[
                                                  class_ind])
            mis_described = np.unique(
                np.where(np.multiply(vec[class_ind], center_coord) < 0)[0])
            well_described = np.array([ind for ind
                                       in range(
                    self.n_samples_per_class[class_ind])
                                       if ind not in mis_described])
            self.bayes_error[class_ind] = mis_described.shape[0]
            self.descriptions[class_ind, mis_described] = -1
            self.descriptions[class_ind, well_described] = 1
        data = self.rs.uniform(low=np.min(vec), high=np.max(vec), size=(self.n_classes, max(self.n_samples_per_class), self.n_features))
        data[:,:,:self.n_relevant_features] = vec
        return data

    def gen_report(self):
        base_str = BaseSubProblem.gen_report(self)
        base_str += "\n\nThis view has {} features, among which {} are relevant for classification (they are the {} first columns of the view) the other are filled with uniform noise.".format(
            self.n_features, self.n_relevant_features, self.n_relevant_features)
        base_str += "\n\n Its empirical bayesian classifier is a decision stump"
        return base_str

    def get_bayes_classifier(self):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=1)

class TreesGenerator(BaseSubProblem):
    """We stay with depth 2 trees ATM"""

    def gen_data(self):
        """
        Generates the samples according to gaussian distributions with scales
        computed with the given error and class separation

        :param view_index:
        :return:
        """
        self.n_relevant_features = math.ceil(math.log2(self.n_classes))
        self.view_name = "tree_depth_2"
        class_sep = self.config["class_sep"]
        vertices = np.array(
            [np.array([coord for coord in coords]) for coords in
             itertools.product(
                 *zip([-1 for _ in range(self.n_relevant_features)],
                      [1 for _ in range(self.n_relevant_features)]))])

        if self.random_vertices == True:
            selected_vertices = self.rs.choice(np.arange(len(vertices)),
                                               self.n_classes,
                                               replace=False)
        else:
            selected_vertices = np.arange(self.n_classes)
        self.selected_vertices = vertices[selected_vertices,
                                 :] * class_sep
        self.covs = np.zeros((self.n_classes, self.n_relevant_features, self.n_relevant_features))
        vec = np.zeros((self.n_classes, max(self.n_samples_per_class),
                        self.n_relevant_features))
        blob_centers = np.zeros((self.n_classes, self.n_relevant_features+1, self.n_relevant_features))
        for class_ind, center_coord in enumerate(
                self.selected_vertices):
            mis_described = []
            error = self.errors[class_ind]/(self.n_relevant_features+1)
            blob_centers[class_ind, 0, :] = center_coord
            internal_error_percentage = self.n_relevant_features*2/(self.n_relevant_features*2+self.n_relevant_features**2)
            internal_scale = (class_sep / math.sqrt(2)) * (1 / (
                erfinv(2 * (1 - error/internal_error_percentage) ** (
                        1 / (2*self.n_relevant_features)) - 1)))
            cov = np.identity(self.n_relevant_features) * internal_scale**2
            self.covs[class_ind] = cov
            n_samples = self.n_samples_per_class[class_ind] - (int(self.n_samples_per_class[class_ind]/(self.n_relevant_features+1)))*self.n_relevant_features
            vec[class_ind, :n_samples, :] = self.rs.multivariate_normal(center_coord, cov,
                                                                        n_samples)

            # mis_described += list(np.unique(np.where(
            #     np.any(abs(vec[class_ind] - center_coord)>class_sep, axis=1))[0]))
            # print(len(mis_described)*2/self.n_samples_per_class)
            n_samples_per_blob = int(self.n_samples_per_class[class_ind]/(self.n_relevant_features+1))
            external_error_percentage = self.n_relevant_features / (
                        self.n_relevant_features * 2 + self.n_relevant_features ** 2)
            external_scale = (class_sep / math.sqrt(2)) * (1 / (
                erfinv(2 * (1 - error / external_error_percentage) ** (
                        1 / self.n_relevant_features) - 1)))
            cov = np.identity(
                self.n_relevant_features) * external_scale**2
            # print(internal_scale, external_scale)
            for dim_index, update_coord in enumerate(center_coord):
                beg = n_samples+dim_index*n_samples_per_blob
                end = n_samples+(dim_index+1)*n_samples_per_blob
                new_center = center_coord.copy()
                new_center[dim_index] = update_coord-4*update_coord
                blob_centers[class_ind, dim_index+1, :] = new_center

                vec[class_ind, beg:end,:] = self.rs.multivariate_normal(new_center, cov,
                                                                            n_samples_per_blob)
                mis_described += list(np.unique(np.where(
                    np.any(abs(vec[class_ind, beg:end] - new_center)>class_sep, axis=1))[0])+beg)
            # mis_described = np.array(mis_described)
            # well_described = np.array([ind for ind
            #                            in range(
            #         self.n_samples_per_class[class_ind])
            #                            if ind not in mis_described])

            # self.bayes_error[class_ind] = mis_described.shape[0]
            # self.descriptions[class_ind, mis_described] = -1
            # self.descriptions[class_ind, well_described] = 1
        for class_ind in range(self.n_classes):
            for sample_ind in range(self.n_samples_per_class[class_ind]):
                if np.argmin(np.min(np.linalg.norm(vec[class_ind, sample_ind, :] - blob_centers, axis=2), axis=1))!= class_ind:
                    self.bayes_error[class_ind] +=1
                    self.descriptions[class_ind, sample_ind] = -1
                else:
                    self.descriptions[class_ind, sample_ind] = +1
        data = self.rs.uniform(low=-1, high=1, size=(
        self.n_classes, max(self.n_samples_per_class), self.n_features))
        data[:, :, :self.n_relevant_features] = vec
        return data

    def gen_report(self):
        base_str = BaseSubProblem.gen_report(self)
        base_str += "\n\nThis view has {} features, among which {} are relevant for classification (they are the {} first columns of the view).".format(self.n_features, self.n_relevant_features, self.n_relevant_features)
        base_str += "\n\n Its empirical bayesian classifier is a decision tree of depth 3"
        return base_str

    def get_bayes_classifier(self):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=2)

class RingsGenerator(BaseSubProblem):

    def gen_data(self):
        """
        Generates the samples according to gaussian distributions with scales
        computed with the given error and class separation

        :param view_index:
        :return:
        """
        if self.n_features<2:
            raise ValueError("n_features for view {} must be at least 2, (now: {})".format(1, self.n_features))
        self.view_name = "rings"
        data = np.zeros((self.n_classes, max(self.n_samples_per_class), self.n_features))
        class_sep = self.config["class_sep"]
        vertices = (np.arange(self.n_classes)+2)*class_sep

        if self.random_vertices == True:
            selected_vertices = self.rs.choice(np.arange(len(vertices)),
                                               self.n_classes,
                                               replace=False)
        else:
            selected_vertices = np.arange(self.n_classes)
        self.selected_vertices = vertices[selected_vertices]
        radii = np.zeros((self.n_classes, max(self.n_samples_per_class)))
        for class_ind, center_coord in enumerate(
                self.selected_vertices):
            error = self.errors[class_ind]
            scale = ((class_sep/2) / math.sqrt(2)) *  (1 /
                erfinv(1 - 2*error))
            radii[class_ind, :] = self.rs.normal(center_coord, scale,
                                                 self.n_samples_per_class[
                                                     class_ind])
            first_angle = self.rs.uniform(low=0, high=2*math.pi, size=(self.n_samples_per_class[class_ind],1))
            if self.n_features>2:
                other_angles = self.rs.uniform(low=0, high=1, size=(self.n_samples_per_class[class_ind], self.n_features-2))
                other_angles = np.arccos( 1 - 2 * other_angles)
                angles = np.concatenate((other_angles, first_angle), axis=1)
            else:
                angles = first_angle
            cartesian = np.array([to_cartesian(r, angle) for r, angle in zip(radii[class_ind], angles)])
            data[class_ind, :, :] = cartesian
            back_to_radii = np.sqrt(np.sum(np.square(cartesian), axis=1))
            if class_ind>1 and class_ind<self.n_classes-1:
                mis_described = np.unique(
                    np.where(np.logical_or(back_to_radii < vertices[class_ind]-(vertices[class_ind]-vertices[class_ind-1])/2, back_to_radii > vertices[class_ind]+(vertices[class_ind+1]-vertices[class_ind])/2))[0])
            elif class_ind==0:
                mis_described = np.unique(np.where(back_to_radii > vertices[class_ind]+(vertices[class_ind + 1] - vertices[class_ind]) / 2)[0])
            else:
                mis_described = np.unique(np.where(back_to_radii < vertices[class_ind]-(vertices[class_ind] - vertices[class_ind - 1]) / 2)[0])
            well_described = np.array([ind for ind
                                       in range(
                    self.n_samples_per_class[class_ind])
                                       if ind not in mis_described])
            self.bayes_error[class_ind] = mis_described.shape[0]
            self.descriptions[class_ind, mis_described] = -1
            self.descriptions[class_ind, well_described] = 1
        return data

    def gen_report(self):
        base_str = BaseSubProblem.gen_report(self)
        base_str += "\n\nThis view has {} features, all of them are relevant for classification.".format(
            self.n_features)
        base_str += "\n\n Its empirical bayesian classifier is any algorithm used with an RBF kernel."
        return base_str

    def get_bayes_classifier(self):
        from sklearn.svm import SVC
        return SVC(kernel='rbf', gamma=0.1, C=0.001)

def to_cartesian(radius, angles):
    a = np.concatenate((np.array([2 * np.pi]), angles))
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return si * co * radius
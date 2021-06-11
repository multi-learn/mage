import yaml
import numpy as np


def format_array(input, size, type_needed=int):
    """
    Used to test that :
    *  if the input is an array, it is the right size,
    * if it is either a string, or a saclar, build an array with ``input``
    repeated ``size`` times.

    :param input:  either a string, a scalar or an array-like
    :param size: an int, the size of the output array
    :return: a ``numpy.ndarray`` of shape (``size``, )
    """
    if isinstance(input, type_needed):
        if type_needed==str:
            return [input for _ in range(size)]
        else:
            return np.zeros(size, dtype=type_needed) + input
    elif isinstance(input, list) and isinstance(input[0], type_needed):
        if size == len(input):
            return np.asarray(input)
        else:
            raise ValueError("Input len did not match "
                             "size : {} != {}".format(len(input), size))
    elif isinstance(input, np.ndarray) and isinstance(input[0], type_needed):
        if size == input.shape[0]:
            return input
        else:
            raise ValueError("Input shape did not match "
                             "size : {} != {}".format(input.shape[0], size))
    else:
        raise ValueError("Must provide {} or array-like, "
                         "provided {}".format(type_needed, type(input)))


def get_config_from_file(file_path):
    """
    Loads the configuration for the yaml config file

    :param file_path: path to the config file.
    :return:
    """
    with open(file_path) as config_file:
        yaml_config = yaml.safe_load(config_file)
    return yaml_config


def init_class_weights(class_weights, n_classes):
    """
    Initializes the class weights. Sets a unifrom distribution if no
    distribution is specified.

    :param class_weights:
    :param n_classes:
    :return:
    """
    if class_weights is None:
        class_weights = np.ones(n_classes)
    return class_weights / np.sum(class_weights)


# def init_sub_problem_config(sub_problem_configs, n_views):
#     if sub_problem_configs is None:
#         return [{"n_informative":1,
#                  "n_redundant":1,
#                  "n_repeated":1,
#                  "n_clusters_per_class":1,
#                  "class_sep":1,} for _ in range(n_views)]


def init_error_matrix(error_matrix, n_classes, n_views):
    """
    Initializes the error matrix

    :param error_matrix:
    :param n_classes:
    :param n_views:
    :return:
    """
    if error_matrix is None:
        error_matrix = np.zeros((n_classes, n_views)) + 0.3
    elif isinstance(error_matrix, np.ndarray):
        if error_matrix.shape != (n_classes, n_views):
            raise ValueError("Confusion matrix must be of shape "
                             "(n_classes x n_views), here it is of shape {} "
                             "and n_classes={}, n_view={}".format(
                error_matrix.shape,
                n_classes,
                n_views))
        else:
            error_matrix = error_matrix
    elif isinstance(error_matrix, list):
        error_matrix = np.asarray(error_matrix)
    else:
        raise ValueError("Confusion matrix of wrong type : "
                         "{} instead of np.array, list or None".format(
            type(error_matrix)))
    return error_matrix


def init_random_state(random_state):
    """
    Initalizes the random state.

    :param random_state:
    :return:
    """
    if isinstance(random_state, int):
        rs = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        rs = random_state
    else:
        raise ValueError("Random state must be either en int or a "
                         "np.random.RandomState object, "
                         "here it is {}".format(random_state))
    return rs


def init_array_attr(attr, n_repeat, base_val=0):
    """
    Transforms a unique attribute into an array with the same value.

    :param attr:
    :param n_repeat:
    :param base_val:
    :return:
    """
    if attr is None:
        return np.ones((n_repeat, 1)) * base_val
    elif type(attr) == float or type(attr) == int:
        return np.ones((n_repeat, 1)) * attr
    elif isinstance(attr, np.ndarray):
        return attr.reshape((n_repeat, 1))
    else:
        raise ValueError("Wrong type for attr : {}".format(type(attr)))


def init_list(input, size, type_needed=dict):
    """
    Transforms a unique attribute into a list with the same value.

    :param attr:
    :param n_repeat:
    :param base_val:
    :return:
    """
    if isinstance(input, type_needed):
        return [input for _ in range(size)]
    elif isinstance(input, list):
        if len(input) == size:
            return input
        else:
            raise ValueError("Input is a list but is "
                             "not of the right length : input length = {},"
                             " expected legth = {]".format(
                len(input), size))
    else:
        raise ValueError("Input must be either a list "
                         "or a {}. here it is {}".format(type_needed,
            type(input)))


import unittest
import numpy as np

from ..multiple_sub_problems import MultiViewSubProblemsGenerator


class Test_MultiVieSubProblemsGenerator():

    def __init__(self):
        self.conf = np.array([
            np.array([0.0, 0.1, 0.1, 0.9]),
            np.array([0.0, 0.2, 0.1, 0.0]),
            np.array([0.0, 0.3, 0.1, 0.0]),
            np.array([0.0, 0.4, 0.2, 0.0]),
            np.array([0.0, 0.5, 0.2, 0.0]),
            np.array([0.0, 0.6, 0.2, 0.0]),
            np.array([0.0, 0.7, 0.2, 0.0]),
            np.array([0.0, 0.8, 0.1, 0.]),
        ])
        self.n_views = 4
        self.n_folds = 10
        self.n_classes = 8
        self.n_samples = 2000
        self.class_sep = 1.5
        self.class_weights = [0.125, 0.1, 0.15, 0.125, 0.01, 0.2, 0.125, 0.125, ]


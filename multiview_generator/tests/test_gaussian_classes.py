import unittest

from ..gaussian_classes import MultiViewGaussianSubProblemsGenerator

class Test_MultiViewGaussianSubProblemsGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_simple(self):
        gene = MultiViewGaussianSubProblemsGenerator()

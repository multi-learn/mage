import unittest
import numpy as np

from ..multiple_sub_problems import MultiViewSubProblemsGenerator


class Test_MultiViewSubProblemsGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_simple(self):
        gene = MultiViewSubProblemsGenerator()


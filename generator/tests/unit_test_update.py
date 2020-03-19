import unittest
import numpy as np

from ..update_baptiste import MultiviewDatasetGenetator

class TestSubSmaple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.random_state = np.random.RandomState(42)
        cls.indices = np.arange(100)
        cls.quantity = 10
        cls.method = "block"
        cls.beggining = 0
        cls.generator = MultiviewDatasetGenetator(random_state=cls.random_state)

    def test_block_simple(self):
        chosen_indices = self.generator.sub_sample(self.indices, self.quantity, self.method, self.beggining)
        np.testing.assert_array_equal(np.array([0,1,2,3,4,5,6,7,8,9]), chosen_indices)

    def test_block_too_big(self):
        chosen_indices = self.generator.sub_sample(self.indices, 121,
                                                   self.method, self.beggining)
        np.testing.assert_array_equal(np.arange(100),
                                      chosen_indices)

    def test_block_no_beg(self):
        chosen_indices = self.generator.sub_sample(self.indices, 10,
                                                   self.method, None)
        np.testing.assert_array_equal(np.array([82, 83, 84, 85, 86, 87, 88, 89, 90, 91,]),
                                      chosen_indices)

    def test_block_no_beg_too_long(self):
        chosen_indices = self.generator.sub_sample(self.indices, 120,
                                                   self.method, None)
        np.testing.assert_array_equal(np.arange(100),
                                      chosen_indices)
    def test_choice_simple(self):
        chosen_indices = self.generator.sub_sample(self.indices, 10,
                                                   "choice")
        np.testing.assert_array_equal(np.array([77, 10,  4, 83, 62, 67, 30, 45, 95, 11]),
                                      chosen_indices)

    def test_choice_too_big(self):
        chosen_indices = self.generator.sub_sample(self.indices, 105,
                                                   "choice")
        self.assertEqual(100, chosen_indices.shape[0])
        self.assertEqual(100, np.unique(chosen_indices).shape[0])



if __name__ == '__main__':
    unittest.main()

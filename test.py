from reach import Reach
from src.cat_aspect_extraction import CAt
import numpy as np
import unittest

class TestCat(unittest.TestCase):

    def setUp(self) -> None:
        mtr = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        words = ['cat', 'dog', 'bird', 'fish', 'mouse', 'elephant', 'tiger', 'lion']

        self.r = Reach(mtr, words)
        

    def test_init_candidate_aspects(self):
        cat = CAt(self.r)
        cat.init_candidate(['cat', 'dog', 'bird'])

        self.assertTrue((cat.candidates == np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0]
        ])).all())

    def test_add_topic(self):
        cat = CAt(self.r)
        cat.add_topic('felin', ['cat', 'tiger', 'lion'])
        cat.add_topic('canine', ['dog'])

        cat.topics = ['felin', 'canine']

        self.assertTrue((cat.topics_matrix.round(1) == np.array([
            [.6, 0, 0, 0, 0, 0, .6, .6],
            [0, 1, 0, 0, 0, 0, 0, 0]
        ])).all())

    def test_compute(self):
        cat = CAt(self.r)
        cat.init_candidate(['cat', 'tiger', 'lion'])
        cat.add_topic('felin', ['cat', 'tiger', 'lion'])
        cat.add_topic('canine', ['dog'])
        scores = cat.compute(['cat', 'dog', 'bird', 'fish', 'mouse', 'elephant', 'tiger', 'lion'])
        assert scores[0][0] == 'felin'

    def test_compute_oov(self):
        cat = CAt(self.r)
        cat.init_candidate(['cat', 'tiger', 'lion'])
        cat.add_topic('felin', ['cat', 'tiger', 'lion'])
        cat.add_topic('canine', ['dog'])
        scores = cat.compute(['cat', 'dog', 'bird', 'fish', 'mouse', 'elephant', 'tiger', 'lion', "horse"])
        assert scores[0][0] == 'felin'

if __name__ == '__main__':
    unittest.main()
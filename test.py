from reach import Reach
from src.cat_aspect_extraction import CAt
import numpy as np
import unittest

class TestCat(unittest.TestCase):

    def setUp(self) -> None:
        mtr = np.array([
            [2, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 8]
        ])

        words = ['cat', 'dog', 'bird', 'fish', 'mouse', 'elephant', 'tiger', 'lion']

        self.r = Reach(mtr, words)
        

    def test_init_candidate(self):
        cat = CAt(self.r)
        cat.init_candidate(['cat', 'dog', 'bird'])

        self.assertTrue((cat.candidates == np.array([
            [2, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0]
        ])).all())

    def test_add_topic(self):
        cat = CAt(self.r)
        cat.add_topic('felin', ['cat', 'tiger', 'lion'])
        cat.add_topic('canine', ['dog'])

        cat.topics = ['felin', 'canine']

        print(cat.topics_matrix.round(1))

        self.assertTrue((cat.topics_matrix.round(1) == np.array([
            [.3, 0, 0, 0, 0, 0, .3, .3],
            [1, 0, 0, 0, 0, 0, 0, 0]
        ])).all())

    def test_compute(self):
        cat = CAt(self.r)
        cat.init_candidate(['tiger'])
        cat.add_topic('felin', ['cat', 'tiger', 'lion'])
        cat.add_topic('canine', ['dog'])
        cat.add_topic('bird', ['bird'])
        scores = cat.compute(['cat'])
        assert scores[0][0] == 'canine' # because cat and dog are same direction

    def test_compute_oov(self):
        cat = CAt(self.r)
        cat.init_candidate(['cat', 'tiger', 'lion', 'dog'])
        cat.add_topic('felin', ['cat', 'tiger', 'lion'])
        cat.add_topic('canine', ['dog'])
        scores = cat.compute(['cat', 'tiger', 'lion', 'dog', "horse"])
        assert scores[0][0] == 'felin'

if __name__ == '__main__':
    unittest.main()
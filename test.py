from reach import Reach
from src.cat_aspect_extraction import CAt
from src.cat_aspect_extraction.attention import RBFAttention, CosineVarianceAttention, SoftmaxAttention, MeanAttention
import numpy as np
import unittest

class TestCat(unittest.TestCase):

    def setUp(self) -> None:
        mtr = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 8]
        ])

        words = ['cat', 'dog', 'bird', 'fish', 'mouse', 'elephant', 'tiger', 'lion']

        self.r = Reach(mtr, words)
        

    def test_add_candidate(self):
        cat = CAt(self.r)
        cat.add_candidate('cat')
        cat.add_candidate('dog')
        cat.add_candidate('bird')

        self.assertTrue((cat.candidates_matrix == np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0]
        ])).all())

    def test_add_topic(self):
        cat = CAt(self.r)
        cat.add_topic('felin', ['cat', 'tiger', 'lion'])
        cat.add_topic('canine', ['dog'])

        self.assertTrue((cat.topics_matrix.round(1) == np.array([
            [.1, 0, 0, 0, 0, 0, .7, .7],
            [0, 1, 0, 0, 0, 0, 0, 0]
        ])).all())

    def test_get_scores(self):
        cat = CAt(self.r)
        cat.add_candidate('tiger')
        cat.add_topic('felin', ['cat', 'tiger', 'lion'])
        cat.add_topic('canine', ['dog'])
        cat.add_topic('bird', ['bird'])
        scores = cat.get_scores(['cat','felin','tiger'], attention_func=RBFAttention())
        assert scores[0][0] == 'felin'
    
    def test_rbf_attention(self):
        candidates = [self.r['cat'], self.r['dog']]
        vectors = [self.r['cat'], self.r['tiger'], self.r['lion']]
        attention = RBFAttention()
        scores = attention.attention(vectors, candidates)
        assert scores.shape == (1, 3)

    def test_cosine_variance_attention(self):
        candidates = [self.r['cat'], self.r['dog']]
        vectors = [self.r['cat'], self.r['tiger'], self.r['lion']]
        attention = CosineVarianceAttention()
        scores = attention.attention(vectors, candidates)
        assert scores.shape == (1, 3)
    
    def test_softmax_attention(self):
        candidates = np.array([self.r['cat'], self.r['dog']])
        vectors = np.array([self.r['cat'], self.r['tiger'], self.r['lion']])
        attention = SoftmaxAttention()
        scores = attention.attention(vectors, candidates)
        assert scores.shape == (2, 3)

    def test_mean_attention(self):
        vectors = np.array([self.r['cat'], self.r['tiger'], self.r['lion']])
        attention = MeanAttention()
        scores = attention.attention(vectors)
        assert scores.shape == (1, 3)

if __name__ == '__main__':
    unittest.main()
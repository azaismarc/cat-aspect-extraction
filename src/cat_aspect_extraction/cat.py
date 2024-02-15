import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from reach import Reach
from collections import Counter

class CAt():
    """
    Implementation of Contrastive Attention Topic Modeling describe in "Embarrassingly Simple Unsupervised Aspect Extraction"
    
    URL : https://aclanthology.org/2020.acl-main.290

    Calculating attribution topic scores for a list of tokens.
    Scores are computed using an approach based on RBF (Radial Basis Function) similarity functions between tokens and candidate aspects, 
    and then using attention to aggregate scores of topics associated with candidate aspects.
    """

    def __init__(self, r: Reach, rbf: bool = True, gamma: float = .03,) -> None:
        """
        Parameters:
        -----------
        - r (Reach) : A reach instance for vectorization
        - rbf (bool) : Whether to use RBF similarity function (True) or cosine similarity (False) (default True)
        - gamma (float) : Gamma parameter of RBF similarity function (default 0.03)
        """
        self.r = r
        self.gamma = gamma
        self.rbf = rbf
        self.candidates = None
        self.topics = []
        self.topics_matrix = None

    def init_candidate(self, aspects: list[str]) -> None:
        """
        Initialize candidate words for aspect extraction

        Parameters:
        -----------
        - aspects (list[str]) : List of aspects as candidates
        """
        self.candidates = np.array([self.r[a] for a in aspects])
    
    def add_topic(self, topic: str, aspects: list[str]) -> None:
        """
        Add topic and compute its vector based on its composition (mean vector of multiple words)

        Parameters:
        -----------
        - topic (str) : Name of topic
        - aspects (list[str]) : List of aspects that compose the topic
        """

        self.topics.append(topic)
        topic_vector = normalize(np.mean([self.r[a] for a in aspects], axis=0).reshape(1, -1))
        if self.topics_matrix is None: self.topics_matrix = topic_vector
        else: self.topics_matrix = np.vstack((self.topics_matrix, topic_vector.squeeze()))

    def attention(self, matrix: np.array) -> np.ndarray:
        """
        Compute attention vector for a given list of tokens as vector

        Parameters:
        -----------
        - matrix (np.ndarray) : Matrix of tokens as vector

        Returns:
        --------
        - np.ndarray : Attention vector
        """

        z = rbf_kernel(matrix, self.candidates, gamma=self.gamma) if self.rbf else cosine_similarity(matrix, self.candidates)
        s = z.sum()
        if s == 0: return np.ones((1, matrix.shape[0])) / matrix.shape[0]
        return (z.sum(axis=1) / s).reshape(1, -1)
    
    def compute(self, tokens: list[str]) -> list[(str,float)]:
        """
        Compute the score of each topics

        Parameters:
        -----------
        - tokens (list[str]) : A list of tokens for which to compute scores.

        Returns:
        --------
        - list(tuple(str, float)) : A list of tuples containing labels and their associated scores, 
          sorted in descending order of score.
        """

        assert self.candidates is not None, "No candidate aspects have been initialized"
        assert len(self.topics) > 0, "No labels have been added"

        score = Counter({topic: 0 for topic in self.topics})
        if len(tokens) == 0: return score.most_common() # No tokens to process
        tokens_matrix = np.array([self.r[t] for t in tokens if t in self.r.items])
        if len(tokens_matrix) == 0: return score.most_common() # No tokens to process

        att = self.attention(tokens_matrix)
        z = att.dot(tokens_matrix)
        x = normalize(z).dot(self.topics_matrix.T)
        scores = x.sum(axis=0)

        for i, topic in enumerate(self.topics):
            score[topic] = scores[i]
        return score.most_common()
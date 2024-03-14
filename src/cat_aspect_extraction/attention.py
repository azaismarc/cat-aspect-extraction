import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity, euclidean_distances

from abc import ABC, abstractmethod

class Attention(ABC):

    def super_attention(self, z: np.array, n: int) -> np.ndarray:
        s = z.sum()
        if s == 0: return np.ones((1, n)) / n
        return (z.sum(axis=1) / s).reshape(1, -1)

    @abstractmethod
    def attention(self, vectors: np.array, candidates: np.array) -> np.ndarray:
        """
        Compute attention vector for a given list of tokens as vector

        Parameters:
        -----------
        - matrix (np.ndarray) : Matrix of tokens as vector (shape: (n, d))
        - candidates (np.ndarray) : Matrix of candidate words as vector (shape: (m, d))

        Returns:
        --------
        - np.ndarray : Attention vector (shape: (h, n))
        """
        pass

class RBFAttention(Attention):

    def __init__(self, gamma: float = .03) -> None:
        """
        Parameters:
        -----------
        - gamma (float) : Gamma parameter for RBF kernel (default 0.03)
        """
        self.gamma = gamma

    def attention(self, vectors: np.array, candidates: np.array) -> np.ndarray:
        z = rbf_kernel(vectors, candidates, gamma=self.gamma)
        return self.super_attention(z, len(vectors))

class CosineVarianceAttention(Attention):

    def attention(self, vectors: np.array, candidates: np.array) -> np.ndarray:
        z = cosine_similarity(vectors, candidates)
        var_z = np.var(z, axis=1) # Compute variance of cosine similarity
        s = var_z.sum()
        if s == 0: return np.ones((1, len(vectors))) / len(vectors)
        return (var_z / var_z.sum()).reshape(1, -1)
    

class CosineAttention(Attention):

    def attention(self, vectors: np.array, candidates: np.array) -> np.ndarray:
        z = cosine_similarity(vectors, candidates)
        z = (1 - z) / 2 # Convert cosine similarity to distance
        z = z.clip(0, 1) # Clip values to be between 0 and 1
        z = 1 - z # Convert distance to similarity
        return self.super_attention(z, len(vectors))
    
class EuclideanAttention(Attention):

    def attention(self, vectors: np.array, candidates: np.array) -> np.ndarray:
        z = euclidean_distances(vectors, candidates)
        z = 1 - z / z.max() # convert distance to similarity
        return self.super_attention(z, len(vectors))
    
class SoftmaxAttention(Attention):

    def attention(self, vectors: np.array, candidates: np.array) -> np.ndarray:
        z = candidates.dot(vectors.T)
        z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)
    
class MeanAttention(Attention):

    def attention(self, vectors: np.array, candidates: np.array = None) -> np.ndarray:
        return (np.ones((1, len(vectors))) / len(vectors)).reshape(1, -1)
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity

from abc import ABC, abstractmethod

class Attention(ABC):

    def super_attention(self, z: np.array, n: int) -> np.ndarray:
        s = z.sum()
        if s == 0: return np.ones((1, len(n))) / len(n)
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
        - np.ndarray : Attention vector (shape: (1, n))
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
    

class CosineAttention(Attention):

    def attention(self, vectors: np.array, candidates: np.array) -> np.ndarray:
        z = cosine_similarity(vectors, candidates)
        return self.super_attention(z, len(vectors))
    
class EuclideanAttention(Attention):

    def attention(self, vectors: np.array, candidates: np.array) -> np.ndarray:
        z = np.linalg.norm(vectors - candidates, axis=1)
        z = 1 / (1 + z)
        return self.super_attention(z, len(vectors))
    
class SoftmaxAttention(Attention):

    def attention(self, vectors: np.array, candidates: np.array) -> np.ndarray:
        z = np.exp(candidates.dot(vectors.T))
        return self.super_attention(z, len(vectors))
    
class MeanAttention(Attention):

    def attention(self, vectors: np.array, candidates: np.array) -> np.ndarray:
        z = np.mean(vectors.dot(candidates.T), axis=1)
        return self.super_attention(z, len(vectors))
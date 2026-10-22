import numpy as np
from .base import DenseClassifier


class TanhNetwork(DenseClassifier):
    name='tanh'
    def activation(self,z):return np.tanh(z)
    def derivative(self,z):return 1-np.tanh(z)**2

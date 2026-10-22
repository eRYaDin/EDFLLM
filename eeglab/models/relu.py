import numpy as np
from .base import DenseClassifier


class ReLUNetwork(DenseClassifier):
    name='relu'
    def activation(self,z):return np.maximum(z,0)
    def derivative(self,z):return (z>0).astype(float)

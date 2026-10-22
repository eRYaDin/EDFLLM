import numpy as np
from .base import DenseClassifier


class SoftmaxNetwork(DenseClassifier):
    """Baseline klasifikasi dengan hidden ReLU dan output Softmax, bukan hidden Softmax."""
    name='softmax'
    def activation(self,z):return np.maximum(z,0)
    def derivative(self,z):return (z>0).astype(float)

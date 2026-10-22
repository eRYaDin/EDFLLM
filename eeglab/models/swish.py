from scipy.special import expit
from .base import DenseClassifier


class SwishNetwork(DenseClassifier):
    name='swish'
    def activation(self,z):return z*expit(z)
    def derivative(self,z):
        s=expit(z);return s+z*s*(1-s)

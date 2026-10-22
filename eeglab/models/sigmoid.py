from scipy.special import expit
from .base import DenseClassifier


class SigmoidNetwork(DenseClassifier):
    name='sigmoid'
    binary=True
    def activation(self,z):return expit(z)
    def derivative(self,z):
        a=expit(z);return a*(1-a)

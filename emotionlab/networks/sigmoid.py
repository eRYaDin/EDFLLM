from scipy.special import expit
from .base import Network

class SigmoidNetwork(Network):
    """Tujuh kepala biner one-vs-rest, satu kelas emosi melawan sisanya."""
    name='sigmoid'
    independent_sigmoid=True
    def activation(self,z):return expit(z)
    def derivative(self,z):
        a=expit(z);return a*(1-a)

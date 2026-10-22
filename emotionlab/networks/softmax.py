import numpy as np
from scipy.special import logsumexp
from .base import Network
class SoftmaxNetwork(Network):
    """Eksperimen aktivasi kompetitif hidden; output tetap Softmax multikelas.

    Bukan pilihan hidden standar. Jacobian-vector product penuh diperlukan.
    """
    name='softmax'
    def activation(self,z):return np.exp(z-logsumexp(z,axis=1,keepdims=True))
    def backward(self,z,gradient):
        a=self.activation(z);return a*(gradient-(gradient*a).sum(axis=1,keepdims=True))

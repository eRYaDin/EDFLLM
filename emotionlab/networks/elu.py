import numpy as np
from .base import Network
class ELUNetwork(Network):
    name='elu'
    def activation(self,z):return np.maximum(z,0)+np.expm1(np.minimum(z,0))
    def derivative(self,z):return np.where(z>0,1.,np.exp(np.minimum(z,0)))

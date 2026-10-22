import numpy as np
from .base import Network
class LeakyReLUNetwork(Network):
    name='leaky_relu'
    def activation(self,z):return np.where(z>0,z,.01*z)
    def derivative(self,z):return np.where(z>0,1.,.01)

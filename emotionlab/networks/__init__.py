from .sigmoid import SigmoidNetwork
from .relu import ReLUNetwork
from .softmax import SoftmaxNetwork
from .tanh import TanhNetwork
from .leaky_relu import LeakyReLUNetwork
from .elu import ELUNetwork
from .swish import SwishNetwork

NETWORKS={c.name:c for c in (SigmoidNetwork,ReLUNetwork,SoftmaxNetwork,TanhNetwork,LeakyReLUNetwork,ELUNetwork,SwishNetwork)}

def create(name,inputs,classes,hidden=(32,16,8)):
    if name not in NETWORKS:raise ValueError('Jaringan tidak dikenal: '+name)
    return NETWORKS[name](inputs,classes,hidden)

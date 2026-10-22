"""File pemanggil tujuh model; implementasi aktivasi berada di modul masing-masing."""
from .models.sigmoid import SigmoidNetwork
from .models.relu import ReLUNetwork
from .models.softmax import SoftmaxNetwork
from .models.tanh import TanhNetwork
from .models.leaky_relu import LeakyReLUNetwork
from .models.elu import ELUNetwork
from .models.swish import SwishNetwork

MODELS={model.name:model for model in (SigmoidNetwork,ReLUNetwork,SoftmaxNetwork,
                                      TanhNetwork,LeakyReLUNetwork,ELUNetwork,SwishNetwork)}


def create_model(name,n_features,n_classes,hidden=(32,16,8),seed=42):
    if name not in MODELS:raise ValueError('Model tidak dikenal: '+name)
    return MODELS[name](n_features,n_classes,hidden,seed)

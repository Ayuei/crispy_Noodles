import ABC
import Activation.

class loss(ABC.ABC):

    def __init__(self, activation):
        self.activation = 
        assert hasattr(activation, '__deriv__')
        assert hasattr(activation, '__activate__')


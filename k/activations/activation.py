from abc import ABC, abstractmethod

class Activation(ABC):

    @abstractmethod
    def __deriv__(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def __activate__(self, *args):
        raise NotImplementedError()

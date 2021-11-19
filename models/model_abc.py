import abc


class Model(abc.ABC):

    @abc.abstractmethod
    def run(self, data):
        print(data)
        return NotImplemented

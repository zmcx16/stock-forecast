import abc


class Model(abc.ABC):

    @abc.abstractmethod
    def run_validate(self, data):
        print(data)
        return NotImplemented

    @abc.abstractmethod
    def run_predict(self, data):
        print(data)
        return NotImplemented

from core.em_data import EMData

from abc import ABC, abstractmethod

class EMDataLoader(ABC):
    @staticmethod
    @abstractmethod
    def load(path: str) -> EMData:
        pass

class EMDataSaver(ABC):
    @staticmethod
    @abstractmethod
    def save(data: EMData, path: str):
        pass
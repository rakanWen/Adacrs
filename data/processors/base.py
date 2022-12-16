import abc
import os
from typing import Union


class Processor(abc.ABC):
    def __init__(self, data_dir: Union[str, os.PathLike]):
        self.data_dir = data_dir
        self.load_entities()
        self.load_relations()

    @abc.abstractmethod
    def get_relation(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_entities(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_relations(self):
        raise NotImplementedError


class Graph(abc.ABC):
    def __init__(self, dataset):
        self.G = dict()
        self._load_entities(dataset)
        self._load_knowledge(dataset)
        self._clean()

    @abc.abstractmethod
    def _load_entities(self, dataset):
        raise NotImplementedError

    @abc.abstractmethod
    def _load_knowledge(self, dataset):
        raise NotImplementedError

    @abc.abstractmethod
    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        raise NotImplementedError

    @abc.abstractmethod
    def _clean(self):
        raise NotImplementedError

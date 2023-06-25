from abc import abstractmethod
from typing import List, Union
import torch
from pipeline.data_obj.datapoint import DataPoint


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class BaseDetector(metaclass=SingletonMeta):
    """Abstract class using for implementing a Deep Learning Model"""
    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError


class PipelineComponent(metaclass=SingletonMeta):
    """Abstract Class using for implementing a Pipeline Component"""
    @abstractmethod
    def serve(self, dp: DataPoint) -> None:
        raise NotImplementedError


class PreProcessComponent(metaclass=SingletonMeta):
    @abstractmethod
    def serve(self, text: str) -> DataPoint:
        raise NotImplementedError


class PostProcessComponent(metaclass=SingletonMeta):
    @abstractmethod
    def serve(self, dp: DataPoint) -> any:
        raise NotImplementedError


class BasePipeline(metaclass=SingletonMeta):
    pipeline_components = []

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs) -> any:
        """
        Abstract method to build the pipeline.
        """
        raise NotImplementedError

    @abstractmethod
    def analyze(self, *args, **kwargs) -> any:
        """
        Abstract method for analyzing the pipeline.
        """
        raise NotImplementedError

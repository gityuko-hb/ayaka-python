from abc import ABC, abstractmethod

from ayaka.configs.device_config import DeviceConfig
from ayaka.configs.loader_config import LoaderConfig
from ayaka.configs.model_config import ModelConfig


class BaseModelLoader(ABC):
    def __init__(self, loader_config: LoaderConfig):
        self.loader_config = loader_config
    
    @abstractmethod
    def download_model(self, model_config: ModelConfig):
        raise NotImplementedError
    
    @abstractmethod
    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig
    ):
        raise NotImplementedError
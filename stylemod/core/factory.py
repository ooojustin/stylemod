import importlib
import pkgutil
from stylemod.core.base_model import BaseModel
from typing import Dict, Callable, Type


class ModelFactory:
    """Factory class responsible for creating and registering model instances."""

    model_mapping: Dict[str, Callable[..., BaseModel]] = {}

    @staticmethod
    def create(model_name: str) -> BaseModel:
        """Create and return an instance of the requested model."""
        try:
            model_name = model_name.lower()
            model_class = ModelFactory.model_mapping[model_name]
            model = model_class()
        except KeyError:
            raise ValueError(
                f"Model '{model_name}' not supported. Available models: {list(ModelFactory.model_mapping.keys())}")
        return model

    @staticmethod
    def register(model_name: str, model_class: Type[BaseModel]):
        """Dynamically register a new model to the factory."""
        model_name = model_name.lower()
        if model_name in ModelFactory.model_mapping:
            raise ValueError(f"Model '{model_name}' is already registered.")
        ModelFactory.model_mapping[model_name] = model_class

    @staticmethod
    def auto_register_models():
        """Automatically import all modules in the models directory and register them."""
        package = "stylemod.models"
        for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(package).__path__):
            importlib.import_module(f"{package}.{module_name}")


ModelFactory.auto_register_models()

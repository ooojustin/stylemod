import importlib
import pkgutil
import inspect
from stylemod.models import Model
from stylemod.core.base import BaseModel
from stylemod.core.cnn import CNNBaseModel
from stylemod.core.transformer import TransformerBaseModel
from typing import Dict, Callable, Type, Union


class ModelFactory:
    """Factory class responsible for creating and registering model instances."""

    model_mapping: Dict[str, Callable[..., BaseModel]] = {}

    @staticmethod
    def create(model: Union[str, Model]) -> BaseModel:
        """Create and return an instance of the requested model."""
        try:
            if isinstance(model, Model):
                model_class = model.value
            else:
                model_name = model.upper()
                model_class = ModelFactory.model_mapping[model_name]
            model_instance = model_class()
        except KeyError:
            raise ValueError(
                f"Model '{model}' not supported. Available models: {list(ModelFactory.model_mapping.keys())}"
            )
        return model_instance

    @staticmethod
    def register(model_name: str, model_class: Type[BaseModel]):
        """Dynamically register a new model to the factory."""
        model_name = model_name.upper()
        if model_name in ModelFactory.model_mapping:
            raise ValueError(f"Model '{model_name}' is already registered.")
        ModelFactory.model_mapping[model_name] = model_class

    @staticmethod
    def auto_register_models():
        """Automatically import all modules in the models directory and register them."""
        package = "stylemod.models"
        for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(package).__path__):
            module = importlib.import_module(f"{package}.{module_name}")

            # inspect the module for classes that inherit from BaseModel
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseModel) and obj not in [BaseModel, CNNBaseModel, TransformerBaseModel]:

                    model_enum_name = None
                    for enum_model in Model:
                        if enum_model.value == obj:
                            model_enum_name = enum_model.name
                            break

                    # use the enum name if found, otherwise use the detected class name
                    model_name = model_enum_name or name
                    ModelFactory.register(model_name, obj)


ModelFactory.auto_register_models()

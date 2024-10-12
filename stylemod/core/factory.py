import importlib
import pkgutil
import inspect
from stylemod.models import Model
from stylemod.core.base import BaseModel
from stylemod.core.cnn import CNNBaseModel
from stylemod.core.transformer import TransformerBaseModel
from typing import Dict, Callable, Type, Union


class ModelFactory:
    _models: Dict[str, Callable[..., BaseModel]] = {}

    @staticmethod
    def create(model: Union[str, Model]) -> BaseModel:
        try:
            if isinstance(model, Model):
                model_class = model.value
            else:
                model_name = model.upper()
                model_class = ModelFactory._models[model_name]
            model_instance = model_class()
        except KeyError:
            raise ValueError(
                f"Model '{model}' not supported. Available models: {list(ModelFactory._models.keys())}"
            )
        return model_instance

    @staticmethod
    def register(model_name: str, model_class: Type[BaseModel]):
        model_name = model_name.upper()
        if model_name in ModelFactory._models:
            raise ValueError(f"Model '{model_name}' is already registered.")
        ModelFactory._models[model_name] = model_class

    @staticmethod
    def _register_models():
        pkg = "stylemod.models"
        for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(pkg).__path__):
            module = importlib.import_module(f"{pkg}.{module_name}")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseModel) and obj not in [BaseModel, CNNBaseModel, TransformerBaseModel]:
                    model_enum_name = None
                    for enum_model in Model:
                        if enum_model.value == obj:
                            model_enum_name = enum_model.name
                            break
                    model_name = model_enum_name or name
                    ModelFactory.register(model_name, obj)


ModelFactory._register_models()

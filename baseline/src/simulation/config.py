import argparse
import ast
import json
import pprint
from enum import Enum, auto
from inspect import isclass
from pathlib import Path
from typing import Annotated, get_type_hints, Tuple
import platform
import abrain


class Config:
    simulation_time: Annotated[float, "Duration of the simulation"] = 20
    control_frequency: Annotated[float, "How frequently to call the controller"] = 10
    sampling_frequency : Annotated[float, "How frequently to sample/collect Environment States"] = 40
    ground_size: Annotated[float, "Total size of the arena"] = 10
    

    abrain = abrain.Config

    class OpenGLLib(str, Enum):
        GLFW = auto()
        EGL = auto()
        OSMESA = auto()

    if platform.system() == "Linux":
        opengl_lib: Annotated[str, "OpenGL back-end for vision"] = OpenGLLib.EGL.name
    elif platform.system() == "Windows":
        opengl_lib: Annotated[str, "OpenGL back-end for vision"] = OpenGLLib.GLFW.name
    else:
        print("Unknown operating system")    

    
    @classmethod
    def write_json(cls, file: Path):
        with open(file, 'w') as f:
            json.dump(Config._get_items(Config), f, indent=2)
            f.write('\n')

    @classmethod
    def read_json(cls, file: Path):
        with open(file, 'r') as f:
            Config._set_items(Config, json.load(f))

    @classmethod
    def print(cls):
        pprint.pprint(cls._items())

    @classmethod
    def argparse_setup(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Config", f"Alter configuration values (see {__file__})")
        hints = get_type_hints(cls, include_extras=True)
        for k, v in cls._get_items(cls).items():
            hint = hints.get(k)
            if hint is not None:
                k_type = hint.__args__[0]
                if k_type == bool:
                    k_type = ast.literal_eval
                elif isinstance(v, Enum):
                    k_type = lambda s, k_t=k_type: k_t[s]
                group.add_argument('--config-' + k, dest="config_" + k, metavar='V',
                                   type=k_type,
                                   help=f"{'.'.join(hint.__metadata__)}"
                                        f" (default: {v}, type: %(type)s)")

    @classmethod
    def argparse_process(cls, args):
        for k, v in filter(lambda i: i[0].startswith('config_'), args.__dict__.items()):
            if v is not None:
                setattr(cls, k[7:], v)

    @classmethod
    def _items(cls):
        return Config._get_items(Config)

    @staticmethod
    def _set_items(cls, dct):
        for k, v in dct.items():
            attr = getattr(cls, k, None)
            if not isclass(attr):
                setattr(cls, k, v)

    @staticmethod
    def _get_items(cls):
        return {k: v_ for k in dir(cls) if (v_ := Config._value(cls, k))}

    @staticmethod
    def _value(cls, key):
        attr = getattr(cls, key)
        if key.startswith("_"):
            attr = None
        elif isclass(attr):
            if attr == abrain.Config:
                attr = abrain.Config.to_json()
            elif not issubclass(attr, Enum):
                attr = Config._get_items(attr)
            else:
                attr = None
        elif callable(attr):
            attr = None

        return attr

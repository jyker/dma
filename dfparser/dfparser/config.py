import os
import toml
from typing import Dict
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.joinpath('data/config.toml')


class Config:
    fields = ['separator', 'prefix', 'suffix', 'gentag']

    def __init__(self, cfg_path: os.PathLike = CONFIG_PATH):
        self.cfg = Config.load(cfg_path)

    def get(self, key, engine):
        engine = engine.replace('+', '-')
        default = self.cfg[key]['default']
        if key == 'separator':
            return self.cfg[key].get(engine, default)
        else:
            return self.cfg[key].get(engine, []) + default

    def get_separator(self, engine: str):
        return self.get('separator', engine)

    def get_prefix(self, engine: str):
        return self.get('prefix', engine)

    def get_suffix(self, engine: str):
        return self.get('suffix', engine)

    def get_gentag(self, engine: str):
        return self.get('gentag', engine)

    def get_genlabel(self, engine: str):
        return self.get('genlabel', engine)

    def get_genfamily(self, engine: str):
        return self.get('genfamily', engine)

    @classmethod
    def check(cls, cfg: Dict):
        for key in cls.fields:
            try:
                cfg[key]
            except KeyError:
                raise KeyError(f"ERR_CFG: {key} is not found")

    @staticmethod
    def load(cfg_path: os.PathLike):
        try:
            cfg = toml.load(cfg_path)
        except BaseException as e:
            raise ValueError(f"ERR_CFG: {e}")
        Config.check(cfg)
        return cfg

    @staticmethod
    def loads(cfg: str):
        try:
            cfg = toml.loads(cfg)
        except BaseException as e:
            raise ValueError(f"ERR_CFG: {e}")
        Config.check(cfg)
        return cfg

    @staticmethod
    def makeExample(fpath: os.PathLike = None):
        """make example config
        """
        config = Config()
        if fpath is None:
            cfg_path = Path(os.getcwd()).joinpath(f'trma_example.toml')
        else:
            cfg_path = fpath

        with open(cfg_path, "w") as f:
            toml.dump(config.cfg, f)
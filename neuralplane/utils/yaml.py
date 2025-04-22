import yaml

from easydict import EasyDict as edict
from pathlib import Path

def load_yaml(file_path: Path) -> edict:
    config = yaml.load(file_path.read_text(), Loader=yaml.Loader)

    base_config = yaml.load((file_path.parent / config['_BASE_']).read_text(), Loader=yaml.Loader)
    base_config.update(config)

    return edict(base_config)

def update_yaml(file_path: Path, config: edict):
    old_config = yaml.load(file_path.read_text(), Loader=yaml.Loader)
    base_config = yaml.load((file_path.parent / old_config['_BASE_']).read_text(), Loader=yaml.Loader)

    new_config = dict(old_config.items())
    config = to_dict(config)
    for key in config:
        if key not in base_config:
            new_config[key] = config[key]
        else:
            if base_config[key] != config[key]:
                new_config[key] = config[key]

    file_path.write_text(yaml.dump(new_config), encoding='utf-8')

def to_dict(config: edict) -> dict:
    new_config = {}
    for key in config:
        if isinstance(config[key], edict):
            new_config[key] = to_dict(config[key])
        else:
            new_config[key] = config[key]
    return new_config

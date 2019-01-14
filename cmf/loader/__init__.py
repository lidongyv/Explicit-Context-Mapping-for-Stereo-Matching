import json

from cmf.loader.Flying3d import Flying3d
from cmf.loader.SceneFlow import SceneFlow
from cmf.loader.KITTI import KITTI
def get_loader(name):
    """get_loader

    :param name:
    """
    print(name)
    return {
        'flying3d': Flying3d,
        'sceneflow':SceneFlow,
        'kitti':KITTI,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']

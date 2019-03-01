import json

from cmf.loader.Flying3d import Flying3d
from cmf.loader.SceneFlow import SceneFlow
from cmf.loader.KITTI import KITTI
from cmf.loader.KITTI12 import KITTI12
from cmf.loader.KITTI15 import KITTI15
def get_loader(name):
    """get_loader

    :param name:
    """
    print(name)
    return {
        'flying3d': Flying3d,
        'sceneflow':SceneFlow,
        'kitti':KITTI,
        'kitti12':KITTI12,
        'kitti15':KITTI15,
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']

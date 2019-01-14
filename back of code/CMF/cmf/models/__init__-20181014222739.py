# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-03-18 15:24:33
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-14 22:27:36

import torchvision.models as models

from cmf.models.cmfsm import *
from cmf.models.cmf import *
def get_model(name):
    model = _get_model_instance(name)

    model = model()

    return model

def _get_model_instance(name):
    try:
        return {
            'cmf':cmf,
            'cmfsm':cmfsm,
        }[name]
    except:
        print('Model {} not available'.format(name))

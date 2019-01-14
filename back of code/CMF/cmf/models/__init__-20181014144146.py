# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-03-18 15:24:33
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-25 09:45:46

import torchvision.models as models


from pssm.models.rstereo import *
def get_model(name):
    model = _get_model_instance(name)

    model = model()

    return model

def _get_model_instance(name):
    try:
        return {
            'rstereo':rstereo,
        }[name]
    except:
        print('Model {} not available'.format(name))

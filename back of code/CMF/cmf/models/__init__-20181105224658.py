# -*- coding: utf-8 -*-
# @Author: yulidong
# @Date:   2018-03-18 15:24:33
# @Last Modified by:   yulidong
# @Last Modified time: 2018-11-05 22:46:55

import torchvision.models as models

from cmf.models.cmfsm import *
from cmf.models.cmfsm_sub_8 import *
from cmf.models.cmfsm_sub_16 import *
from cmf.models.cmf import *
from cmf.models.bilinear_cmf import *
from cmf.models.bilinear_cmf_sub_8 import *
from cmf.models.bilinear_cmf_sub_16 import *
from cmf.models.cm_sub_8 import *
def get_model(name):
    model = _get_model_instance(name)

    model = model()

    return model

def _get_model_instance(name):
    try:
        return {
            'cmf':cmf,
            'cmfsm':cmfsm,
            'bilinear_cmf':bilinear_cmf,
            'cmfsm_sub_8':cmfsm_sub_8,
            'cmfsm_sub_16':cmfsm_sub_16,
            'bilinear_cmf_sub_8':bilinear_cmf_sub_8, 
            'bilinear_cmf_sub_16':bilinear_cmf_sub_16,  
            'cm_sub_8':cm_sub_8,
        }[name]
    except:
        print('Model {} not available'.format(name))

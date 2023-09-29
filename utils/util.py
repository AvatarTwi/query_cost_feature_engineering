import os
import pickle
import time
import config
from dataset.postgres_tpch_dataset.dim_dict import get_feature


class Utils:

    @staticmethod
    def path_build(root):
        if not os.path.exists(root):
            os.makedirs(root)

def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        config.total += float(round((e_time - s_time) * 1000000000) / 1000)
        return res

    return inner


def print_feature():
    with open('v2200-2000-2000-2000/tpch/knob_model_linear_exp1/save_model/1024/openfe_values_array.pickle', 'rb') as f:
            attr_val_dict = pickle.load(f)

    for key in attr_val_dict.keys():
        print(key,len(attr_val_dict[key]),[(k.name,k.get_fnode(),get_feature(key,k.get_fnode())) for k in attr_val_dict[key]])

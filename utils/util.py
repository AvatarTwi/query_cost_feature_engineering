import os
import pickle
import time
import config

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
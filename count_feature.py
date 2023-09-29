import pickle
import random

import numpy as np
from matplotlib import pyplot as plt

import config

from dataset.postgres_tpch_dataset.tpch_utils_origin import tpch_dim_dict as origin_tpch_dim
from dataset.postgres_tpch_dataset.tpch_utils_knob import tpch_dim_dict as knob_tpch_dim

from dataset.sysbench_dataset.sysbench_utils_origin import sysbench_dim_dict as origin_sysbench_dim
from dataset.sysbench_dataset.sysbench_utils_knob import sysbench_dim as knob_sysbench_dim

from dataset.job_dataset.job_utils_origin import job_dim as origin_job_dim
from dataset.job_dataset.job_utils_knob import job_dim as knob_job_dim

from utils.util import Utils

PSQLTPCH = {
    "dim_dict": {
        "origin_model": origin_tpch_dim,
        "knob_model": knob_tpch_dim,
    },
}
PSQLSysbench = {
    "dim_dict": {
        "origin_model": origin_sysbench_dim,
        "knob_model": knob_sysbench_dim,
    },
}
PSQLJOB = {
    "dim_dict": {
        "origin_model": origin_job_dim,
        "knob_model": knob_job_dim,
    },
}

DATASET_TYPE = {
    "tpch": PSQLTPCH,
    "sysbench": PSQLSysbench,
    "job": PSQLJOB
}

def return_all_feature_num(model_type, type):
    if model_type == 'MSCN':
        with open('2200-2000-2000-2000/' + type + '/knob_model/serialize_knob_dim_dict.pickle',
                  'rb') as f:
            dim_dict = pickle.load(f)['table_list_len']
    else:
        dim_dict = DATASET_TYPE[type]['dim_dict']['knob_model'](config.cost_factor_dict)

    return dim_dict


plt.rcParams['font.size'] = 10

def return_shap_feature_num(model_type, type, shap, num="0"):
    if num == "0":
        pickle_file = shap + '_values_array.pickle'
    else:
        pickle_file = shap + '_values_array_' + num + '.pickle'

    if model_type == 'MSCN':
        with open(
                '2200-2000-2000-2000/' + type + '/knob_model/save_model_MSCN/1024/' + pickle_file,
                'rb') as f:
            dim_dict = len(pickle.load(f))
    else:
        with open(
                '2200-2000-2000-2000/' + type + '/knob_model/save_model_QPPNet/1024/' + pickle_file,
                'rb') as f:
            attr_val_dict = pickle.load(f)
        dim_dict = {}
        for key in attr_val_dict.keys():
            dim_dict[key] = len(attr_val_dict[key])

    return dim_dict


def plot_bar(name, all_feat_num, shap_feat_nums):
    fig, ax = plt.subplots(figsize=(20, 4))

    fig.subplots_adjust(left=0.05, right=0.980, top=0.9, bottom=0.3,
                        wspace=0.2, hspace=0.1)
    keys = []
    del_keys = []
    method_values_dict = {}
    x = 0

    method_values_dict['inner'] = []
    for ky in shap_feat_nums.keys():
        method_values_dict[ky] = []

    for key in all_feat_num.keys():
        x += 1
        flag = 1
        keys.append(key)
        method_values_dict['inner'].append(all_feat_num[key])
        for ky in shap_feat_nums.keys():
            try:
                dict = shap_feat_nums[ky]
                method_values_dict[ky].append(dict[key])
            except:
                method_values_dict[ky].append(all_feat_num[key])
            if method_values_dict['inner'][-1] != method_values_dict[ky][-1]:
                flag = 0
        if flag == 1:
            if random.randint(1, 10) < 0:
                continue
            else:
                del_keys.append(key)
                del method_values_dict['inner'][-1]
                for ky in shap_feat_nums.keys():
                    del method_values_dict[ky][-1]
    for key in del_keys:
        keys.remove(key)
    x = np.arange(x - len(del_keys))

    width = 0.21
    multiplier = 0

    for method, values in method_values_dict.items():
        offset = width * multiplier
        method1 = method.replace("R2", "GREEDY")
        method1 = method1.replace("shap", "FR")
        method1 = method1.replace("grad", "GD")
        method1 = method1.replace("inner", "SFO")
        rects = ax.bar(x + offset, values, width, label=method1)
        ax.bar_label(rects, padding=4)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(name, fontsize=16)
    ax.set_xticks(x + width, keys)
    ax.legend(loc='upper right', ncols=4, fontsize=14)
    ax.set_ylim(0, 200)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

    plt.savefig("visualization/plan/res/fig5/" + name)


def count_for_all():
    shap_list = ['shap', 'grad', 'R2']
    ds_list = [
        'tpch',
        # 'sysbench',
        # 'job'
    ]

    for type in ds_list:
        all_feat_num = return_all_feature_num('QPPNet', type)
        shap_feat_nums = {}
        for shap in shap_list:
            shap_feat_nums[shap] = return_shap_feature_num('QPPNet', type, shap)
        plot_bar(type, all_feat_num, shap_feat_nums)


def count_for_shap():
    numbers = [200, 250, 300, 350, 400, 450, 500]

    all_feat_num = return_all_feature_num('QPPNet', 'tpch')
    sum = 0
    for ky in all_feat_num.keys():
        sum += all_feat_num[ky]

    shap_feat_nums = {}
    for number in numbers:
        part_sum = 0
        shap_feat_nums[number] = return_shap_feature_num('QPPNet', 'tpch', 'shap', str(number))
        keys = [key for key in shap_feat_nums[number].keys()]

        for ky in all_feat_num.keys():
            if ky not in keys:
                shap_feat_nums[number][ky] = all_feat_num[ky]
            part_sum += shap_feat_nums[number][ky]

        print((sum - part_sum) / sum)


if __name__ == '__main__':
    count_for_all()
    # count_for_shap()

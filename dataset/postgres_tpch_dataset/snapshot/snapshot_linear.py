import collections
import json
import os
import pickle
import time

import numpy as np
from scipy.optimize import curve_fit

# y = c0 * n1 + c1
from sklearn.model_selection import train_test_split

from dataset.postgres_tpch_dataset.attr_rel_dict import all_dicts


def func2(t, c0, c1):
    return c0 * t + c1


# y = c0 * n1 * n2 + c1 * n1 + c2 * n2 + c3
def func4(t, c0, c1, c2, c3):
    return c0 * t[0] + c1 * t[1] + c2 * t[2] + c3


def getMSE(a, b):
    count = 0
    sum = 0.0

    # workload_error=0.0
    for i in range(len(a)):
        if b[i] > 0:
            sum += abs(a[i] - b[i]) / b[i]
            count += 1
    if count == 0:
        return 0

    return sum / count


def linear2(X, y):
    if len(X) <= 2:
        a, pcov = curve_fit(func2, X.T, y)
        error = 0
    else:
        for i in range(exp):
            X, test_X, y, test_y = train_test_split(X, y, train_size=train_size)
        train_X = X
        train_y = y
        a, pcov = curve_fit(func2, train_X.T, train_y)

        re = []

        for i in range(len(test_X)):
            re.append(a[0] * test_X[i] + a[1] + bias)

        error = getMSE(re, test_y)

    return a, error


def linear4(X, y):
    if len(X) <= 2:
        a, pcov = curve_fit(func4, X.T, y)
        error = 0
        # a = [0]
        # error = 0
    else:
        for i in range(exp):
            X, test_X, y, test_y = train_test_split(X, y, train_size=train_size)
        train_X = X
        train_y = y

        a, pcov = curve_fit(func4, train_X.T, train_y)

        re = []

        for i in range(len(test_X)):
            re.append(a[0] * test_X[i, 0] * test_X[i, 1] + a[1] * test_X[i, 0] + a[2] * test_X[i, 1] + a[3] + bias)

        error = getMSE(re, test_y)

    return a, error


def get_scan_cost_factor_dic(data):
    X = data[:, 0]
    y = data[:, 1]

    a, error = linear2(X, y)
    return a, error


def get_materialize_cost_factor_dic(data):
    X = data[:, 0]
    y = data[:, 1]

    a, error = linear2(X, y)
    return a, error


def get_sort_cost_factor_dic(data):
    X = data[:, 0:1]
    X[:, 0] = np.ceil(X[:, 0] * np.log(X[:, 0]))
    X = X[:, 0]
    y = data[:, 1]

    a, error = linear2(X, y)
    return a, error


def get_agg_cost_factor_dic(data):
    X = data[:, 0]
    y = data[:, 1]

    a, error = linear2(X, y)
    return a, error


def get_hashjoin_cost_factor_dic(data):
    X = data[:, 0]
    y = data[:, 1]

    a, error = linear2(X, y)
    return a, error


def get_mergejoin_cost_factor_dic(data):
    X = data[:, 0]
    y = data[:, 1]

    a1, error1 = linear2(X, y)

    return a1, error1


def get_index_cost_factor_dic(data):
    X = data[:, 0]
    y = data[:, 1]

    a, error = linear2(X, y)
    return a, error


def get_nestedloop_cost_factor_dic(data):
    X = np.ones((data.shape[0], 3), dtype=float)
    x0_list = np.multiply(list(data[:, 2]), list(data[:, 3]))

    X[:, 0] = x0_list
    X[:, 1] = data[:, 2]
    X[:, 2] = data[:, 3]

    y = data[:, 1]

    a, error = linear4(X, y)

    return a, error


def get_default_cost_factor_dic(data):
    if 5 > data.shape[0]:
        add_on = np.zeros((5 - data.shape[0], data.shape[1]))
        data = np.concatenate((data, add_on), axis=0)

    X = data[:, 0]
    y = data[:, 1]

    a, error = linear2(X, y)
    return a, error


# TODO 针对每个算子，计算cost_factor所用的函数
# 其中get_hashjoin_cost_factor_dic和get_mergejoin_cost_factor_dic还暂时粗糙，都只替换为最简单的公式
# get_default_cost_factor_dic只返回[0], 因为杨曼学姐留下的代码中没有对应的公式
TPCH_GET_COST_FACTOR_DIC = \
    {
        "Hash Join": get_hashjoin_cost_factor_dic,
        "Merge Join": get_mergejoin_cost_factor_dic,
        "Seq Scan": get_scan_cost_factor_dic,
        "Index Scan": get_index_cost_factor_dic,
        "Index Only Scan": get_index_cost_factor_dic,
        "Sort": get_sort_cost_factor_dic,
        "Aggregate": get_agg_cost_factor_dic,
        "Nested Loop": get_nestedloop_cost_factor_dic,

        "Gather Merge": get_default_cost_factor_dic,
        "Gather": get_default_cost_factor_dic,
        "Hash": get_default_cost_factor_dic,
        "Memoize": get_default_cost_factor_dic,
        "Materialize": get_default_cost_factor_dic,
        "Bitmap Heap Scan": get_default_cost_factor_dic,
        "Bitmap Index Scan": get_default_cost_factor_dic,
        "Limit": get_default_cost_factor_dic
    }


def func(data):
    return [0], 0


TPCH_GET_COST_FACTOR_DIC = collections.defaultdict(lambda: func, TPCH_GET_COST_FACTOR_DIC)

train_size = 0.8
n_splines = 4
bias = 0.02
exp = 1


def linear(op_data_dic, opt):
    cost_factor_dict = {}
    error = {}

    start_time = time.time()
    for op in all_dicts:
        if op in op_data_dic.keys():
            cost_factor_dict[op], error[op] = TPCH_GET_COST_FACTOR_DIC[op](np.array(op_data_dic[op]))
        else:
            cost_factor_dict[op] = [0]
            error[op] = 0

    end_time = time.time()
    error["total_time"] = str(int(round((end_time - start_time))))

    with open(opt.data_structure + '/cost_factor_dict_linear_' + 'exp' + str(exp) + '.pickle', 'wb') as f:
        pickle.dump(cost_factor_dict, f)
    with open(opt.data_structure + '/cost_factor_dict_linear_error_' + 'exp' + str(exp) + '.pickle', 'wb') as f:
        pickle.dump(error, f)


def linear1(op_data_dic):
    cost_factor_dict = {}
    error = {}

    for op in op_data_dic.keys():
        for o in all_dicts:
            if o in op:
                true_op = o
                break
        # try:
        cost_factor_dict[op], error[op] = TPCH_GET_COST_FACTOR_DIC[true_op](np.array(op_data_dic[op]))
        # except:
        #     continue

    return cost_factor_dict

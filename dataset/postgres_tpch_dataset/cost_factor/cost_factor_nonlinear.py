import collections
import json
import os
import pickle
import numpy as np
from pygam import LinearGAM, s, f
from sklearn.model_selection import train_test_split
from dataset.postgres_tpch_dataset.attr_rel_dict import all_dicts
import time

np.seterr(divide='ignore',invalid='ignore')


def getMSE(a, b):
    count = 0
    sum = 0.0

    for i in range(len(a)):
        if b[i] > 0:
            sum += abs(a[i] - b[i]) / b[i]
            count += 1

    if count == 0:
        return 0
    return sum / count


def linearGam(X, y):
    if len(X) == 0 or len(y) == 0:
        return [0], 0

    for i in range(exp):
        X, test_X, y, test_y = train_test_split(X, y, train_size=train_size)
    train_X = X
    train_y = y

    gam = LinearGAM(n_splines=n_splines)
    gam.gridsearch(train_X, train_y)
    predict_y = gam.predict(test_X)
    error = getMSE(predict_y, test_y)

    return gam.coef_, error


def get_scan_cost_factor(data):
    X = data[:, 0]
    y = data[:, 1]

    a, error = linearGam(X.reshape(-1, 1), y)
    return a, error


def get_materialize_cost_factor(data):
    X = data[:, 0]
    y = data[:, 1]

    a, error = linearGam(X.reshape(-1, 1), y)
    return a, error


def get_sort_cost_factor(data):
    X = data[:, 0:1]
    X[:, 0] = np.ceil(X[:, 0] * np.log(X[:, 0]))
    X = X[:, 0]
    y = data[:, 1]

    a, error = linearGam(X.reshape(-1, 1), y)
    return a, error


def get_agg_cost_factor(data):
    X = data[:, 0]
    y = data[:, 1]

    a, error = linearGam(X.reshape(-1, 1), y)
    return a, error


def get_hashjoin_cost_factor(data):
    X = data[:, 0]
    y = data[:, 1]

    a, error = linearGam(X.reshape(-1, 1), y)
    return a, error


def get_mergejoin_cost_factor(data):
    X = data[:, 0]
    y = data[:, 1]

    a1, error1 = linearGam(X.reshape(-1, 1), y)

    x0_list = np.multiply(list(data[:, 2]), list(data[:, 3]))

    X = np.ones((data.shape[0], 3), dtype=float)
    X[:, 0] = x0_list
    X[:, 1] = data[:, 2]
    X[:, 2] = data[:, 3]

    a2, error2 = linearGam(X, y)

    if error2 > error1:
        a = a1
        error = error1
    else:
        a = a2
        error = error2

    return a, error


def get_index_cost_factor(data):
    X = data[:, 0]
    y = data[:, 1]

    a, error = linearGam(X.reshape(-1, 1), y)
    return a, error


def get_nestedloop_cost_factor(data):
    X = np.ones((data.shape[0], 3), dtype=float)
    x0_list = np.multiply(list(data[:, 2]), list(data[:, 3]))

    X[:, 0] = x0_list
    X[:, 1] = data[:, 2]
    X[:, 2] = data[:, 3]
    y = data[:, 1]

    a, error = linearGam(X, y)

    return a, error


def get_default_cost_factor(data):
    X = data[:, 0]
    y = data[:, 1]

    a, error = linearGam(X.reshape(-1, 1), y)
    return a, error


# TODO 针对每个算子，计算cost_factor所用的函数
# 其中get_hashjoin_cost_factor和get_mergejoin_cost_factor还暂时粗糙，都只替换为最简单的公式
# get_default_cost_factor只返回[0], 因为杨曼学姐留下的代码中没有对应的公式
TPCH_GET_COST_FACTOR_DIC = \
    {
        "Hash Join": get_hashjoin_cost_factor,
        "Merge Join": get_mergejoin_cost_factor,
        "Seq Scan": get_scan_cost_factor,
        "Index Scan": get_index_cost_factor,
        "Index Only Scan": get_index_cost_factor,
        "Sort": get_sort_cost_factor,
        "Aggregate": get_agg_cost_factor,
        "Nested Loop": get_nestedloop_cost_factor,

        "Gather Merge": get_default_cost_factor,
        "Gather": get_default_cost_factor,
        "Hash": get_default_cost_factor,
        "Memoize": get_default_cost_factor,
        "Materialize": get_default_cost_factor,
        "Bitmap Heap Scan": get_default_cost_factor,
        "Bitmap Index Scan": get_default_cost_factor,
        "Limit": get_default_cost_factor
    }


def func(data):
    return [0], 0


TPCH_GET_COST_FACTOR_DIC = collections.defaultdict(lambda: func, TPCH_GET_COST_FACTOR_DIC)

cost_factor_dict = {}
error = {}

train_size = 0.8
n_splines = 4
bias = 0.02
exp=1

def nonlinear(op_data_dic, opt):
    start_time=time.time()
    for op in all_dicts:
        if op in op_data_dic.keys():
            cost_factor_dict[op], error[op] = TPCH_GET_COST_FACTOR_DIC[op](np.array(op_data_dic[op]))
        else:
            cost_factor_dict[op] = [0]
            error[op] = 0

    end_time = time.time()
    error["total_time"] = str(int(round((end_time - start_time))))

    with open(opt.data_structure + '/cost_factor_dict_nonlinear_' + 'exp' + str(exp) + '_' + str(
            n_splines + 1) + '.pickle', 'wb') as f:
        pickle.dump(cost_factor_dict, f)
    with open(opt.data_structure + '/cost_factor_dict_nonlinear_error_' + 'exp' + str(exp) + '_' + str(
            n_splines + 1) + '.pickle', 'wb') as f:
        pickle.dump(error, f)

def nonlinear1(op_data_dic):
    for op in all_dicts:
        if op in op_data_dic.keys():
            cost_factor_dict[op], error[op] = TPCH_GET_COST_FACTOR_DIC[op](np.array(op_data_dic[op]))
        else:
            cost_factor_dict[op] = [0]
            error[op] = 0
    return cost_factor_dict
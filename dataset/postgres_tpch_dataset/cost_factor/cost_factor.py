import json
import os
import pickle
import random
import re
import time
import numpy as np

from config import template_num
from dataset.postgres_tpch_dataset.attr_rel_dict import all_dicts
from dataset.postgres_tpch_dataset.cost_factor.cost_factor_linear import linear, linear1
from dataset.postgres_tpch_dataset.cost_factor.cost_factor_nonlinear import nonlinear, nonlinear1
from dataset.postgres_tpch_dataset.cost_factor.cost_factor_smart import smart, smart1


def get_all_plans(fname):
    jsonstrs = []
    curr = ""
    prev = None
    prevprev = None

    with open(fname, 'r', encoding='utf-8') as f:
        for row in f:
            if not ('[' in row or '{' in row or ']' in row or '}' in row or ':' in row):
                continue
            newrow = row.replace('+', "").replace("(1 row)\n", "").strip('\n').strip(' ')

            if 'CREATE' not in newrow and 'DROP' not in newrow and 'Tim' != newrow[:3]:
                curr += newrow

            if prevprev is not None and 'Execution Time' in prevprev:
                jsonstrs.append(curr.strip(' ').strip('QUERY PLAN').strip('-'))
                curr = ""
            prevprev = prev
            prev = newrow

    strings = [s for s in jsonstrs if s[-1] == ']']
    jss = [json.loads(s)[0]['Plan'] for s in strings]

    # jss is a list of json-transformed dicts, one for each query
    return jss


# TODO 现在只暂时提取其中 Plan Rows 和 Actual Total Time 数据，如公式中需要其他数据再进行改进添加
def get_func_input_data(op_data_dic, data):  # Helper for sample_data
    data_tuple = [0.0] * 4

    if 'Plans' in data:
        for plan in data['Plans']:
            get_func_input_data(op_data_dic, plan)

    data_tuple[0] = float(data['Actual Rows'])
    if int(data_tuple[0]) == 0:
        return
    data_tuple[1] = float(data['Actual Total Time'])

    if data["Node Type"] in ['Merge Join', 'Hash Join', 'Nested Loop']:
        rows = [plan['Actual Rows'] for plan in data['Plans']]
        if rows != []:
            data_tuple[2] = float(rows[0])
            data_tuple[3] = float(rows[1])

    if data["Node Type"] not in op_data_dic.keys():
        op_data_dic[data["Node Type"]] = []
    op_data_dic[data["Node Type"]].append(data_tuple)

    # if 'Relation Name' in data:
    #     if data["Node Type"]+"_"+data["Relation Name"] not in op_data_dic.keys():
    #         op_data_dic[data["Node Type"]+"_"+data["Relation Name"]] = []
    # else:
    #     if data["Node Type"] not in op_data_dic.keys():
    #         op_data_dic[data["Node Type"]] = []
    #
    # if 'Relation Name' in data:
    #     op_data_dic[data["Node Type"] + "_" + data["Relation Name"]].append(data_tuple)
    # else:
    #     op_data_dic[data["Node Type"]].append(data_tuple)


def cost_factor_main(opt):
    datapaths = {}
    num = 22
    op_data_dic = {}

    if not os.path.exists(opt.data_structure + '/op_data_dic.pickle'):
        for root, dirs, files in os.walk(opt.data_dir):
            for dir in dirs:
                for i in range(num):
                    if i not in datapaths.keys():
                        datapaths[i] = []
                    datapaths[i].append(root + "/" + dir + "/q" + str(i + 1) + ".json")

        for i in range(num):
            print("-----query" + str(i + 1) + "------")
            for fname in datapaths[i]:
                temp_data = get_all_plans(fname)
                [get_func_input_data(op_data_dic, data) for data in temp_data]

        with open(opt.data_structure + '/op_data_dic.pickle', 'wb') as f:
            pickle.dump(op_data_dic, f)
    else:
        with open(opt.data_structure + '/op_data_dic.pickle', 'rb') as f:
            op_data_dic = pickle.load(f)

    linear(op_data_dic, opt)
    nonlinear(op_data_dic, opt)
    smart(op_data_dic, opt)


def cost_factor_one2one(opt, dir, temp_data, test_size):
    op_data_dic = {}
    op_data_dic1 = {}
    pattern_num = re.compile(r'\d+')

    for t_data in temp_data:
        [get_func_input_data(op_data_dic, data) for data in t_data]
        # [get_func_input_data(op_data_dic, data) for data in t_data[test_size:]]

    if opt.change == True:
        data4cost = []

        # for i in range(22):
        #     fname = opt.data_dir + "_i7/" + str(int(dir)) + "/q" + str(i + 1) + ".json"
        #     data1 = get_all_plans(fname)
        #     data4cost.append(data1)

        # for t_data in data4cost:
        #     [get_func_input_data(op_data_dic1, data) for data in t_data]

        fname = opt.data_dir + "_template" + "/" + dir + "/q1.json"
        data1 = get_all_plans(fname)

        [get_func_input_data(op_data_dic1, data) for data in data1]

        for op in op_data_dic.keys():
            if op not in op_data_dic1.keys():
                op_data_dic1[op] = op_data_dic[op]

        for op in op_data_dic1.keys():
            op_data_dic[op] = op_data_dic1[op]

    if 'template' in opt.mid_data_dir:
        fname = opt.data_dir + "_template" + "/" + dir + "/q1.json"
        data1 = get_all_plans(fname)

        random.seed(10)
        [get_func_input_data(op_data_dic1, data) for data in
         random.sample(data1, min(len(data1), int(pattern_num.findall(opt.mid_data_dir)[-1])))]

        for op in op_data_dic.keys():
            if op not in op_data_dic1.keys() or len(op_data_dic1[op]) < 3:
                op_data_dic1[op] = op_data_dic[op]

        for op in op_data_dic1.keys():
            op_data_dic[op] = op_data_dic1[op]

    linear = linear1(op_data_dic)

    train_data = []
    test_data = []

    for idx, t_data in enumerate(temp_data):
        [add_cost_factor(data, linear) for data in t_data]
        # train_data.append(t_data[test_size:])
        # test_data.append(t_data[:test_size])

    # return train_data,test_data
    return temp_data

    # if 'nonlinear' in opt.mid_data_dir:
    #     nonlinear = nonlinear1(op_data_dic)
    #     for t_data in temp_data:
    #         [add_cost_factor(data, nonlinear) for data in t_data]
    #     return temp_data
    #
    # if 'smart' in opt.mid_data_dir:
    #     smart = smart1(op_data_dic)
    #     for t_data in temp_data:
    #         [add_cost_factor(data, smart) for data in t_data]
    #     return temp_data


def add_cost_factor(data, cost_factor):
    data['inner_vector'] = list(cost_factor[data['Node Type']])

    # if 'Relation Name' in data:
    #     data['inner_vector'] = list(cost_factor[data['Node Type'] + "_" + data["Relation Name"]])
    # else:
    #     data['inner_vector'] = list(cost_factor[data['Node Type']])

    if 'Plans' in data:
        for sub_tree in data['Plans']:
            add_cost_factor(sub_tree, cost_factor)

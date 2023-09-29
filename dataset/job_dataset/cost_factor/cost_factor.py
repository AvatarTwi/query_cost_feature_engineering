import json
import os
import pickle
import random
import re

from dataset.sysbench_dataset.cost_factor.cost_factor_linear import linear, linear1
from dataset.sysbench_dataset.cost_factor.cost_factor_nonlinear import nonlinear, nonlinear1
from dataset.sysbench_dataset.cost_factor.cost_factor_smart import smart, smart1


def get_all_plans(fname):
    jss = []
    print(fname)

    with open(fname, "r", encoding='utf-8') as f:
        lines = [line for line in f.readlines()]
        lineid = 0
        while lineid < len(lines):
            if ' CST [' not in lines[lineid]:
                lineid += 1
                continue
            while lineid < len(lines) and ' CST [' in lines[lineid]:
                plan_strs = []
                lineid += 1
                while lineid < len(lines) and ' CST [' not in lines[lineid]:
                    plan_strs.append(lines[lineid])
                    lineid += 1
                if plan_strs != []:
                    # print(plan_strs)
                    plan_obj = json.loads(s=''.join(plan_strs))
                    jss.append(plan_obj['Plan'])

    return jss


def get_all_plans_sp(fname):
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

    # print(data)
    if 'Plans' in data:
        for plan in data['Plans']:
            get_func_input_data(op_data_dic, plan)

    data_tuple[0] = int(data['Actual Rows'])
    data_tuple[1] = float(data['Actual Total Time'])

    if data["Node Type"] in ['Merge Join', 'Hash Join', 'Nested Loop']:
        rows = [plan['Actual Rows'] for plan in data['Plans']]
        if rows != []:
            data_tuple[2] = float(rows[0])
            data_tuple[3] = float(rows[1])

    if data["Node Type"] not in op_data_dic:
        op_data_dic[data["Node Type"]] = []

    op_data_dic[data["Node Type"]].append(data_tuple)


def cost_factor_main(opt):
    datapaths = []
    op_data_dic = {}

    if not os.path.exists(opt.data_structure + '/op_data_dic.pickle'):
        for root, dirs, files in os.walk(opt.data_dir):
            for dir in dirs:
                datapaths.append(root + "/" + dir + "/serverlog")

        for fname in datapaths:
            temp_data = get_all_plans(fname)
            [get_func_input_data(op_data_dic, data) for data in temp_data]

        with open(opt.data_structure + '/op_data_dic.pickle', 'wb') as f:
            pickle.dump(op_data_dic, f)
    else:
        with open(opt.data_structure + '/op_data_dic.pickle', 'rb') as f:
            op_data_dic = pickle.load(f)

    linear(op_data_dic, opt)
    # nonlinear(op_data_dic, opt)
    # smart(op_data_dic, opt)


def cost_factor_one2one(opt, dir, temp_data, test_size):
    op_data_dic = {}
    op_data_dic1 = {}
    pattern_num = re.compile(r'\d+')

    [get_func_input_data(op_data_dic, data) for data in temp_data]
    # [get_func_input_data(op_data_dic, data) for data in temp_data[test_size:]]

    # if opt.change == True:
    #     fname = opt.data_dir + "/" + str(int(dir) - 1) + "/serverlog"
    #     data4cost = get_all_plans(fname)
    #     [get_func_input_data(op_data_dic1, data) for data in data4cost]
    #
    #     for op in op_data_dic.keys():
    #         if op not in op_data_dic1.keys():
    #             op_data_dic1[op] = op_data_dic[op]
    #
    #     for op in op_data_dic1.keys():
    #         op_data_dic[op] = op_data_dic1[op]

    if 'template' in opt.mid_data_dir:
        fname = opt.data_dir + "_template" + "/" + dir + "/q1.json"
        data1 = get_all_plans_sp(fname)

        random.seed(4)
        [get_func_input_data(op_data_dic1, data) for data in
         random.sample(data1, min(len(data1), int(pattern_num.findall(opt.mid_data_dir)[-1])))]

        for op in op_data_dic.keys():
            if op not in op_data_dic1.keys():
                op_data_dic1[op] = op_data_dic[op]

        for op in op_data_dic1.keys():
            op_data_dic[op] = op_data_dic1[op]

    linear = linear1(op_data_dic)

    [add_cost_factor(data, linear) for data in temp_data]
    train_data = temp_data[test_size:]
    test_data = temp_data[:test_size]

    # if 'nonlinear' in opt.mid_data_dir:
    #     nonlinear = nonlinear1(op_data_dic)
    #     [add_cost_factor(data, nonlinear) for data in temp_data]
    #
    # if 'smart' in opt.mid_data_dir:
    #     smart = smart1(op_data_dic)
    #     [add_cost_factor(data, smart) for data in temp_data]

    return temp_data
    # return train_data,test_data


def add_cost_factor(data, cost_factor):
    data['inner_vector'] = list(cost_factor[data['Node Type']])

    if 'Plans' in data:
        for sub_tree in data['Plans']:
            add_cost_factor(sub_tree, cost_factor)

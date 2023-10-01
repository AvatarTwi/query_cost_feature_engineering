import json
import os
import pickle
import random
import re

from dataset.sysbench_dataset.snapshot.snapshot_linear import linear1


def get_all_plans(fname):
    jss = []

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

def cost_factor_one2one(opt, dir, temp_data):
    op_data_dic = {}
    op_data_dic1 = {}
    pattern_num = re.compile(r'\d+')

    [get_func_input_data(op_data_dic, data) for data in temp_data]

    if 'template' in opt.mid_data_dir:
        fname = opt.data_dir + "_template" + "/" + dir + "/serverlog"
        data1 = get_all_plans(fname)

        [get_func_input_data(op_data_dic1, data) for data in
         random.sample(data1, max(len(data1), int(pattern_num.findall(opt.mid_data_dir)[-1])))]

        for op in op_data_dic.keys():
            if op not in op_data_dic1.keys():
                op_data_dic1[op] = op_data_dic[op]

        for op in op_data_dic1.keys():
            op_data_dic[op] = op_data_dic1[op]

    linear = linear1(op_data_dic)

    [add_cost_factor(data, linear) for data in temp_data]

    return temp_data


def add_cost_factor(data, cost_factor):
    data['inner_vector'] = list(cost_factor[data['Node Type']])

    if 'Plans' in data:
        for sub_tree in data['Plans']:
            add_cost_factor(sub_tree, cost_factor)

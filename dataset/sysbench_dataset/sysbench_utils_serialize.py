import random
import re
import shutil

import numpy as np
import collections, os, json

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from torch.autograd import Variable
from torch.utils.data import dataset, random_split, DataLoader

from dataset.sysbench_dataset.attr_rel_dict import *
import pickle

from dataset.sysbench_dataset.cost_factor.cost_factor import cost_factor_main, cost_factor_one2one

import config
basics = 3  # get_basics(plan_dict)
num_rel = len(rel_names)
max_num_attr = 20

num_index = len(index_names)
SCALE = 1



op_data_dic = {}

TRAIN_TEST_SPLIT = 0.8

normalizer = Normalizer(norm='l2')

with open('dataset/sysbench_dataset/attr_val_dict.pickle', 'rb') as f:
    attr_val_dict = pickle.load(f)


# need to normalize Plan Width, Plan Rows, Total Cost, Hash Bucket
def get_basics(plan_dict):
    return [plan_dict['Plan Width'], plan_dict['Plan Rows'], plan_dict['Total Cost']]


def get_rel_one_hot(rel_name):
    arr = [0] * num_rel
    arr[rel_names.index(rel_name)] = 1
    return arr


def get_index_one_hot(index_name):
    arr = [0] * num_index
    arr[index_names.index(index_name)] = 1
    return arr


def get_rel_attr_one_hot(rel_name, filter_line):
    attr_list = rel_attr_list_dict[rel_name]

    med_vec, min_vec, max_vec = [0] * max_num_attr, [0] * max_num_attr, [0] * max_num_attr

    for idx, attr in enumerate(attr_list):
        if attr in filter_line:
            med_vec[idx] = attr_val_dict['med'][rel_name][idx]
            min_vec[idx] = attr_val_dict['min'][rel_name][idx]
            max_vec[idx] = attr_val_dict['max'][rel_name][idx]
    return min_vec + med_vec + max_vec


def get_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Seq Scan'
    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])
    try:
        rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                            plan_dict['Filter'])
    except:
        try:
            rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                                plan_dict['Recheck Cond'])
        except:
            if 'Filter' in plan_dict:
                print('************************* default *************************')
                print(plan_dict)
            rel_attr_vec = [0] * max_num_attr * 3

    return get_basics(plan_dict) + rel_vec + rel_attr_vec


def get_index_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Index Scan'

    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])
    index_vec = get_index_one_hot(plan_dict['Index Name'])

    try:
        rel_attr_vec = get_rel_attr_one_hot(plan_dict['Relation Name'],
                                            plan_dict['Index Cond'])
    except:
        if 'Index Cond' in plan_dict:
            print('********************* default rel_attr_vec *********************')
            print(plan_dict)
        rel_attr_vec = [0] * max_num_attr * 3

    res = get_basics(plan_dict) + rel_vec + rel_attr_vec + index_vec \
          + [1 if plan_dict['Scan Direction'] == 'Forward' else 0]

    return res


def get_bitmap_index_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Bitmap Index Scan'
    index_vec = get_index_one_hot(plan_dict['Index Name'])

    return get_basics(plan_dict) + index_vec


def get_hash_input(plan_dict):
    return get_basics(plan_dict) + [plan_dict['Hash Buckets'] if 'Hash Buckets' in plan_dict.keys() else 1]


def get_join_input(plan_dict):
    type_vec = [0] * len(join_types)
    type_vec[join_types.index(plan_dict['Join Type'].lower())] = 1
    par_rel_vec = [0] * len(parent_rel_types)
    if 'Parent Relationship' in plan_dict:
        par_rel_vec[parent_rel_types.index(plan_dict['Parent Relationship'].lower())] = 1
    return get_basics(plan_dict) + type_vec + par_rel_vec


def get_nested_input(plan_dict):
    type_vec = [0] * len(join_types)
    type_vec[join_types.index(plan_dict['Join Type'].lower())] = 1

    return get_basics(plan_dict) + type_vec


def get_sort_key_input(plan_dict):
    kys = plan_dict['Sort Key']
    one_hot = [0] * (num_rel * max_num_attr)
    for key in kys:
        key = key.replace('(', ' ').replace(')', ' ')
        for subkey in key.split(" "):
            if subkey != ' ' and '.' in subkey:
                rel_name, attr_name = subkey.split(' ')[0].split('.')
                if rel_name in rel_names:
                    one_hot[rel_names.index(rel_name) * max_num_attr
                            + rel_attr_list_dict[rel_name].index(attr_name.lower())] = 1

    return one_hot


def get_sort_input(plan_dict):
    sort_meth = [0] * len(sort_algos)
    if 'Sort Method' in plan_dict:
        if "external" not in plan_dict['Sort Method'].lower():
            sort_meth[sort_algos.index(plan_dict['Sort Method'].lower())] = 1

    return get_basics(plan_dict) + sort_meth


def get_aggreg_input(plan_dict):
    strat_vec = [0] * len(aggreg_strats)
    strat_vec[aggreg_strats.index(plan_dict['Strategy'].lower())] = 1
    partial_mode_vec = [0] if plan_dict['Parallel Aware'] == 'false' else [1]
    return get_basics(plan_dict) + strat_vec + partial_mode_vec


def get_modify_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Seq Scan'
    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])

    return get_basics(plan_dict) + rel_vec


sysbench_GET_INPUT = \
    {
        "Hash Join": get_join_input,
        "Merge Join": get_join_input,
        "Nested Loop": get_nested_input,
        "Seq Scan": get_scan_input,
        "Index Scan": get_index_scan_input,
        "Index Only Scan": get_index_scan_input,
        "Bitmap Heap Scan": get_scan_input,
        "Bitmap Index Scan": get_bitmap_index_scan_input,
        "Sort": get_sort_input,
        "Hash": get_hash_input,
        "Aggregate": get_aggreg_input,
        "ModifyTable": get_modify_input,
    }

sysbench_GET_INPUT = collections.defaultdict(lambda: get_basics, sysbench_GET_INPUT)


###############################################################################
#       Parsing data from csv files that contain json output of queries       #
###############################################################################

class SysbenchDataset():
    def __init__(self, opt):
        """
            Initialize the dataset by parsing the data files.
            Perform train test split and normalize each feature using mean and max of the train dataset.

            self.dataset is the train dataset
            self.test_dataset is the test dataset
        """
        mid_data_dir = opt.mid_data_dir
        self.num_sample_per_q = int(config.num_per_q[-1] * TRAIN_TEST_SPLIT)

        if not os.path.exists(mid_data_dir):
            os.makedirs(mid_data_dir)

        self.batch_size = opt.batch_size
        self.num_q = 1
        self.SCALE = SCALE
        self.input_func = sysbench_GET_INPUT
        self.cost_factor_dict = {}

        if opt.new_mid_data:

            self.input_func = sysbench_GET_INPUT

            self.grp_idxes = []
            self.num_grps = [0] * self.num_q

            data = []
            datas = {}
            all_groups, all_groups_test, all_groups_train = [], [], []

            if opt.new_data_structure:
                for root, dirs, files in os.walk(opt.data_dir):
                    for dir in dirs:
                        fname = root + "/" + dir + "/serverlog"
                        temp_data = self.get_all_plans(fname)

                        temp_data = cost_factor_one2one(opt, dir, temp_data)

                        for i in range(self.num_q):
                            if i not in datas.keys():
                                datas[i] = []
                            datas[i].extend(temp_data)
                with open(opt.data_structure + '/datas.pickle', 'wb') as f:
                    pickle.dump(datas, f)
            else:
                with open(opt.data_structure + '/datas.pickle', 'rb') as f:
                    datas = pickle.load(f)

            random.seed(1)
            for i in datas.keys():
                temp_data = datas[i]

                temp_data = random.sample(temp_data, int(self.num_sample_per_q*5/4))

                enum, num_grp = self.grouping(temp_data)
                groups = [[] for _ in range(num_grp)]
                for j, grp_idx in enumerate(enum):
                    groups[grp_idx].append(temp_data[j])
                all_groups += groups

                self.grp_idxes += enum
                self.num_grps[i] = num_grp
                data += temp_data

            self.datasize = len(data)

            print("Number of groups per query: ", self.num_grps)

            if not os.path.exists(mid_data_dir):
                os.mkdir(mid_data_dir)

            self.mean_range_dict = self.normalize(all_groups)

            self.table_list = []
            self.join_onehot = []
            self.source_list = []
            self.total_time = []
            self.total_cost = []

            for grp in all_groups:
                if grp == []:
                    continue
                new_samp_dict = self.get_input(grp)

                # self.table_list.extend([new_samp_dict["node_type"] for i in range(len(new_samp_dict["feat_vec"]))])
                self.join_onehot.extend([new_samp_dict["join_onehot"] for i in range(len(new_samp_dict["feat_vec"]))])
                self.source_list.extend(new_samp_dict["feat_vec"])
                self.total_time.extend(new_samp_dict["total_time"])
                self.total_cost.extend(new_samp_dict["total_cost"])


            # self.align_table()
            self.align()
            self.total_time, min_val, max_val = self.normalize_labels(self.total_time)


            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            torch.random.manual_seed(1)
            self.all_dataset = self.make_dataset(self.join_onehot, self.source_list, self.total_time)

            self.datasize = self.num_sample_per_q
            test_size = int(self.num_sample_per_q / 4)

            self.dataset, self.test_dataset = random_split(self.all_dataset, [self.datasize, test_size])

            self.dim_dict = {
                "min_val": min_val,
                "max_val": max_val,
                "table_list_len": len(rel_names),
                "maxlen_plan": len(self.source_list[0])
            }
            self.train_dataset = DataLoader(self.dataset, batch_size=opt.batch_size)

            with open(mid_data_dir + '/serialize_dim_dict.pickle', 'wb') as f:
                pickle.dump(self.dim_dict, f)
            with open(mid_data_dir + '/serialize_grp_idxes.pickle', 'wb') as f:
                pickle.dump(self.grp_idxes, f)
            with open(mid_data_dir + '/serialize_num_grps.pickle', 'wb') as f:
                pickle.dump(self.num_grps, f)
            with open(mid_data_dir + '/serialize_data.pickle', 'wb') as f:
                pickle.dump(self.dataset, f)
            with open(mid_data_dir + '/serialize_test_dataset.pickle', 'wb') as f:
                pickle.dump(self.test_dataset, f)
            with open(mid_data_dir + '/serialize_train_dataset.pickle', 'wb') as f:
                pickle.dump(self.train_dataset, f)
            with open(mid_data_dir + '/serialize_all_dataset.pickle', 'wb') as f:
                pickle.dump(self.all_dataset, f)
        else:
            with open(mid_data_dir + '/serialize_dim_dict.pickle', 'rb') as f:
                self.dim_dict = pickle.load(f)
            with open(mid_data_dir + '/serialize_grp_idxes.pickle', 'rb') as f:
                self.grp_idxes = pickle.load(f)
            with open(mid_data_dir + '/serialize_num_grps.pickle', 'rb') as f:
                self.num_grps = pickle.load(f)
            with open(mid_data_dir + '/serialize_data.pickle', 'rb') as f:
                self.dataset = pickle.load(f)
                self.datasize = len(self.dataset)
            with open(mid_data_dir + '/serialize_test_dataset.pickle', 'rb') as f:
                self.test_dataset = pickle.load(f)
            with open(mid_data_dir + '/serialize_train_dataset.pickle', 'rb') as f:
                self.train_dataset = pickle.load(f)
            with open(mid_data_dir + '/serialize_all_dataset.pickle', 'rb') as f:
                self.all_dataset = pickle.load(f)

    def align_table(self):
        max_len = max([len(list) for list in self.table_list])
        for idx, table in enumerate(self.table_list):
            self.table_list[idx] = np.vstack((table, np.zeros((max_len - len(table), len(all_dicts)))))

    def align(self):
        max_len = max([len(list) for list in self.source_list])
        for idx, source in enumerate(self.source_list):
            self.source_list[idx] = np.concatenate((source, np.zeros(max_len - len(source))), axis=0)

    def normalize_labels(self, labels):
        labels = np.array([np.log(float(l)) for l in labels if l != 0])
        min_val = labels.min()
        max_val = labels.max()
        labels_norm = (labels - min_val) / (max_val - min_val)
        labels_norm = np.minimum(labels_norm, 1)
        labels_norm = np.maximum(labels_norm, 0)
        return labels_norm, min_val, max_val

    def make_dataset(self, join_onehots, predicates, labels):
        # table_masks = []
        # table_tensors = []
        # maxlen_tables = max([len(t) for t in tables])
        # for table in tables:
        #     table_tensor = np.vstack(table)
        #     num_pad = maxlen_tables - table_tensor.shape[0]
        #     table_mask = np.ones_like(table_tensor).mean(1, keepdims=True)
        #     table_tensor = np.pad(table_tensor, ((0, num_pad), (0, 0)), 'constant')
        #     table_mask = np.pad(table_mask, ((0, num_pad), (0, 0)), 'constant')
        #     table_tensors.append(np.expand_dims(table_tensor, 0))
        #     table_masks.append(np.expand_dims(table_mask, 0))
        # table_tensors = np.vstack(table_tensors)
        # table_tensors = torch.FloatTensor(table_tensors)
        # table_masks = np.vstack(table_masks)
        # table_masks = torch.FloatTensor(table_masks)

        join_onehot_masks = []
        join_onehot_tensors = []
        for join_onehot in join_onehots:
            join_onehot_tensor = np.vstack([join_onehot])
            join_onehot_tensor = np.array(join_onehot_tensor)
            join_onehot_mask = np.ones_like(join_onehot_tensor).mean(1, keepdims=True)
            join_onehot_tensors.append(np.expand_dims(join_onehot_tensor, 0))
            join_onehot_masks.append(np.expand_dims(join_onehot_mask, 0))

        join_onehot_tensors = np.vstack(join_onehot_tensors)
        join_onehot_tensors = torch.FloatTensor(join_onehot_tensors)
        join_onehot_masks = np.vstack(join_onehot_masks)
        join_onehot_masks = torch.FloatTensor(join_onehot_masks)

        predicate_masks = []
        predicate_tensors = []
        for predicate in predicates:
            predicate_tensor = np.vstack([predicate])
            predicate_tensor = np.array(predicate_tensor)
            predicate_mask = np.ones_like(predicate_tensor).mean(1, keepdims=True)
            predicate_tensors.append(np.expand_dims(predicate_tensor, 0))
            predicate_masks.append(np.expand_dims(predicate_mask, 0))

        predicate_tensors = np.vstack(predicate_tensors)
        predicate_tensors = torch.FloatTensor(predicate_tensors)
        predicate_masks = np.vstack(predicate_masks)
        predicate_masks = torch.FloatTensor(predicate_masks)

        target_tensor = torch.FloatTensor(labels)

        # tables, predicates, targets = table_tensors.cuda(), predicate_tensors.cuda(), target_tensor.cuda()
        # table_masks, predicate_masks = table_masks.cuda(), predicate_masks.cuda()
        #
        # table_tensors, predicate_tensors, target_tensor = Variable(tables), Variable(predicates), Variable(targets)
        # table_masks, predicate_masks = Variable(table_masks), Variable(predicate_masks)

        join_onehots, predicates, targets = join_onehot_tensors.cuda(), predicate_tensors.cuda(), target_tensor.cuda()
        join_onehot_masks, predicate_masks = join_onehot_masks.cuda(), predicate_masks.cuda()

        join_onehot_tensors, predicate_tensors, target_tensor = Variable(join_onehots), Variable(predicates), Variable(
            targets)
        join_onehot_masks, predicate_masks = Variable(join_onehot_masks), Variable(predicate_masks)

        trainX = torch.concatenate((join_onehot_tensors, predicate_tensors, join_onehot_masks, predicate_masks),dim=2)

        # return dataset.TensorDataset(table_tensors, predicate_tensors, table_masks, predicate_masks, target_tensor)
        # return dataset.TensorDataset(join_onehot_tensors, predicate_tensors, join_onehot_masks, predicate_masks,
        #                              target_tensor)
        return dataset.TensorDataset(trainX, target_tensor)

    def normalize(self, all_groups_train):  # compute the mean and std vec of each operator
        """
            For each operator, normalize each input feature to have a mean of 0 and maximum of 1

            Returns:
            - mean_range_dict: a dictionary where the keys are the Operator Names and the values are 2-tuples (mean_vec, max_vec):
                -- mean_vec : a vector of mean values for input features of this operator
                -- max_vec  : a vector of max values for input features of this operator
        """
        feat_vec_col = {operator: [] for operator in all_dicts}

        def parse_input(data):
            feat_vec = [np.hstack((self.input_func[jss["Node Type"]](jss))) for
                        jss
                        in data]

            if 'Plans' in data[0]:
                for i in range(len(data[0]['Plans'])):
                    parse_input([jss['Plans'][i] for jss in data])
            feat_vec_col[data[0]["Node Type"]].append(np.array(feat_vec).astype(np.float32))

        def cmp_mean_range(feat_vec_lst):
            if len(feat_vec_lst) == 0:
                return (0, 1)
            else:
                total_vec = np.concatenate(feat_vec_lst)
                return (np.mean(total_vec, axis=0),
                        np.max(total_vec, axis=0) + np.finfo(np.float32).eps)

        [parse_input(grp) for grp in all_groups_train if grp != []]

        mean_range_dict = {operator: cmp_mean_range(feat_vec_col[operator]) \
                           for operator in all_dicts}
        return mean_range_dict
    
    def get_all_plans(self, fname):
        """
            Parse from data file

            Args:
            - fname: the name of data file to be parsed

            Returns:
            - jss: a sanitized list of dictionary, one per query, parsed from the input data file
        """
        pattern_num = re.compile(r'\d+.?\d*')
        jss = []

        with open(fname, "r") as f:
            lines = [line for line in f.readlines()]
            lineid = 0
            while lineid < len(lines):
                if ' CST [' not in lines[lineid]:
                    lineid += 1
                    continue

                while lineid < len(lines) and ' CST [' in lines[lineid]:
                    if 'duration' in lines[lineid]:
                        duration = pattern_num.findall(lines[lineid])[-1]
                    else:
                        lineid += 1
                        continue
                    plan_strs = []
                    lineid += 1
                    while lineid < len(lines) and ' CST [' not in lines[lineid]:
                        plan_strs.append(lines[lineid])
                        lineid += 1
                    if plan_strs != []:
                        plan_obj = json.loads(s=''.join(plan_strs))
                        plan_obj['Plan']['Planning Time'] = float(duration) - plan_obj['Plan']['Actual Total Time']
                        jss.append(plan_obj['Plan'])
        return jss

    def grouping(self, data):
        """
            Groups the queries by their query plan structure

            Args:
            - data: a list of dictionaries, each being a query from the dataset

            Returns:
            - enum    : a list of same length as data, containing the group indexes for each query in data
            - counter : number of distinct groups/templates
        """

        def hash(plan_dict):
            res = plan_dict['Node Type']
            if 'Plans' in plan_dict:
                for chld in plan_dict['Plans']:
                    res += hash(chld)
            return res

        counter = 0
        string_hash = []
        enum = []
        for plan_dict in data:
            string = hash(plan_dict)
            try:
                idx = string_hash.index(string)
                enum.append(idx)
            except:
                idx = counter
                counter += 1
                enum.append(idx)
                string_hash.append(string)

        assert (counter > 0)
        return enum, counter

    def get_input(self, data):  # Helper for sample_data
        """
            Vectorize the input of a list of queries that have the same plan structure (of the same template/group)

            Args:
            - data: a list of plan_dict, each plan_dict correspond to a query plan in the dataset;
                    requires that all plan_dicts is of the same query template/group

            Returns:
            - new_samp_dict: a dictionary, where each level has the following attribute:
                -- node_type     : name of the operator
                -- subbatch_size : number of queries in data
                -- feat_vec      : a numpy array of shape (batch_size x feat_dim) that's
                                   the vectorized inputs for all queries in data
                -- children_plan : list of dictionaries with each being an output of
                                   a recursive call to get_input on a child of current node
                -- total_time    : a vector of prediction target for each query in data
                -- is_subplan    : if the queries are subplans
        """

        def get_op_one_hot(op):
            arr = [0] * len(all_dicts)
            arr[all_dicts.index(op)] = 1
            return np.array(arr)

        # Merge Cond | Hash Cond
        def get_join_one_hot(cond):
            arr = [0] * len(rel_names)
            for idx, rel in enumerate(rel_names):
                if rel in cond:
                    arr[idx] = 1
            return np.array(arr)

        def join_combine(array1, array2):
            for idx, arr1 in enumerate(array1):
                if arr1 == 1:
                    array2[idx] = 1
            return array2

        new_samp_dict = {}
        new_samp_dict["node_type"] = get_op_one_hot(data[0]["Node Type"])

        if data[0]["Node Type"] == 'Hash Join':
            new_samp_dict["join_onehot"] = get_join_one_hot(data[0]["Hash Cond"])
        elif data[0]["Node Type"] == 'Merge Join':
            new_samp_dict["join_onehot"] = get_join_one_hot(data[0]["Merge Cond"])
        else:
            new_samp_dict["join_onehot"] = np.array([0] * len(rel_names))

        new_samp_dict["subbatch_size"] = len(data)

        feat_vec = np.array(
            [self.input_func[jss["Node Type"]](jss) for jss in data])

        # normalize feat_vec
        feat_vec = (feat_vec -
                    self.mean_range_dict[data[0]["Node Type"]][0]) \
                   / self.mean_range_dict[data[0]["Node Type"]][1]
        # feat_vec = normalizer.transform(feat_vec)

        feat_vec = np.array(
            [np.hstack(
                (new_samp_dict["node_type"], feat))
                for feat in feat_vec])

        total_time = [jss['Actual Total Time'] for jss in data]
        total_cost = [jss['Total Cost'] for jss in data]

        if 'Planning Time' in data[0]:
            plan_time = [jss['Planning Time'] for jss in data]
            new_samp_dict["plan_time"] = np.array(plan_time).astype(np.float32) / self.SCALE

        if 'Plans' in data[0]:
            for i in range(len(data[0]['Plans'])):
                child_plan_dict = self.get_input([jss['Plans'][i] for jss in data])
                new_samp_dict["node_type"] = np.vstack((child_plan_dict["node_type"], new_samp_dict["node_type"]))
                new_samp_dict["join_onehot"] = join_combine(child_plan_dict["join_onehot"],
                                                            new_samp_dict["join_onehot"])
                feat_vec = np.hstack((child_plan_dict["feat_vec"], feat_vec))

        new_samp_dict["feat_vec"] = np.array(feat_vec).astype(np.float32)
        new_samp_dict["total_time"] = np.array(total_time).astype(np.float32) / self.SCALE
        new_samp_dict["total_cost"] = np.array(total_cost).astype(np.float32) / self.SCALE

        if 'Subplan Name' in data[0]:
            new_samp_dict['is_subplan'] = True
        else:
            new_samp_dict['is_subplan'] = False

        return new_samp_dict

    ###############################################################################
    #       Sampling subbatch data from the dataset; total size is batch_size     #
    ###############################################################################
    def sample_data(self, batch_size):
        """
            Randomly sample a batch of data points from the train dataset

            Returns:
            - parsed_input: a list of dictionaries with inputs vectorized by get_input,
                            each dictionary contains all samples in the batch that comes from this group
        """
        # dataset: all queries used in training

        

        return self.dataset

    def add_knobs(self, data, knobs):
        data['knob_values'] = knobs

        if 'Plans' in data:
            for sub_tree in data['Plans']:
                self.add_knobs(sub_tree, knobs)

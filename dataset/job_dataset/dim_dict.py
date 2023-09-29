from dataset.sysbench_dataset.attr_rel_dict import rel_names, index_names, join_types, parent_rel_types, \
    aggreg_strats

max_num_attr = 4  # 3*4 min_vec + med_vec + max_vec
# 'c_custkey','c_name','c_address','c_nationkey','c_phone','c_acctbal','c_mktsegment','c_comment'import config
basics = 3  # get_basics(plan_dict)
num_rel = len(rel_names)
num_index = len(index_names)

# 1,2,6,3,3,1
other = ['Scan Direction', 'sort_meth', 'join_types', 'parent_rel_types', 'aggreg_strats', 'Parallel Aware']
all = ['basics', 'num_rel', 'max_num_attr3', 'num_index', 'sort_key', other, 'len(dic_factor["op"])', 'child']

#                 [basics,num_rel, max_num_attr * 3,len(dic_factor["Seq Scan"])]
dim_dict = {'Seq Scan': [3, 1, 12, 0, [0, 0, 0, 0, 0, 0], 2, 0],
                 # [basics,num_rel, max_num_attr * 3, 1('Scan Direction'), len(dic_factor["Index Scan"])]
                 'Index Scan': [3, 1, 12, 0, 0, [1, 0, 0, 0, 0, 0], 2, 0],
                 # [basics, num_rel, max_num_attr * 3, num_index, 1('Scan Direction'), len(dic_factor["Index Only Scan"])],
                 'Index Only Scan': [3, 1, 12, 1, 0, [1, 0, 0, 0, 0, 0], 2, 0],
                 # [basics, num_rel, max_num_attr * 3, len(dic_factor["Bitmap Heap Scan"]), 32],
                 'Bitmap Heap Scan': [3, 1, 12, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 # [basics, num_index, len(dic_factor["Bitmap Index Scan"])],
                 'Bitmap Index Scan': [3, 0, 0, 1, 0, [0, 0, 0, 0, 0, 0], 2, 0],
                 # [basics, 128, len(sort_algos), len(dic_factor["Sort"]), 32],
                 'Sort': [3, 0, 0, 0, 128, [0, 2, 0, 0, 0, 0], 2, 32],
                 # [basics + 1 + 32 + len(dic_factor["Hash"])],
                 'Hash': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 1], 2, 32],
                 # [basics + len(join_types) + len(parent_rel_types) + 32 * 2 + len(dic_factor["Hash Join"])],
                 'Hash Join': [3, 0, 0, 0, 0, [0, 0, 6, 3, 0, 1], 2, 32 * 2],
                 # [basics + len(join_types) + len(parent_rel_types) + 32 * 2 + len(dic_factor["Merge Join"]),
                 'Merge Join': [3, 0, 0, 0, 0, [0, 0, 6, 3, 0, 1], 2, 32 * 2],
                 # [basics + len(aggreg_strats) + 1('') + 32 + len(dic_factor["Aggregate"]),]
                 'Aggregate': [3, 0, 0, 0, 0, [0, 0, 0, 0, 3, 1], 2, 32],
                 # [32 * 2 + len(join_types) + basics + len(dic_factor["Nested Loop"])]
                 'Nested Loop': [3, 0, 0, 0, 0, [0, 0, 6, 0, 0, 0], 4, 32 * 2],
                 # [32 + basics + len(dic_factor["Limit"])]
                 'Limit': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 # [32 + basics + len(dic_factor["Subquery Scan"])]
                 'Subquery Scan': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 # [32 + basics + len(dic_factor["Materialize"])]
                 'Materialize': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 # [32 + basics + len(dic_factor["Gather Merge"])]
                 'Gather Merge': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 # [32 + basics + len(dic_factor["Gather"])]
                 'Gather': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 # [32 * 2 + basics + len(dic_factor["BitmapAnd"])]
                 'BitmapAnd': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 # [32 + basics + len(dic_factor["Memoize"])]
                 'Memoize': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 # [num_rel + 32 + basics + len(dic_factor["ModifyTable"])],
                 'ModifyTable':  [3, 1, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 # [basics + len(dic_factor["Result"])],
                 'Result': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 0],
                 # [basics + len(dic_factor["LockRows"])],
                 'LockRows': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 'Append': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 'Unique': [3, 0, 0, 0, 0, [0, 0, 0, 0, 0, 0], 2, 32],
                 }


def get_feature(op, tuple):
    len = 0
    ans = ['0','0']
    for idx, temp in enumerate(dim_dict[op]):
        if idx != 5:
            if temp != 0:
                len += temp
            for idz, t in enumerate(tuple):
                if int(t) < len:
                    ans[idz] = all[idx]
                    tuple[idz] = 100000
        else:
            for idy, i in enumerate(temp):
                if i != 0:
                    len += i
                else:
                    continue
                for idz, t in enumerate(tuple):
                    if int(t) < len:
                        ans[idz] = other[idy]
                        tuple[idz] = 100000
    return ans

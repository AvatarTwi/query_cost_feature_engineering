# all operators used in tpc-h
import os
import random

all_dicts = ['Aggregate', 'Gather Merge', 'Sort', 'Seq Scan', 'Index Scan',
             'Index Only Scan', 'Bitmap Heap Scan', 'Bitmap Index Scan',
             'Limit', 'Hash Join', 'Hash', 'Nested Loop', 'Materialize',
             'Merge Join', 'Subquery Scan', 'Gather', 'BitmapAnd', 'Memoize']

join_types = ['semi', 'inner', 'anti', 'full', 'right', 'left']

parent_rel_types = ['inner', 'outer', 'subquery']

sort_algos = ['quicksort', 'top-n heapsort']

aggreg_strats = ['plain', 'sorted', 'hashed']

rel_names = ['customer', 'lineitem', 'nation', 'orders', 'part', 'partsupp',
             'region', 'supplier']

index_names = ['c_ck', 'c_nk', 'p_pk', 's_sk', 's_nk', 'ps_pk', 'ps_sk',
               'ps_pk_sk', 'ps_sk_pk', 'o_ok', 'o_ck', 'o_od', 'l_ok', 'l_pk',
               'l_sk', 'l_sd', 'l_cd', 'l_rd', 'l_pk_sk', 'l_sk_pk', 'n_nk',
               'n_rk', 'r_rk']

rel_attr_list_dict = \
    {
        'customer':
            ['c_custkey',
             'c_name',
             'c_address',
             'c_nationkey',
             'c_phone',
             'c_acctbal',
             'c_mktsegment',
             'c_comment'],
        'lineitem':
            ['l_orderkey',
             'l_partkey',
             'l_suppkey',
             'l_linenumber',
             'l_quantity',
             'l_extendedprice',
             'l_discount',
             'l_tax',
             'l_returnflag',
             'l_linestatus',
             'l_shipdate',
             'l_commitdate',
             'l_receiptdate',
             'l_shipinstruct',
             'l_shipmode',
             'l_comment'],
        'nation':
            ['n_nationkey',
             'n_name',
             'n_regionkey',
             'n_comment'],
        'orders':
            ['o_orderkey',
             'o_custkey',
             'o_orderstatus',
             'o_totalprice',
             'o_orderdate',
             'o_orderpriority',
             'o_clerk',
             'o_shippriority',
             'o_comment'],
        'part':
            ['p_partkey',
             'p_name',
             'p_mfgr',
             'p_brand',
             'p_type',
             'p_size',
             'p_container',
             'p_retailprice',
             'p_comment'],
        'partsupp':
            ['ps_partkey',
             'ps_suppkey',
             'ps_availqty',
             'ps_supplycost',
             'ps_comment'],
        'region':
            ['r_regionkey',
             'r_name',
             'r_comment'],
        'supplier':
            ['s_suppkey',
             's_name',
             's_address',
             's_nationkey',
             's_phone',
             's_acctbal',
             's_comment']
    }

attr_val_dict = {'med': {'customer': [7500.0, 0, 0, 12.0, 0, 4404.87, 0, 0],
                         'lineitem': [300486.0, 10003.0, 501.0, 3.0, 26.0, 34461.75, 0.05, 0.04, 0, 0, 0, 0, 0, 0, 0,
                                      0], 'nation': [12.0, 0, 2.0, 0],
                         'orders': [300000.0, 7484.0, 0, 136058.42, 0, 0, 0, 0.0, 0],
                         'part': [10000.0, 0, 0, 0, 0, 25.0, 0, 1409.49, 0],
                         'partsupp': [10000.0, 500.0, 4995.0, 498.72, 0], 'region': [2.0, 0, 0],
                         'supplier': [500.0, 0, 0, 12.0, 0, 4422.77, 0]},
                 'min': {'customer': [1.0, 0, 0, 0.0, 0, -999.95, 0, 0],
                         'lineitem': [1.0, 1.0, 1.0, 1.0, 1.0, 901.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'nation': [0.0, 0, 0.0, 0], 'orders': [1.0, 1.0, 0, 833.4, 0, 0, 0, 0.0, 0],
                         'part': [1.0, 0, 0, 0, 0, 1.0, 0, 901.0, 0], 'partsupp': [1.0, 1.0, 1.0, 1.01, 0],
                         'region': [0.0, 0, 0], 'supplier': [1.0, 0, 0, 0.0, 0, -966.2, 0]},
                 'max': {'customer': [15000.0, 0, 0, 24.0, 0, 9999.72, 0, 0],
                         'lineitem': [600000.0, 20000.0, 1000.0, 7.0, 50.0, 95949.5, 0.1, 0.08, 0, 0, 0, 0, 0, 0, 0, 0],
                         'nation': [24.0, 0, 4.0, 0], 'orders': [600000.0, 14999.0, 0, 479129.21, 0, 0, 0, 0.0, 0],
                         'part': [20000.0, 0, 0, 0, 0, 50.0, 0, 1918.99, 0],
                         'partsupp': [20000.0, 1000.0, 9999.0, 999.99, 0], 'region': [4.0, 0, 0],
                         'supplier': [1000.0, 0, 0, 24.0, 0, 9993.46, 0]}}


class TPCH:

    @staticmethod
    def random_table_attr_value(table=0, attr=0):
        while True:
            try:
                if table == 0:
                    table = random.choice(list(rel_attr_list_dict.keys()))
                attr = random.choice(rel_attr_list_dict[table])
                id = rel_attr_list_dict[table].index(attr)
                min = int(attr_val_dict['min'][table][id])
                max = int(attr_val_dict['max'][table][id])
                val = random.randint(min, max)
                if attr_val_dict['min'][table][id] != 0 or attr_val_dict['max'][table][id] != 0:
                    break
            except:
                continue

        return table, attr, val

    @staticmethod
    def random_get_join(table1=0):
        while True:
            if table1 == 0:
                table1 = random.choice(list(rel_attr_list_dict.keys()))

            for attr in rel_attr_list_dict[table1]:
                for table2 in rel_attr_list_dict.keys():
                    if table2 == table1:
                        continue
                    if attr not in rel_attr_list_dict[table2]:
                        continue
                    else:
                        break
                break
            if attr in rel_attr_list_dict[table2]:
                break
        attr1 = attr
        attr2 = attr
        return table1, table2, attr1, attr2

    @staticmethod
    def collect_table_attr(root='tpch_queries'):
        file_dirs = [os.path.join(root, file) for file in os.listdir(root)]
        dict = {}
        joinon = set()
        groupby = {}
        orderby = {}
        for file in file_dirs:
            with open(file, 'r+') as f:
                file_ = f.read().split("from")[-1]
                group_by = file_.split("group by")[-1].split("order by")[0]
                order_by = file_.split("order by")[-1]

                contain = [file_, group_by, order_by]
                dicts = [dict, groupby, orderby]

                rels = []

                for i, a in enumerate(contain):
                    for rel in rel_attr_list_dict.keys():
                        for attr in rel_attr_list_dict[rel]:
                            if attr in a:
                                if rel not in dicts[i].keys():
                                    dicts[i][rel] = set()
                                dicts[i][rel].add(attr)
                                if 'key' in attr:
                                    rels.append(rel)
                for rel1 in rels:
                    for rel2 in rels:
                        if rel1 != rel2:
                            for attr1 in rel_attr_list_dict[rel1]:
                                if attr1 in file_ and 'key' in attr1:
                                    break
                            for attr2 in rel_attr_list_dict[rel2]:
                                if attr2 in file_ and attr2.split("_")[-1] in attr1.split("_")[-1]:
                                    joinon.add((rel1, rel2,attr1,attr2))
                                    break

        return dict, joinon, groupby, orderby


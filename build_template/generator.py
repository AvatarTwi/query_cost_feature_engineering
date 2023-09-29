import collections
import os
import random

from build_template.attr_rel_dic_tpch import TPCH
from build_template.attr_rel_dict_job import JOB

explain = "EXPLAIN (ANALYZE,VERBOSE,COSTS,BUFFERS,TIMING,SUMMARY,FORMAT JSON) "
select = " SELECT {} "
join = " JOIN {} "
on = " ON {} = {} "
from_t = " FROM {} "
where = " WHERE "
group_by = " GROUP BY {} "
order_by = " ORDER BY {} "
between = " {} BETWEEN {} AND {} "
in_ = " {} IN ({}) "
eq = " {} = {} "
ge = " {} >= {} "
le = "  {} >= {} "
interval = " {} <= date '1998-12-01' - interval '{}' day "
notlike = " {} NOT LIKE '{}' "
like = " {} LIKE '{}' "


def where_with_value(rel, attr, val):
    where_type = []
    where_types = []

    if "%" in attr:
        where_type.append(like.format(attr, val))
        where_type.append(notlike.format(attr, val))
    elif type(val) != str:
        where_type.append(ge.format(attr, val))
        where_type.append(le.format(attr, val))

    where_types.append(where_type[random.randint(0, len(where_type) - 1)])

    return where + " OR ".join(where_types)


def where_without_value():
    where_type = []
    where_types = []
    where_type.append(ge)
    where_type.append(le)

    where_types.append(where_type[random.randint(0, len(where_type) - 1)])

    return where + " OR ".join(where_types)


def agg(method):
    dict, joinon, groupby, orderby = method.collect_table_attr()

    if groupby == {}:
        return []
    else:
        sqls = []
        for rel in groupby.keys():
            for attr in groupby[rel]:
                sql = explain + select + from_t + "where_without_value"
                sql = sql.format("COUNT(*)", rel, attr)
                sqls.append((rel, sql))
        return sqls


def hashjoin(method):
    dict, joinon, groupby, orderby = method.collect_table_attr()

    if joinon == {}:
        return []
    else:
        sqls = []
        for rel1, rel2, attr1, attr2 in joinon:
            sql = explain + select + from_t + join + on + "where_without_value"
            sql = sql.format("*", rel1, rel2, rel1 + "." + attr1, rel2 + "." + attr2)
            sqls.append((rel1, sql))
        return sqls


def mergejoin(method):
    dict, joinon, groupby, orderby = method.collect_table_attr()

    if joinon == {}:
        return []
    else:
        sqls = []
        for rel1, rel2, attr1, attr2 in joinon:
            sql = explain + select + from_t + join + on + "where_without_value"
            rel, attr, val = method.random_table_attr_value(rel1)
            sql = sql.format("*", rel1, rel2, rel1 + "." + attr1, rel2 + "." + attr2, rel + "." + attr)
            sqls.append((rel, sql))
        return sqls


def seqscan(method):
    dict, joinon, groupby, orderby = method.collect_table_attr()

    if dict == {}:
        return []
    else:
        sqls = []
        for rel in dict.keys():
            sql = explain + select + from_t + "where_without_value"
            sql = sql.format("*", rel)
            sqls.append((rel, sql))
        return sqls


def sort(method):
    dict, joinon, groupby, orderby = method.collect_table_attr()

    if orderby == {}:
        return []
    else:
        sqls = []
        for rel in orderby.keys():
            for attr in orderby[rel]:
                sql = explain + select + from_t + "where_without_value" + order_by
                sql = sql.format("*", rel, attr)
                sqls.append((rel, sql))
        return sqls


OP_INPUT_DICT = {
    "agg": agg,
    "hashjoin": hashjoin,
    "mergejoin": mergejoin,
    "seqscan_index": seqscan,
    "sort": sort
}

OP_INPUT_DICT = collections.defaultdict(lambda: agg, OP_INPUT_DICT)
METHOD = {
    'JOB': JOB,
    "TPCH": TPCH
}

if __name__ == '__main__':
    root_dir = "sqls"
    for ds in METHOD.keys():
        if not os.path.exists(root_dir + "/simp_" + ds + "_queries"):
            os.makedirs(root_dir + "/simp_" + ds + "_queries")

        sqls_dict = {}
        ops = 0
        for idx, key in enumerate(OP_INPUT_DICT.keys()):
            sqls = OP_INPUT_DICT[key](METHOD[ds])
            if sqls:
                ops += 1
        for idx, key in enumerate(OP_INPUT_DICT.keys()):
            print(key)
            sqls = OP_INPUT_DICT[key](METHOD[ds])
            if not sqls:
                continue
            sqls_dict[key] = sqls

        count = 0
        for key in sqls_dict.keys():
            for sqls in sqls_dict[key]:
                for i in range(1):
                    with open(root_dir + "/simp_" + ds + "_queries/" + str(count + 1) + ".sql", "wb") as f:
                        count += 1
                        rel, sql = sqls
                        rel, attr, val = METHOD[ds].random_table_attr_value(rel)
                        sql = sql.replace("where_without_value", where_without_value().format(rel + "." + attr, val))
                        f.write(str.encode(sql, 'utf-8'))

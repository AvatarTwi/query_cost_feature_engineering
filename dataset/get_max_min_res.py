import pickle
from job_dataset.attr_rel_dict import *

def build_attr_queries():
    with open('attr_queries_min.sql', 'w+') as f:
        for table, cols in rel_attr_list_dict.items():
            min = []
            for col in cols:
                # min.append("min(" + col + ") as " + col)
                min.append("min(" + col + ")")

            f.write(
                "select \'\"{}\":[\', \'\"\'||{}||\'\"],\' from {};\n".format(table, "||\'\",\',\'\"\'||".join(min),
                                                                              table))

    with open('attr_queries_max.sql', 'w+') as f:
        for table, cols in rel_attr_list_dict.items():
            max = []
            for col in cols:
                max.append("max(" + col + ")")

            f.write(
                "select \'\"{}\":[\', \'\"\'||{}||\'\"],\' from {};\n".format(table, "||\'\",\',\'\"\'||".join(max),
                                                                              table))

    with open('attr_queries_med.sql', 'w+') as f:
        for table, cols in rel_attr_list_dict.items():
            med = []
            for col in cols:
                med.append("\'\"\'||percentile_disc(0.5) within group (order by {}) ||\'\",\'".format(col))
            f.write(
                "select \'\"{}\":[\',".format(table) + ",".join(med) + "||\'],\' from {}".format(
                    table) + ";\n")


build_attr_queries()

def convert(input):
    try:
        res = float(input)
    except:
        res = 0
    return res


def pickle_dump():
    attr_val_dict = {}
    for a in attr.keys():
        attr_val_dict[a] = {}
        for b in attr[a].keys():
            attr_val_dict[a][b] = [convert(v) for v in attr[a][b]]
    print(attr_val_dict)

    # with open('dataset/oltp_dataset/attr_val_dict.pickle', 'wb') as f:
    #     pickle.dump(attr_val_dict, f)


def pickle_dump1():
    with open('oltp_dataset/attr_val_dict.pickle', 'rb') as f:
        attr_val_dict = pickle.load(f)

    attr_val_dict['max']["bmsql_warehouse"] = ["10", "0.1658", "4899705.01", "895411111", "KU", "WMLtrxN3EowPFGSykL",
                                               "VGzaaWp832mxe4TL3vJ",
                                               "X965aEwmhkjN8a7Tovl", "Ve17t8"]
    attr_val_dict['med']["bmsql_warehouse"] = ["5", "0.0991", "4515629.72", "543011111", "EO", "MQZtvYwNb2x",
                                               "Pj0HscFCjZ9pcB3d",
                                               "FyhY0C4WDI7", "LTFM08Q0", ]
    attr_val_dict['min']["bmsql_warehouse"] = ["1", "0.0253", "4271920.79", "222811111", "BH", "anAAt23ZYlL4uyk",
                                               "fJBfJW8FoUY",
                                               "aBrhz9MjElLrUAFbc", "gbSJsIIUab"]

    for a in attr_val_dict.keys():
        attr_val_dict[a]["bmsql_warehouse"] = [convert(value) for value in attr_val_dict[a]["bmsql_warehouse"]]

    with open('oltp_dataset/attr_val_dict.pickle', 'wb') as f:
        pickle.dump(attr_val_dict, f)

# pickle_dump1()

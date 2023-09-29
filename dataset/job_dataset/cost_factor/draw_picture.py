import os
import pickle

from matplotlib import pyplot as plt
from matplotlib.pyplot import *
from dataset.postgres_tpch_dataset.attr_rel_dict import all_dicts

# 设置字体大小
rc('font', size=15)


def translate(str):
    return sum([ord(i) for i in str])


def draw(errors):
    ops = all_dicts

    # 设置画布大小
    fig = plt.figure(figsize=(40, 20))

    fig.subplots_adjust(left=0.035, right=0.980,
                        top=0.9, bottom=0.150,
                        wspace=0.5, hspace=0.3)

    count = 1

    for i in range(len(ops)):
        ax = fig.add_subplot(2, 10, count)
        count += 1

        temp_error = []
        x_label = []
        for key in errors.keys():
            x_label.append(key)
            temp_error.append(errors[key][ops[i]])

        pattern = re.compile(r"\w+")
        barlist = ax.bar(x_label, temp_error)
        for idx, b in enumerate(barlist):
            barlist[idx].set_color(
                "#" + str(sum([ord(i) * 5 for i in re.findall(pattern, x_label[idx])[0]])) + "FF")

        for a, b in enumerate(temp_error):  # 柱子上的数字显示
            plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=6)

        plt.title(ops[i])

        # y轴值域
        # plt.ylim(0, 1000)
        plt.xticks(x_label, rotation=45, fontsize=12)
        # 设置y轴标签名
        plt.ylabel("Error")

    time = []
    x_label = []
    ax = fig.add_subplot(2, 10, count)
    for key in errors.keys():
        x_label.append(key)
        time.append(int(errors[key]["total_time"]))

        ax.bar(x_label, time)

        for a, b in enumerate(time):  # 柱子上的数字显示
            plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=6)
        plt.title("Model Run Time")
        plt.xticks(x_label, rotation=45, fontsize=12)

    return plt


def fig1(root_dir, exp):
    cost_factor_dict_dir = []
    errors = {}

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if 'cost_factor' in file and 'error' in file:
                if 'exp' + str(exp) in file:
                    cost_factor_dict_dir.append(file)

    for dir in cost_factor_dict_dir:
        with open(root_dir + dir, 'rb') as f:
            model_name = dir \
                .replace("cost_factor_dict", "") \
                .replace("error", "") \
                .replace('exp' + str(exp), "") \
                .replace('_', "") \
                .replace(".pickle", "")
            errors[model_name] = pickle.load(f)

    print(errors)
    plt = draw(errors)

    plt.savefig("fig/data_size" + str(int(12 * 12 * 12 * 0.8 ** (exp - 1))) + ".png")


def fig2(root_dir):
    cost_factor_dict_dir = []
    errors = {}

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if 'cost_factor' in file and 'error' in file:
                cost_factor_dict_dir.append(file)

    for dir in cost_factor_dict_dir:
        with open(root_dir + dir, 'rb') as f:
            pattern = re.compile(r'\d+')
            m = pattern.findall(dir)

            model_name = dir \
                .replace("cost_factor_dict_", "") \
                .replace("_error", "") \
                .replace("exp" + m[0], "size_" + str(int(12 * 12 * 12 * 0.8 ** (int(m[0]) - 1)))) \
                .replace(".pickle", "")

            error = pickle.load(f)

            for e in error.keys():
                if e == "total_time":
                    continue
                error[e] = error[e] / int(12 * 12 * 12 * 0.8 ** (int(m[0]) - 1))

            errors[model_name] = error

    print(errors)
    plt = draw(errors)

    plt.savefig("fig/compare.png")


if __name__ == '__main__':

    exps = [1]
    for exp in exps:
        fig1("../../../v6/tpch/data_structure/", exp)
    fig2("../../../v6/tpch/data_structure/")

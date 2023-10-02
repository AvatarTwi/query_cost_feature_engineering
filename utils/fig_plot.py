import os
import pickle
import random
from matplotlib import pyplot as plt

from utils.metric import Metric
from matplotlib.pyplot import *

from utils.plot_util import name_align, sort

plt.rcParams['font.size'] = 15


def plot_scatter_err(dict_alls, save_to_dir):
    fig = plt.figure(figsize=(5 * 2, 5 * (len(dict_alls.keys()))))

    fig.subplots_adjust(left=0.1, right=0.950, top=0.9, bottom=0.10,
                        wspace=0.2, hspace=0.4)

    fig.suptitle('sub', y=1.1, size=20)

    count = 1

    for idx, k in enumerate(dict_alls.keys()):
        ax = fig.add_subplot(len(dict_alls.keys()), 2, count)
        count += 1
        dict_all = dict_alls[k]

        max_value = max([dict_all[key]['q_error'] for key in dict_all.keys() if dict_all[key]['q_error'] < 2]) * 1.05

        min_Time = min([dict_all[key]['avg_train_cost'] for key in dict_all.keys()]) * 0.95
        max_Time = max([dict_all[key]['avg_train_cost'] for key in dict_all.keys()]) * 1.01
        min_Err = min([dict_all[key]['q_error'] for key in dict_all.keys()]) * 0.95
        max_Err = min(max_value, max([dict_all[key]['q_error'] for key in dict_all.keys()])) * 1.01

        random.seed(idx)
        for idy, key in enumerate(dict_all.keys()):
            ax.scatter(
                dict_all[key]['avg_train_cost'],
                min(max_value, dict_all[key]['q_error']),
                marker="o",
                c="#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]),
                label=k + "/" + key
            )
            ax.annotate(str(round(dict_all[key]['q_error'], 3)),
                        xy=(dict_all[key]['avg_train_cost'], min(max_value, dict_all[key]['q_error'])),
                        xytext=(
                            dict_all[key]['avg_train_cost'] * 0.990,
                            min(max_value * 1.005, dict_all[key]['q_error']) * 1.005),
                        fontsize=7
                        )

        plt.legend(loc='best', fontsize=8)
        ax.set_xlim(min_Time, max_Time)
        ax.set_ylim(min_Err, max_Err)

        ax.set_xlabel("time", fontsize=15)
        ax.set_ylabel("q_error", fontsize=15)

        ax.set_title(k + ":time-err")

        ax = fig.add_subplot(len(dict_alls.keys()), 2, count)
        count += 1

        temp_error = []
        x_label = []

        for key in dict_all.keys():
            x_label.append(key)
            temp_error.append(dict_all[key]['avg_eval_cost'])

        colors = ['pink', 'lightblue', 'lightgreen', 'yellow']
        ax.bar(x_label, temp_error, color=colors)

        ax.set_ylim(0, max(temp_error) * 1.1)
        y_formatter = ScalarFormatter(useMathText=True)
        y_formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(y_formatter)

        for a, b in enumerate(temp_error):
            ax.text(a, b, '%.2e' % b, ha='center', va='bottom', fontsize=15)

        ax.set_title(k)

    plt.savefig(save_to_dir + "/scatter_err.png")


def plot_scatter_time(dict_alls, save_to_dir):
    length = max([len(dict_alls[key].keys()) for key in dict_alls.keys()])
    fig = plt.figure(figsize=(5 * length, 5 * (len(dict_alls.keys()))))

    fig.subplots_adjust(left=0.1, right=0.950, top=0.9, bottom=0.10,
                        wspace=0.2, hspace=0.1)

    fig.suptitle('sub', y=1.1, size=20)

    for idx, k in enumerate(dict_alls.keys()):
        dict_all = dict_alls[k]
        count = length * idx + 1
        for key in dict_all.keys():
            ax = fig.add_subplot(len(dict_alls.keys()), length, count)
            count += 1

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            max_pred = max(dict_all[key]['pred_times']) * 1.1
            max_tt = max(dict_all[key]['total_times']) * 1.1
            ax.plot(
                [0, max(max_tt, max_pred)],
                [0, max(max_tt, max_pred)],
                linewidth=1,
                color="black",
                linestyle="--",
            )
            ax.scatter(
                dict_all[key]['total_times'],
                dict_all[key]['pred_times'],
                marker="x",
                color="r",
                label="Optimized",
            )

            ax.set_xlabel("true_time", fontsize=15)
            ax.set_ylabel("pred_time", fontsize=15)

            ax.set_xlim(0, max_tt)
            ax.set_ylim(0, max_pred)
            txt = []
            for v in dict_all[key]['values'].keys():
                txt.append(v + ":" + str('%.3f' % dict_all[key]['values'][v]))
            ax.text(x=max_tt * 0.7, y=max_pred / 20, s="\n".join(txt))

            ax.set_title(k + "/" + key, fontsize=15, fontweight="bold")

    plt.savefig(save_to_dir + "/scatter_time.png")


def plot_box(dict_alls, save_to_dir):
    fig = plt.figure(figsize=(20 * len(dict_alls.keys()), 5))

    fig.subplots_adjust(left=0.05, right=0.980, top=0.9, bottom=0.1,
                        wspace=0.2, hspace=0.1)
    count = 1
    num = save_to_dir.split("/")[-1].split("-")[-1]
    for k in dict_alls.keys():
        ax = fig.add_subplot(1, len(dict_alls.keys()), count)
        count += 1
        dict_all = dict_alls[k]
        data = []
        labels = []
        for key in dict_all.keys():
            data.append(dict_all[key])
            labels.append(key)

        bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        if max([np.percentile(a, 90) for a in data]) > 10:
            ax.set_yscale('log', base=10)

        colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'cyan', 'burlywood', 'slategrey']
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)

        box_colors = ['darkkhaki', 'royalblue']
        num_boxes = len(data)
        medians = np.empty(num_boxes)
        for i in range(num_boxes):
            med = bp['medians'][i]
            median_x = []
            median_y = []
            for j in range(2):
                median_x.append(med.get_xdata()[j])
                median_y.append(med.get_ydata()[j])
            medians[i] = median_y[0]

        pos = np.arange(num_boxes) + 1
        upper_labels = [str(round(s, 2)) for s in medians]
        colors = ['red', 'blue', 'green', 'orange', 'cyan', 'brown', 'grey']
        for tick, label, color in zip(range(num_boxes), ax.get_xticklabels(), colors):
            k_ = tick % 2
            ax.text(pos[tick], .95, upper_labels[tick],
                    transform=ax.get_xaxis_transform(),
                    horizontalalignment='center',
                    color=color, fontsize=15)

        ax.tick_params(labelsize=25)
        ax.set_title(k + "(" + num + ")", fontsize=20, fontweight="bold")

    plt.savefig(save_to_dir + "/2-" + num + "-box.png")


def fig(plan):
    pattern_num = re.compile(r'\d+.?\d*')
    pattern_type = re.compile(r'\w*\(?\w*\)?:')
    for v_dir in plan['v_dirs']:
        dict_alls = {}
        dict_time_prederrs = {}
        q_error_alls = {}

        save_to_dir = "./res/" + plan['name'] + "/" + v_dir.split("/")[-1]
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)

        root_dirs = [os.path.join(v_dir, file) for file in plan['ds']]

        for root_dir in root_dirs:
            dict_all = {}
            q_error_all = {}
            dict_time_prederr = {}
            for root, dirs, files in os.walk(root_dir):  # tpch 和 tpcc 下
                for dir in dirs:
                    for root1, dirs1, files1 in os.walk(root_dir + "/" + dir):  # mid_models 下
                        for save_dir in dirs1:
                            file_root = root_dir + "/" + dir + "/" + save_dir  # save_model path
                            for r1, d1, f1 in os.walk(file_root):
                                if d1 != []:
                                    file_dir = r1 + "/" + d1[0]
                                else:
                                    continue
                                print(file_dir)
                                if [file for file in f1 if 'times' in file] is not None:
                                    with open(file_dir + "/pred_times.pickle", "rb") as f:
                                        pred_times = pickle.load(f)
                                        if 'QPPNet' in save_dir:
                                            pred_timess = [time.cpu().detach().numpy() for time in pred_times]
                                        else:
                                            pred_timess = [time for time in pred_times]
                                    with open(file_dir + "/total_times.pickle", "rb") as f:
                                        total_times = pickle.load(f)
                                        if 'QPPNet' in save_dir:
                                            total_timess = [time.cpu().detach().numpy() for time in total_times]
                                        else:
                                            total_timess = [time for time in total_times]
                                else:
                                    continue

                                for idx, pt in enumerate(pred_timess):
                                    tempTT = total_timess[idx]
                                    tempPT = pt
                                    if idx == 0:
                                        total_times = list(tempTT)
                                        pred_times = list(tempPT)
                                        continue
                                    total_times.extend(list(tempTT))
                                    pred_times.extend(list(tempPT))

                                model = dir + "/" + save_dir

                                if model not in plan['model']:
                                    continue

                                dict_all[model] = {}
                                dict_all[model]['pred_times'] = pred_times
                                dict_all[model]['total_times'] = total_times

                                x_ = pred_times - np.mean(pred_times)
                                y_ = total_times - np.mean(total_times)
                                r = np.dot(x_, y_) / (np.linalg.norm(x_) * np.linalg.norm(y_))

                                q_error = Metric.q_error_numpy(total_times, pred_times, 0.001)

                                dict = {}
                                dict['pearson'] = r
                                dict['q_error'] = q_error[4]
                                dict['q_error99'] = q_error[0]
                                dict['q_error95'] = q_error[1]
                                dict['q_error90'] = q_error[2]

                                q_error_all[model] = Metric.q_error_all(total_times, pred_times, 0.001)

                                with open(file_dir + "/eval.txt", "r+") as f:
                                    txt = f.read()
                                    type = [i.replace(":", "") for i in pattern_type.findall(txt)]
                                    num = [float(i.replace("s", "")) for i in pattern_num.findall(txt)]

                                for idx, t in enumerate(type):
                                    if t != 'pred_err':
                                        continue
                                    dict[t] = num[idx]
                                dict_all[model]['values'] = dict

                                dict_time_prederr[model] = {}

                                dict_time_prederr[model]['avg_train_cost'] = num[-1]
                                dict_time_prederr[model]['q_error'] = q_error[4]

                                with open(file_dir + "/eval_time.txt", "r+") as f:
                                    txt = f.read()
                                    num = [float(i.replace("u", "")) for i in pattern_num.findall(txt)]

                                dict_time_prederr[model]['avg_eval_cost'] = (len(pred_times) / num[-1]
                                                                             if num[-1] != 0 else 1) * 1000000

            dict_alls[root_dir.split("\\")[-1]] = dict_all
            dict_time_prederrs[root_dir.split("\\")[-1]] = dict_time_prederr
            q_error_alls[root_dir.split("\\")[-1]] = q_error_all

        plan["sort"](plan, q_error_alls)
        plan["sort"](plan, dict_time_prederrs)
        plot_box(q_error_alls, save_to_dir)
        plot_scatter_err(dict_time_prederrs, save_to_dir)


plan1 = {
    'v_dirs': [
        "../../2000",
        # "../../4000",
        # "../../6000",
        # "../../8000",
        # "../../10000",
    ],
    'ds': ['TPCH', 'Sysbench', 'job-light'],
    'name': 'fig1',
    'model': ['origin_model/save_model_QPPNet',
              'origin_model/save_model_MSCN',
              'snapshot_model_FR/save_model_QPPNet',
              'snapshot_model_FR/save_model_MSCN'],
    'name_align':name_align,
    'sort':sort,
    'func':[fig]
}

if __name__ == '__main__':
    plans = [
        plan1,
    ]
    print("start fig")
    for plan in plans:
        for func in plan['func']:
            func(plan)
    print("end fig")
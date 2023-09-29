import sys

import pandas as pd
import shap
import torch
import os
import numpy as np
import json

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from R2.tree import TreeR2
from utils.util import get_time
from utils.metric import Metric
from sklearn.ensemble import RandomForestRegressor
import pickle

from openfe import openfe

basic = 3


# For computing loss
def squared_diff(output, target):
    return np.sum((output - target) ** 2)


###############################################################################
#                               QPP Net Architecture                          #
###############################################################################


class RandomForest():
    def __init__(self, opt, pass_dim_dict):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
            else torch.device('cpu:0')
        self.save_dir = opt.mid_data_dir + opt.save_dir + "/" + str(opt.batch_size)
        self.test = False
        self.eval = False
        self.last = False
        # self.test_time = opt.test_time
        self.batch_size = opt.batch_size
        self.dataset = opt.dataset
        self.latest_save_RF_freq = opt.save_latest_epoch_freq
        self.latest_save_freq = opt.save_latest_epoch_freq
        self.best_epoch = 0

        self.dim_dict = pass_dim_dict
        filter_type = opt.mid_data_dir.split("_")[-1]

        self.filter = True if "knob_model_" in opt.mid_data_dir else False
        if os.path.exists("./2200-2000-2000-2000/" + opt.data_dir.split("/")[-1] + "/knob_model/save_model_RandomForest/"
                          + str(opt.batch_size) + "/"+filter_type+"_values_array.pickle"):

            with open("./2200-2000-2000-2000/" + opt.data_dir.split("/")[-1] + "/knob_model/save_model_RandomForest/"
                      + str(opt.batch_size) + "/"+filter_type+"_values_array.pickle", "rb") as f:
                self.save_values_array = pickle.load(f)

        self.select = True if "select" in opt.mid_data_dir else False
        if os.path.exists(
                "./2200-2000-2000-2000/" + opt.data_dir.split("/")[-1] + "/knob_model/save_model_RandomForest/"
                + str(opt.batch_size) + "/openfe_values_array.pickle"):
            with open("./2200-2000-2000-2000/" + opt.data_dir.split("/")[-1] + "/knob_model/save_model_RandomForest/"
                      + str(opt.batch_size) + "/openfe_values_array.pickle", "rb") as f:
                self.features_array = pickle.load(f)

        self.save_X = {}
        self.save_y = {}
        self.total_time = {}

        for operator in self.dim_dict:  # 对回归器进行复用
            self.save_X[operator] = []
            self.save_y[operator] = []
            self.total_time[operator] = []

        self.test_loss = None
        self.last_total_loss = None
        self.last_test_loss = None
        self.last_pred_err = None
        self.best_total_loss = sys.float_info.max
        self.pred_err = None

        if not os.path.exists(opt.mid_data_dir + opt.save_dir):
            os.mkdir(opt.mid_data_dir + opt.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Initialize the neural units
        self.models = {}

        for operator in self.dim_dict:  # 对回归器进行复用
            self.models[operator] = RandomForestRegressor(max_depth=9, n_estimators=50, warm_start=False,
                                                          random_state=1)

        self.loss_fn = squared_diff
        self.dummy = np.zeros(1)
        self.acc_loss = {operator: [self.dummy] for operator in self.dim_dict}
        self.curr_losses = {operator: 0 for operator in self.dim_dict}
        self.total_loss = None

    def set_input(self, samp_dicts):
        self.input = samp_dicts

    def number_to_output_vec(self, number, dimx, dimy):
        # 创建一个dimx行dimy维的零矩阵
        vector = np.zeros((dimx, dimy))
        # 将每一行的首元素设置为给定的数值
        vector[:, 0] = number
        return vector

    ########################################################################
    #                       RF            Models                           #
    ########################################################################
    def regress_OneElement(self, samp_batch):
        feat_vec = samp_batch['feat_vec']
        y_time = samp_batch['total_time']
        operator = samp_batch['node_type']
        y_time = y_time.reshape(-1, 1)

        input_vec = feat_vec

        for child_plan_dict in samp_batch['children_plan']:
            child_output_vec, _ = self.regress_OneElement(child_plan_dict)
            if not child_plan_dict['is_subplan']:
                input_vec = np.concatenate((input_vec, child_output_vec), axis=1)

        expected_len = self.dim_dict[operator]
        if expected_len > input_vec.shape[1]:
            add_on = np.zeros([input_vec.shape[0], expected_len - input_vec.shape[1]])
            input_vec = np.concatenate((input_vec, add_on), axis=1)

        y_time_raveled = y_time.ravel()

        if not self.last:
            self.save_X[operator].extend(input_vec)
            self.save_y[operator].extend(y_time_raveled)
            # self.total_time[operator].extend(samp_batch['total_time'])

            if not self.test:
                if self.filter:
                    # print(operator)
                    # print(np.array(self.save_X[operator]).shape)
                    # print(self.save_values_array[samp_batch['node_type']])
                    self.models[operator].fit(
                        self.filterfunc1(np.array(self.save_X[operator]),
                                         self.save_values_array[samp_batch['node_type']]),
                        self.save_y[operator])
                else:
                    self.models[operator].fit(self.save_X[operator], self.save_y[operator])

            if self.filter:
                pred_time = self.models[operator].predict(
                    self.filterfunc1(input_vec, self.save_values_array[samp_batch['node_type']]))
            else:
                pred_time = self.models[operator].predict(input_vec)

        else:
            if samp_batch['node_type'] in self.features_array.keys():
                ofe = openfe.OpenFE()
                ofe.verbose = False
                ofe.tmp_save_path = "./openfe_tmp_data_RF.feather"

                input_vec = self.filterfunc1(input_vec, self.save_values_array[samp_batch['node_type']])
                temp = self.ofe_transform(np.array(input_vec), operator)
                pred_time = self.models[operator].predict(temp)
            else:
                pred_time = self.models[operator].predict(input_vec)

        if len(feat_vec[0]) > 31:
            output_vec = np.concatenate((pred_time.reshape(-1, 1), feat_vec[:, :30]), axis=1)
        else:
            add_on = np.zeros([feat_vec.shape[0], 31 - feat_vec.shape[1]])
            feat_vec = np.concatenate((feat_vec, add_on), axis=1)
            output_vec = np.concatenate((pred_time.reshape(-1, 1), feat_vec[:, :30]), axis=1)

        loss = (pred_time - samp_batch['total_time']) ** 2

        self.acc_loss[operator].append(loss)
        return output_vec, pred_time

    def regress(self, epoch):
        data_size = 0
        total_loss = 0
        total_losses = {operator: [np.zeros(1)] for operator in self.dim_dict}
        if self.test:
            test_loss = []
            pred_err = []

        if self.eval:
            self.total_times = []
            self.pred_times = []
            self.total_costs = []
            self.plan_times = []

        all_tt, all_pred_time = None, None

        total_mean_mae = 0
        for idx, samp_dict in enumerate(self.input):
            del self.acc_loss
            self.acc_loss = {operator: [self.dummy] for operator in self.dim_dict}
            _, pred_time = self.regress_OneElement(samp_dict)

            if self.dataset == "PSQLTPCH":
                epsilon = 1.1920928955078125e-07
            else:
                epsilon = 0.

            data_size += len(samp_dict['total_time'])

            if self.test:
                tt = samp_dict['total_time']

                if self.eval:
                    self.total_times.append(tt)
                    self.pred_times.append(pred_time)
                    self.total_costs.append(samp_dict['total_cost'])
                    self.plan_times.append(samp_dict['plan_time'])

                test_loss.append(np.abs(tt - pred_time))
                curr_pred_err = Metric.pred_err_numpy(tt, pred_time, epsilon)
                pred_err.append(curr_pred_err)

                all_tt = tt if all_tt is None else np.concatenate((all_tt, tt), axis=0)
                all_pred_time = pred_time if all_pred_time is None \
                    else np.concatenate((all_pred_time, pred_time), axis=0)

                curr_mean_mae = Metric.mean_mae_numpy(tt, pred_time, epsilon)
                total_mean_mae += curr_mean_mae * len(tt)

                if epoch % 50 == 0:
                    print("####### eval by temp: idx {}, test_loss {}, pred_err {}" \
                          .format(idx, np.mean(np.abs(tt - pred_time)),
                                  np.mean(curr_pred_err)))
            D_size = 0
            subbatch_loss = np.zeros(1)
            for operator in self.acc_loss:
                all_loss = np.concatenate(self.acc_loss[operator], axis=0)
                D_size += all_loss.shape[0]
                subbatch_loss += np.sum(all_loss)

                total_losses[operator].append(all_loss)

            subbatch_loss = np.mean(np.sqrt(subbatch_loss / D_size))
            total_loss += subbatch_loss * samp_dict['subbatch_size']

        if self.test:
            all_test_loss = np.concatenate(test_loss, axis=0)

            all_test_loss = np.mean(all_test_loss)
            self.test_loss = all_test_loss

            all_pred_err = np.concatenate(pred_err, axis=0)
            self.pred_err = np.mean(all_pred_err)

            if epoch % 50 == 0:
                print("test batch Pred Err: {}".format(self.pred_err))

        else:
            self.total_loss = np.mean(total_loss / self.batch_size)
            self.curr_losses = {operator: np.mean(np.concatenate(total_losses[operator], axis=0)) for operator in
                                self.dim_dict}

    def evaluate(self, dataset):
        self.set_input(dataset)
        self.eval = True
        self.test = True
        self.regress(0)
        self.last_total_loss = self.total_loss
        self.last_test_loss = self.test_loss
        self.last_pred_err = self.pred_err
        self.test_loss, self.pred_err = None, None

        with open(self.save_dir + "/pred_times.pickle", "wb") as f:
            pickle.dump(self.pred_times, f)
        with open(self.save_dir + "/total_times.pickle", "wb") as f:
            pickle.dump(self.total_times, f)
        with open(self.save_dir + "/total_costs.pickle", "wb") as f:
            pickle.dump(self.total_costs, f)
        with open(self.save_dir + "/plan_times.pickle", "wb") as f:
            pickle.dump(self.plan_times, f)

    def get_current_losses(self):
        return self.curr_losses

    def load(self, name):
        paths = os.listdir(self.save_dir)
        for model_name in paths:
            model_name_without_extension = model_name.split("_")[0]
            if 'best' in model_name:
                with open(os.path.join(self.save_dir, model_name), 'rb') as f:
                    f.seek(0)
                    self.models[model_name_without_extension] = pickle.load(f)

    def calculate_shap(self, eval_dataset):
        self.test = True
        self.set_input(eval_dataset)
        self.eval = True
        self.save = True
        self.filter = False
        self.select = False
        self.regress(0)
        self.last_test_loss = self.test_loss.item()
        self.last_pred_err = self.pred_err.item()
        self.test_loss, self.pred_err = None, None

        shap_values_array = {}

        for operator in self.dim_dict:

            if self.save_X[operator] == []:
                continue

            samp1 = np.random.choice(np.arange(len(self.save_X[operator])),
                                     min(100, max(int(len(self.save_X[operator]) / 4), 50)), replace=True)
            samp2 = np.random.choice(np.arange(len(self.save_X[operator])),
                                     min(20, max(int(len(self.save_X[operator]) / 16), 10)), replace=True)
            background_data = np.array(self.save_X[operator])[samp1, :]
            data = np.array(self.save_X[operator])[samp2, :]
            del self.save_X[operator]

            explainer = shap.TreeExplainer(self.models[operator], background_data)
            shap_values = explainer.shap_values(data, check_additivity=False)
            n_array = np.array(shap_values)

            sum_array = []
            write_info_array = []
            flag = 1
            print(n_array.shape)
            for n in range(n_array.shape[1]):
                if float(np.sum(np.abs(n_array[:, n]))) > 0:
                    sum_array.append(n)
                    write_info_array.append(float(np.sum(np.abs(n_array[:, n]))))
                    flag = 0
            if flag == 1:
                for n in range(n_array.shape[0]):
                    sum_array.append(n)

            shap_values_array[operator] = sum_array

            with open(self.save_dir + "/shap_info.txt", "a") as f:
                f.write(operator + ": " + ",".join([str(s) for s in write_info_array]) + "\n")

            fig = plt.figure()
            shap.summary_plot(shap_values, data, show=False)
            plt.savefig(self.save_dir + "/" + operator, bbox_inches='tight')

        with open(self.save_dir + "/shap_values_array.pickle", "wb") as f:
            pickle.dump(shap_values_array, f)

        return shap_values_array

    def calculate_openfe(self, eval_dataset):

        self.test = True
        self.set_input(eval_dataset)
        self.eval = True
        self.save = True
        self.regress(0)
        self.last_test_loss = self.test_loss.item()
        self.last_pred_err = self.pred_err.item()
        self.test_loss, self.pred_err = None, None

        if os.path.exists(self.save_dir + "/openfe_values_array.pickle"):
            with open(self.save_dir + "/openfe_values_array.pickle", "rb") as f:
                features_array = pickle.load(f)
        else:
            features_array = {}

        for operator in self.dim_dict:
            print(operator)
            if operator in features_array.keys():
                continue

            if self.save_X[operator] == []:
                continue
            samp = np.random.choice(np.arange(len(self.save_X[operator])), 40, replace=True)
            X = np.array(self.save_X[operator])[samp]
            y = np.array(self.save_y[operator])[samp]

            del self.save_X[operator]
            del self.save_y[operator]

            ofe = openfe.OpenFE()
            try:
                train_x = pd.DataFrame(X, columns=[str(col) for col in range(X.shape[1])])
                train_y = pd.DataFrame(y, columns=['1'])
                features = ofe.fit(data=train_x,
                                   label=train_y,
                                   tmp_save_path='./openfe_tmp_data_RF.feather',
                                   n_jobs=8,
                                   verbose=False)
                features_array[operator] = features
            except:
                continue

            with open(self.save_dir + "/openfe_values_array.pickle", "wb") as f:
                pickle.dump(features_array, f)

        return features_array

    def calculate_R2(self, eval_dataset):
        self.test = True
        self.set_input(eval_dataset)
        self.eval = True
        self.save = True
        self.filter = False
        self.select = False
        self.regress(0)
        self.last_test_loss = self.test_loss.item()
        self.last_pred_err = self.pred_err.item()
        self.test_loss, self.pred_err = None, None

        R2_models = self.models
        R2_values_array = {}

        for operator in self.dim_dict:

            if self.save_X[operator] == []:
                continue

            TrainX, testX, TrainY, testY = \
                train_test_split(self.save_X[operator], self.save_y[operator], train_size=min(
                    len(self.save_X[operator])-1,max(1024,int(len(self.save_X[operator])/10))), random_state=2)
            # TrainX, testX, TrainY, testY = \
            #     train_test_split(self.save_X[operator], self.save_y[operator], train_size=0.8, random_state=2)

            TrainX = np.array(TrainX)
            TrainY = np.array(TrainY)

            print(TrainX.shape)
            del self.save_X[operator]

            R2 = TreeR2(R2_models[operator], TrainX, TrainY)
            R2_values_array[operator] = np.array(R2.filter_values)
            print(R2_values_array[operator].shape)

            with open(self.save_dir + "/R2_values_array.pickle", "wb") as f:
                pickle.dump(R2_values_array, f)

        return R2_values_array

    def optimize_parameters(self, epoch):
        self.test = False
        self.regress(epoch)
        self.last_total_loss = self.total_loss
        self.save_units(epoch)

        self.input = self.test_dataset
        self.test = True
        self.regress(epoch)
        self.last_test_loss = self.test_loss
        self.last_pred_err = self.pred_err
        self.test_loss, self.pred_err = None, None

    def openfe_train(self):
        self.test = False
        print("begin train")
        self.regress(0)

        print("begin openfe_train")
        for operator in self.dim_dict:

            if self.save_X[operator] == []:
                continue
            if operator in self.features_array.keys():
                ofe = openfe.OpenFE()
                ofe.verbose = False
                ofe.tmp_save_path = "./openfe_tmp_data_RF.feather"
                try:
                    input_vec = self.filterfunc1(np.array(self.save_X[operator]), self.save_values_array[operator])
                    temp = self.ofe_transform(input_vec, operator)
                    self.models[operator].fit(np.array(temp), np.array(self.save_y[operator]))
                except:
                    self.models[operator].fit(self.save_X[operator], self.save_y[operator])
                    del self.features_array[operator]
                    print('del self.features_array[operator]:', operator)
            else:
                self.models[operator].fit(
                    self.save_X[operator],
                    self.save_y[operator])

        print("save_units")
        self.save_units(0)

    def openfe_eval(self, dataset):
        self.eval = True
        self.test = True
        self.last = True
        print("begin openfe_eval")

        self.set_input(dataset)
        self.regress(0)

        self.last_total_loss = self.total_loss
        self.last_test_loss = self.test_loss
        self.last_pred_err = self.pred_err
        self.test_loss, self.pred_err = None, None

        with open(self.save_dir + "/pred_times.pickle", "wb") as f:
            pickle.dump(self.pred_times, f)
        with open(self.save_dir + "/total_times.pickle", "wb") as f:
            pickle.dump(self.total_times, f)
        with open(self.save_dir + "/total_costs.pickle", "wb") as f:
            pickle.dump(self.total_costs, f)
        with open(self.save_dir + "/plan_times.pickle", "wb") as f:
            pickle.dump(self.plan_times, f)

    def save_units(self, epoch):
        self.best_total_loss = self.total_loss
        self.best_epoch = epoch

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for operator in self.dim_dict:
            model_path = os.path.join(self.save_dir, operator + '_best.pickle')
            with open(model_path, 'wb') as file_model:
                pickle.dump(self.models[operator], file_model)

    @get_time
    def filterfunc1(self, input_vec, filter_array):
        return input_vec[:, filter_array]

    @get_time
    def ofe_transform(self, input_vec, op):
        # input_vec = self.transform(input_vec, 0)

        ofe = openfe.OpenFE()
        ofe.verbose = False
        ofe.tmp_save_path = './openfe_tmp_data_QPP.feather'

        input_vec = ofe.transform1(data=input_vec,
                                   new_features_list=self.features_array[op],
                                   n_jobs=8)

        return np.array(input_vec)

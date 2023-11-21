import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from pyDOE import lhs

from greedy.deepNN import DeepR2
from greedy.tree import TreeR2
from config import filter_type
from utils.util import get_time
from utils.metric import Metric

basic = 3


# For computing loss
def squared_diff(output, target):
    return torch.sum((output - target) ** 2)


###############################################################################
#                        Operator Neural Unit Architecture                    #
###############################################################################
# Neural Unit that covers all operators
class NeuralUnit(nn.Module):
    """Define a Resnet block"""

    def __init__(self, node_type, dim_dict, num_layers=5, hidden_size=128,
                 output_size=32, norm_enabled=False):
        """
        Initialize the InternalUnit
        """
        super(NeuralUnit, self).__init__()
        self.node_type = node_type
        self.dense_block = self.build_block(num_layers, hidden_size, output_size,
                                            input_dim=dim_dict[node_type])

    def build_block(self, num_layers, hidden_size, output_size, input_dim):
        """Construct a block consisting of linear Dense layers.
        Parameters:
            num_layers  (int)
            hidden_size (int)           -- the number of channels in the conv layer.
            output_size (int)           -- size of the output layer
            input_dim   (int)           -- input size, depends on each node_type
            norm_layer                  -- normalization layer
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        assert (num_layers >= 2)
        dense_block = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for i in range(num_layers - 2):
            dense_block += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        dense_block += [nn.Linear(hidden_size, output_size), nn.ReLU()]

        for layer in dense_block:
            try:
                nn.init.xavier_uniform_(layer.weight)
            except:
                pass
        return nn.Sequential(*dense_block)

    def forward(self, x):
        """ Forward function """
        out = self.dense_block(x)
        return out

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_GD = False
        for param in self.network.fc.parameters():
            param.require_GD = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_GD = True


###############################################################################
#                               QPP Net Architecture                          #
###############################################################################

class QPPNet():
    def __init__(self, opt, pass_dim_dict):
        self.device = torch.device('cuda:0') \
            if torch.cuda.is_available() \
            else torch.device('cpu:0')
        self.save_dir = opt.mid_data_dir + opt.save_dir + "/" + str(opt.batch_size)
        self.test = False
        self.shap = False
        self.save = False
        self.eval = False
        self.test_time = True if opt.mode == 'test' else False
        self.batch_size = opt.batch_size
        self.dataset = opt.dataset

        self.dim_dict = pass_dim_dict

        if '_eval' in opt.mode:
            self.filter_type = opt.mode.replace("_eval", "")
        else:
            self.filter_type = opt.mid_data_dir.split("_")[-1]

        print(self.filter_type)
        self.filter = True if "snapshot_model_" in opt.mid_data_dir else False

        print("./2000/" + opt.mid_data_dir.split("/")[-2] + "/snapshot_model/save_model_QPPNet/"
              + str(opt.batch_size) + "/" + self.filter_type + "_values_array.pickle")

        if os.path.exists(
                "./2000/" + opt.mid_data_dir.split("/")[-2] + "/snapshot_model/save_model_QPPNet/"
                + str(opt.batch_size) + "/" + self.filter_type + "_values_array.pickle"):

            with open("./2000/" + opt.mid_data_dir.split("/")[-2] + "/snapshot_model/save_model_QPPNet/"
                      + str(opt.batch_size) + "/" + self.filter_type + "_values_array.pickle", "rb") as f:
                self.save_values_array = pickle.load(f)

            if self.filter:
                for key in self.save_values_array:
                    self.dim_dict[key] = len(self.save_values_array[key])

        self.last_total_loss = None
        self.last_pred_err = None
        self.pred_err = None

        if not os.path.exists(opt.mid_data_dir + opt.save_dir):
            os.mkdir(opt.mid_data_dir + opt.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Initialize the neural units
        self.units = {}
        self.optimizers, self.schedulers = {}, {}
        self.best = 100000

        for operator in self.dim_dict:
            self.units[operator] = NeuralUnit(operator, self.dim_dict).to(self.device)

            if opt.SGD:
                optimizer = torch.optim.SGD(self.units[operator].parameters(),
                                            lr=opt.lr, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(self.units[operator].parameters(),
                                             opt.lr)  # opt.lr

            if opt.scheduler:
                sc = lr_scheduler.StepLR(optimizer, step_size=opt.step_size,
                                         gamma=opt.gamma)
                self.schedulers[operator] = sc

            self.optimizers[operator] = optimizer

        self.loss_fn = squared_diff
        # Initialize the global loss accumulator dict
        self.dummy = torch.zeros(1).to(self.device)
        self.acc_loss = {operator: [self.dummy] for operator in self.dim_dict}
        self.curr_losses = {operator: 0 for operator in self.dim_dict}
        self.total_loss = None
        self._test_losses = dict()

        if opt.start_epoch != 0:
            self.load(opt.start_epoch)

    def set_input(self, samp_dicts):
        self.input = samp_dicts

    def _forward_oneQ_batch(self, samp_batch):
        '''
        Calcuates the loss for a batch of queries from one query template

        compute a dictionary of losses for each operator

        return output_vec, where 1st col is predicted time
        '''

        feat_vec = samp_batch['feat_vec']

        input_vec = torch.from_numpy(feat_vec).to(self.device)

        subplans_time = []
        for child_plan_dict in samp_batch['children_plan']:
            child_output_vec, _ = self._forward_oneQ_batch(child_plan_dict)
            if not child_plan_dict['is_subplan']:
                input_vec = torch.cat((input_vec, child_output_vec), axis=1).to(self.device)
            else:
                subplans_time.append(
                    torch.index_select(child_output_vec, 1, torch.zeros(1, dtype=torch.long).to(self.device)).to(
                        self.device))

        expected_len = self.dim_dict[samp_batch['node_type']]
        if expected_len > input_vec.size()[1]:
            add_on = torch.zeros(input_vec.size()[0], expected_len - input_vec.size()[1]).to(self.device)
            input_vec = torch.cat((input_vec, add_on), axis=1)

        if self.filter:
            if samp_batch['node_type'] in self.save_values_array.keys():
                output_vec = self.units[samp_batch['node_type']](
                    self.filterfunc1(input_vec, self.save_values_array[samp_batch['node_type']])
                )
            else:
                output_vec = self.units[samp_batch['node_type']](input_vec)
        else:
            output_vec = self.units[samp_batch['node_type']](input_vec)

        pred_time = torch.index_select(output_vec, 1, torch.zeros(1, dtype=torch.long).to(self.device)).to(self.device)

        if self.save == True:
            self.save_X[samp_batch['node_type']].extend(input_vec)
            self.save_y[samp_batch['node_type']].extend(samp_batch['total_time'])

        cat_res = torch.cat([pred_time] + subplans_time, axis=1).to(self.device)
        pred_time = torch.sum(cat_res, 1).to(self.device)

        loss = (pred_time - torch.from_numpy(samp_batch['total_time']).to(self.device)) ** 2
        self.acc_loss[samp_batch['node_type']].append(loss)

        try:
            assert (not (torch.isnan(output_vec).any()))
        except:
            print(samp_batch['node_type'])
            print("feat_vec", feat_vec, "input_vec", input_vec)
            if torch.cuda.is_available():
                print(samp_batch['node_type'], "output_vec: ", output_vec,
                      self.units[samp_batch['node_type']].state_dict())
            else:
                print(samp_batch['node_type'], "output_vec: ", output_vec,
                      self.units[samp_batch['node_type']].cpu().state_dict())
            exit(-1)
        return output_vec, pred_time

    def _forward(self, epoch):
        # self.input is a list of preprocessed plan_vec_dict
        total_loss = torch.zeros(1).to(self.device)
        total_losses = {operator: [torch.zeros(1).to(self.device)] \
                        for operator in self.dim_dict}
        if self.test:
            test_loss = []
            pred_err = []

        if self.eval:
            self.total_times = []
            self.pred_times = []
            self.total_costs = []
            self.plan_times = []

        all_tt, all_pred_time = None, None

        data_size = 0
        total_mean_mae = torch.zeros(1).to(self.device)
        for idx, samp_dict in enumerate(self.input):
            # first clear prev computed losses
            del self.acc_loss
            self.acc_loss = {operator: [self.dummy] for operator in self.dim_dict}

            if self.shap:
                if len(samp_dict['feat_vec']) < 5:
                    backgroundsamp = np.arange(len(samp_dict['feat_vec']))
                    datasamp = np.arange(len(samp_dict['feat_vec']))
                else:
                    backgroundsamp, datasamp, _, _ = train_test_split(np.arange(len(samp_dict['feat_vec']))
                                                                      , np.arange(len(samp_dict['feat_vec']))
                                                                      , train_size=min(200, int(len(
                            samp_dict['feat_vec']) * 0.8))
                                                                      , test_size=min(50, int(len(
                            samp_dict['feat_vec']) * 0.2))
                                                                      , random_state=1)
                _, pred_time = self.collect_forward_oneQ_batch(samp_dict, backgroundsamp, datasamp)
            else:
                _, pred_time = self._forward_oneQ_batch(samp_dict)

            if self.dataset == "PSQLTPCH":
                epsilon = torch.finfo(pred_time.dtype).eps
            else:
                epsilon = 0.001

            data_size += len(samp_dict['total_time'])

            if self.test:
                with torch.no_grad():
                    tt = torch.from_numpy(samp_dict['total_time']).to(self.device)

                    if self.eval:
                        self.total_times.append(tt)
                        self.pred_times.append(pred_time)
                        self.total_costs.append(torch.from_numpy(samp_dict['total_cost']).to(self.device))
                        self.plan_times.append(torch.from_numpy(samp_dict['plan_time']).to(self.device))

                    test_loss.append(torch.abs(tt - pred_time))
                    curr_pred_err = Metric.pred_err(tt, pred_time, epsilon)
                    pred_err.append(curr_pred_err)

                    if np.isnan(curr_pred_err.detach().cpu()).any() or \
                            np.isinf(curr_pred_err.detach().cpu()).any():
                        print("feat_vec", samp_dict['feat_vec'])
                        print("pred_time", pred_time)
                        print("total_time", tt)

                    all_tt = tt if all_tt is None else torch.cat([tt, all_tt])
                    all_pred_time = pred_time if all_pred_time is None else torch.cat([pred_time, all_pred_time])

                    curr_mean_mae = Metric.mean_mae(tt, pred_time, epsilon)
                    total_mean_mae += curr_mean_mae * len(tt)

                    if epoch % 50 == 0:
                        print("# eval by temp: idx {}, test_loss {}, pred_err {}" \
                              .format(idx, torch.mean(torch.abs(tt - pred_time)).item(),
                                      torch.mean(curr_pred_err).item()))

            D_size = 0
            subbatch_loss = torch.zeros(1).to(self.device)
            for operator in self.acc_loss:
                all_loss = torch.cat(self.acc_loss[operator])
                D_size += all_loss.shape[0]
                subbatch_loss += torch.sum(all_loss)

                total_losses[operator].append(all_loss)

            subbatch_loss = torch.mean(torch.sqrt(subbatch_loss / D_size))
            total_loss += subbatch_loss * samp_dict['subbatch_size']

        # test batch Pred Err: {}, R(q): {}, Accumulated Error
        if self.test:
            with torch.no_grad():
                all_test_loss = torch.cat(test_loss).to(self.device)
                all_test_loss = torch.mean(all_test_loss).to(self.device)
                self.test_loss = all_test_loss

                all_pred_err = torch.cat(pred_err).to(self.device)
                self.pred_err = torch.mean(all_pred_err).to(self.device)

                if epoch % 50 == 0:
                    print("test batch Pred Err: {}".format(self.pred_err))
        else:
            self.curr_losses = {operator: torch.mean(torch.cat(total_losses[operator])).item() for operator in
                                self.dim_dict}
            self.total_loss = torch.mean(total_loss / self.batch_size)

    def backward(self):
        self.last_total_loss = self.total_loss.item()
        if self.best > self.total_loss.item():
            self.best = self.total_loss.item()
            self.save_units('best')
        self.total_loss.backward()
        self.total_loss = None

    def backward_GDFalse(self):
        self.last_total_loss = self.total_loss.item()
        if self.best > self.total_loss.item():
            self.best = self.total_loss.item()
            self.save_units('best')

        self.total_loss.requires_GD_(True)
        self.total_loss.backward()
        self.total_loss = None

    def optimize_parameters(self, epoch):

        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.test = False
        self._forward(epoch)
        # clear prev grad first
        for operator in self.optimizers:
            self.optimizers[operator].zero_grad()

        self.backward()

        for operator in self.optimizers:
            self.optimizers[operator].step()
            if len(self.schedulers) > 0:
                self.schedulers[operator].step()

        self.input = self.test_dataset
        self.test = True
        self._forward(epoch)
        self.last_test_loss = self.test_loss.item()
        self.last_pred_err = self.pred_err.item()
        self.test_loss, self.pred_err = None, None

    def part_train(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.test = False

        self._forward(epoch)

        # clear prev grad first
        for operator in self.optimizers:
            self.optimizers[operator].zero_grad()

        self.backward_GDFalse()

        for operator in self.optimizers:
            self.optimizers[operator].step()
            if len(self.schedulers) > 0:
                self.schedulers[operator].step()

        self.input = self.test_dataset
        self.test = True
        self._forward(epoch)
        self.last_test_loss = self.test_loss.item()
        self.last_pred_err = self.pred_err.item()
        self.test_loss, self.pred_err = None, None

        return

    def evaluate(self, eval_dataset):
        self.test = True
        self.set_input(eval_dataset)
        self.eval = True
        self._forward(0)
        self.last_test_loss = self.test_loss.item()
        self.last_pred_err = self.pred_err.item()
        self.test_loss, self.pred_err = None, None

        with open(self.save_dir + "/pred_times.pickle", "wb") as f:
            pickle.dump(self.pred_times, f)
        with open(self.save_dir + "/total_times.pickle", "wb") as f:
            pickle.dump(self.total_times, f)
        with open(self.save_dir + "/total_costs.pickle", "wb") as f:
            pickle.dump(self.total_costs, f)
        with open(self.save_dir + "/plan_times.pickle", "wb") as f:
            pickle.dump(self.plan_times, f)

    def calculate_FR(self, eval_dataset):

        self.save_X = {}
        self.save_y = {}

        for operator in self.dim_dict:  # 对回归器进行复用
            self.save_X[operator] = []
            self.save_y[operator] = []

        self.test = True
        self.set_input(eval_dataset)
        self.eval = True
        self.save = True
        self._forward(0)

        filter_models = {}

        if 'tree' in self.filter_type:
            for operator in self.dim_dict:
                filter_models[operator] = RandomForestRegressor(max_depth=9, n_estimators=50, warm_start=False,
                                                                random_state=1)
        else:
            filter_models = self.units

        for operator in self.dim_dict:
            if self.save_X[operator] != []:
                self.save_X[operator] = np.array([item.cpu().detach().numpy() for item in self.save_X[operator]])
                self.save_y[operator] = np.array(self.save_y[operator])
                if 'tree' in self.filter_type:
                    filter_models[operator].fit(self.save_X[operator], self.save_y[operator])

        shap_values_array = {}

        for operator in self.dim_dict:
            print("calculate " + operator + "'s shap")

            if self.save_X[operator] == []:
                continue

            back_samp = np.random.choice(len(self.save_X[operator]), size=100, replace=True)
            data_samp = np.random.choice(len(self.save_X[operator]), size=25, replace=True)
            background_data = np.array(self.save_X[operator])[back_samp]
            data = np.array(self.save_X[operator])[data_samp]

            if 'tree' in self.filter_type:
                pass
            else:
                background_data = torch.tensor(background_data).to(self.device)
                data = torch.tensor(data).to(self.device)

            del self.save_X[operator]

            print(background_data.shape)
            print(data.shape)

            if 'tree' in self.filter_type:
                explainer = shap.TreeExplainer(filter_models[operator], background_data)
            elif 'grad' in self.filter_type:
                explainer = shap.GradientExplainer(filter_models[operator], background_data)
            else:
                explainer = shap.DeepExplainer(filter_models[operator], background_data)
            shap_values = explainer.shap_values(data)
            n_array = np.array(shap_values)

            sum_array = []
            write_info_array = []
            flag = 1

            if 'tree' in self.filter_type:
                for n in range(n_array.shape[1]):
                    if float(np.sum(np.abs(n_array[:, n]))) > 0:
                        sum_array.append(n)
                        write_info_array.append(float(np.sum(np.abs(n_array[:, n]))))
                        flag = 0
                if flag == 1:
                    for n in range(n_array.shape[0]):
                        sum_array.append(n)
            else:
                for n in range(n_array.shape[2]):
                    if float(np.sum(np.abs(n_array[:, :, n]))) > 0:
                        sum_array.append(n)
                        write_info_array.append(float(np.sum(np.abs(n_array[:, :, n]))))
                        flag = 0
                if flag == 1:
                    for n in range(n_array.shape[0]):
                        sum_array.append(n)

            shap_values_array[operator] = sum_array

            # 某特征在预测值
            with open(self.save_dir + "/" + self.filter_type + "_info.txt", "a") as f:
                f.write(operator + ": " + ",".join([str(s) for s in write_info_array]) + "\n")

            fig = plt.figure()
            shap.summary_plot(shap_values, data, show=False)
            plt.savefig(self.save_dir + "/" + operator, bbox_inches='tight')

            with open(self.save_dir + "/" + self.filter_type + "_values_array.pickle", "wb") as f:
                pickle.dump(shap_values_array, f)

        return shap_values_array

    def calculate_GREEDY(self, eval_dataset):

        self.save_X = {}
        self.save_y = {}

        for operator in self.dim_dict:  # 对回归器进行复用
            self.save_X[operator] = []
            self.save_y[operator] = []

        self.test = True
        self.set_input(eval_dataset)
        self.eval = True
        self.save = True
        self._forward(0)

        # filter_type = 'Tree'
        filter_type = 'Net'
        GREEDY_models = {}

        if 'Tree' == filter_type:
            for operator in self.dim_dict:
                GREEDY_models[operator] = RandomForestRegressor(max_depth=9, n_estimators=50, warm_start=False,
                                                            random_state=1)
        else:
            GREEDY_models = self.units

        for operator in self.dim_dict:
            if self.save_X[operator] != []:
                self.save_X[operator] = np.array([item.cpu().detach().numpy() for item in self.save_X[operator]])
                self.save_y[operator] = np.array(self.save_y[operator])
                if 'Tree' == filter_type:
                    GREEDY_models[operator].fit(self.save_X[operator], self.save_y[operator])

        GREEDY_values_array = {}

        for operator in self.dim_dict:
            print("calculate " + operator + "'s R2")

            if self.save_X[operator] == []:
                continue

            TrainX = self.save_X[operator]
            TrainY = self.save_y[operator]

            print(TrainX.shape)

            if 'Tree' == filter_type:
                TrainX = TrainX
                TrainY = TrainY
            else:
                TrainX = torch.tensor(TrainX).to(self.device)
                TrainY = torch.tensor(TrainY).to(self.device)

            del self.save_X[operator]

            if 'Tree' == filter_type:
                R2 = TreeR2(GREEDY_models[operator], TrainX, TrainY)
            else:
                R2 = DeepR2(GREEDY_models[operator], TrainX, TrainY, operator, self.dim_dict)

            GREEDY_values_array[operator] = np.array(R2.filter_values)

            print(GREEDY_values_array[operator].shape)

            with open(self.save_dir + "/greedy_values_array.pickle", "wb") as f:
                pickle.dump(GREEDY_values_array, f)

        return GREEDY_values_array

    def get_current_losses(self):
        return self.curr_losses

    def save_units(self, epoch):
        for name, unit in self.units.items():
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)

            if torch.cuda.is_available():
                torch.save(unit.state_dict(), save_path)
                unit.to(self.device)
            else:
                torch.save(unit.cpu().state_dict(), save_path)

    def load_unchange(self, epoch):
        for name in self.units:
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir.replace("transfer", "").replace("_template", ""), save_filename)
            if not os.path.exists(save_path):
                raise ValueError("model {} doesn't exist".format(save_path))

            try:
                self.units[name].load_state_dict(torch.load(save_path))
            except:
                print(name)

    def load(self, epoch):
        for name in self.units:
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            if not os.path.exists(save_path):
                raise ValueError("model {} doesn't exist".format(save_path))

            try:
                self.units[name].load_state_dict(torch.load(save_path))
            except:
                print(name)

    @get_time
    def transform(self, input_vec, samp=0):
        try:
            train_x = pd.DataFrame(input_vec, columns=[str(col) for col in range(input_vec.shape[1])])
        except:
            train_x = pd.DataFrame(input_vec, columns=['1'])

        return train_x

    @get_time
    def filterfunc(self, input_vec, filter_array, samp):

        if samp != 0:
            samp = np.random.choice(np.arange(len(input_vec)),
                                    min(300, int(len(input_vec))), replace=False)
            try:
                input_vec = np.array([item.cpu().detach().numpy() for item in input_vec])[samp]
            except:
                input_vec = np.array(input_vec)[samp]
        else:
            try:
                input_vec = np.array([item.cpu().detach().numpy() for item in input_vec])
            except:
                input_vec = np.array(input_vec)

        try:
            return input_vec[:, filter_array]
        except:
            return input_vec

    @get_time
    def filterfunc1(self, input_vec, filter_array):
        return input_vec[:, filter_array]
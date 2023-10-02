import os
import pickle
from pprint import pprint

import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, dataset

from greedy.deepMSCN import DeepR2
from greedy.tree import TreeR2
from utils import metric
from utils.util import get_time


def squared_diff(output, target):
    return torch.sum((output - target) ** 2)


# Define model architecture

class MSCN(nn.Module):
    def __init__(self, table_feats, predicate_feats, hid_units):
        super(MSCN, self).__init__()
        self.table_feats = table_feats
        self.predicate_feats = predicate_feats

        self.table_mlp1 = nn.Linear(table_feats, hid_units)
        self.table_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 2, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, train):
        # tables, predicates, table_mask, predicate_mask = train[0], train[1], train[2], train[3]
        tables, predicates, table_mask, predicate_mask = train[:, :, :self.table_feats], \
                                                         train[:, :,
                                                         self.table_feats:self.table_feats + self.predicate_feats], \
                                                         train[:, :, -2].reshape(-1, 1, 1), \
                                                         train[:, :, -1].reshape(-1, 1, 1)

        hid_table = F.relu(self.table_mlp1(tables))
        hid_table = F.relu(self.table_mlp2(hid_table))
        hid_table = hid_table * table_mask  # Mask
        hid_table = torch.sum(hid_table, dim=1, keepdim=False)
        table_norm = table_mask.sum(1, keepdim=False)
        hid_table = hid_table / table_norm  # Calculate average only over non-masked parts

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        hid_predicate = hid_predicate / predicate_norm

        hid = torch.cat((hid_table, hid_predicate), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out


class MSCNModel():
    def __init__(self, opt, dim_dict):
        self.test_dataset = None
        self.device = torch.device('cuda:0') \
            if torch.cuda.is_available() \
            else torch.device('cpu:0')
        self.save_dir = opt.mid_data_dir + opt.save_dir + "/" + str(opt.batch_size)
        self.save = False
        self.eval = False
        self.batch_size = opt.batch_size
        self.dataset = opt.dataset

        # self.num_epochs = opt.end_epoch
        self.num_epochs = 50

        self.last_total_loss = None
        self.last_pred_err = None
        self.pred_err = None

        if not os.path.exists(opt.mid_data_dir + opt.save_dir):
            os.mkdir(opt.mid_data_dir + opt.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.dim_dict = dim_dict

        # Initialize the neural units
        self.best = 100000000000

        filter_type = opt.mid_data_dir.split("_")[-1]

        self.filter = True if "snapshot_model_" in opt.mid_data_dir else False
        if os.path.exists("./2000/" + opt.mid_data_dir.split("/")[-2] + "/snapshot_model/save_model_MSCN/"
                          + str(opt.batch_size) + "/" + filter_type + "_values_array.pickle"):

            with open("./2000/" + opt.mid_data_dir.split("/")[-2] + "/snapshot_model/save_model_MSCN/"
                      + str(opt.batch_size) + "/" + filter_type + "_values_array.pickle", "rb") as f:
                self.save_values_array = pickle.load(f)

            if self.filter:
                end1 = self.dim_dict["table_list_len"] + self.dim_dict["maxlen_plan"]
                end2 = self.dim_dict["table_list_len"] + self.dim_dict["maxlen_plan"] + 1
                self.table_list = [i for i in self.save_values_array if i < self.dim_dict["table_list_len"]]
                self.dim_dict["table_list_len"] = len(self.table_list)
                self.plan_list = [i for i in self.save_values_array
                                  if self.dim_dict["table_list_len"] < i < self.dim_dict["maxlen_plan"] + self.dim_dict[
                                      "table_list_len"]]
                self.dim_dict["maxlen_plan"] = len(self.plan_list)
                self.filter_list = self.table_list
                self.filter_list.extend(self.plan_list)
                self.filter_list.extend([end1, end2])

        self.model = MSCN(self.dim_dict["table_list_len"], self.dim_dict["maxlen_plan"], 256).cuda()

        if opt.SGD:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=opt.lr, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=opt.lr)  # opt.lr
        if opt.scheduler:
            self._scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opt.step_size,
                                                  gamma=opt.gamma)
        self.scheduler = opt.scheduler

        self.loss_fn = squared_diff
        # Initialize the global loss accumulator dict
        self.dummy = torch.zeros(1).to(self.device)
        self.total_loss = None
        self._test_losses = dict()

        if opt.start_epoch != 0:
            self.load(opt.start_epoch)

    def unnormalize_labels(self, labels_norm, min_val, max_val):
        labels_norm = np.array(labels_norm, dtype=np.float32)
        labels = (labels_norm * (max_val - min_val)) + min_val

        # return np.array(np.round(np.exp(labels)), dtype=np.float32)
        return np.array(np.exp(labels), dtype=np.float32)

    def unnormalize_torch(self, vals, min_val, max_val):
        vals = (vals * (max_val - min_val)) + min_val
        return torch.exp(vals)

    def qerror_loss(self, preds, targets, min_val, max_val):
        qerror = []
        preds = self.unnormalize_torch(preds, min_val, max_val)
        targets = self.unnormalize_torch(targets, min_val, max_val)

        for i in range(len(targets)):
            if (preds[i] > targets[i]).cpu().data.numpy()[0]:
                qerror.append(preds[i] / targets[i])
            else:
                qerror.append(targets[i] / preds[i])

        # return torch.mean((preds - targets) ** 2)
        return torch.mean(torch.cat(qerror))

    def predict(self, model, data_loader):
        preds = []
        model.eval()

        for batch_idx, data_batch in enumerate(data_loader):

            trainX, targets = data_batch  # 转换数据为张量
            outputs = model(trainX)

            for i in range(outputs.data.shape[0]):
                preds.append(outputs.data[i])

        return np.array([item.cpu().detach().numpy() for item in preds])

    def set_input(self, samp_dicts):
        self.input = samp_dicts

    def backward(self, loss):
        self.last_total_loss = loss.item()
        if self.best > loss.item():
            self.best = loss.item()
            self.save_units('best')
        loss.backward()

    def model_train(self, num_epochs):

        if not self.test:
            print("begin train")
            self.model.train()
        else:
            print("begin eval")

        if self.eval:
            self.total_times = []
            self.pred_times = []
            self.total_costs = []
            self.plan_times = []

        for epoch in range(num_epochs):
            self.total_loss = 0.

            for batch_idx, data_batch in enumerate(self.input):
                trainX, targets = data_batch

                self.optimizer.zero_grad()

                outputs = self.model(trainX)

                loss = self.qerror_loss(outputs, targets.float(), self.dim_dict["min_val"], self.dim_dict["max_val"])
                self.total_loss += loss.item()
                self.backward(loss)
                self.optimizer.step()

                if self.scheduler:
                    self._scheduler.step()

            print("Epoch {}, loss: {}".format(epoch, self.total_loss / len(self.input)))

        preds_times = self.predict(self.model, self.input)

        preds_times = self.unnormalize_labels(preds_times,
                                              self.dim_dict["min_val"],
                                              self.dim_dict["max_val"]).flatten()

        total_times = self.unnormalize_labels(
            np.hstack([batch[1].cpu().detach().numpy() for batch in self.input]),
            self.dim_dict["min_val"],
            self.dim_dict["max_val"])

        if self.eval:
            self.pred_times.append(preds_times)
            self.total_times.append(total_times)

        print("\nQ-Error training set:")
        q_error_list = metric.Metric.q_error_numpy(total_times, preds_times, 0.01)
        print(q_error_list)

        self.last_test_loss = np.mean(np.abs(total_times - preds_times))
        self.last_pred_err = np.mean(metric.Metric.pred_err_numpy(total_times, preds_times, 0.01))

    def optimize_parameters(self, epoch):

        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.test = False

        if self.filter:
            self.input = self.filterfunc(self.input)

        self.input = DataLoader(self.input, batch_size=self.batch_size)
        self.model_train(self.num_epochs)

    def evaluate(self, eval_dataset):

        self.test = True

        if self.filter:
            eval_dataset = self.filterfunc(eval_dataset)

        self.input = DataLoader(eval_dataset, batch_size=self.batch_size)
        self.eval = True
        self.model_train(0)

        with open(self.save_dir + "/pred_times.pickle", "wb") as f:
            pickle.dump(self.pred_times, f)
        with open(self.save_dir + "/total_times.pickle", "wb") as f:
            pickle.dump(self.total_times, f)
        with open(self.save_dir + "/total_costs.pickle", "wb") as f:
            pickle.dump(self.total_costs, f)
        with open(self.save_dir + "/plan_times.pickle", "wb") as f:
            pickle.dump(self.plan_times, f)

    def load(self, epoch):
        save_filename = '%s_net_%s.pth' % (epoch, 'MSCN')
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.exists(save_path):
            raise ValueError("model {} doesn't exist".format(save_path))
        self.model.load_state_dict(torch.load(save_path))

    def save_units(self, epoch):
        save_filename = '%s_net_%s.pth' % (epoch, 'MSCN')
        save_path = os.path.join(self.save_dir, save_filename)

        if torch.cuda.is_available():
            torch.save(self.model.state_dict(), save_path)
            self.model.to(self.device)
        else:
            torch.save(self.model.cpu().state_dict(), save_path)

    def calculate_FR(self, eval_dataset):

        self.test = True
        self.input = eval_dataset
        self.eval = True
        self.save = True
        self.model_train(0)

        filter_type = 'grad'

        if 'Tree' == filter_type:
            filter_models = RandomForestRegressor(max_depth=9, n_estimators=50, warm_start=False, random_state=1)
        else:
            filter_models = self.model

        if 'Tree' == filter_type:
            X_data = np.vstack([batch[0].cpu().detach().numpy() for batch in self.input])
            X_data = X_data.reshape(X_data.shape[0], -1)
            target_data = np.hstack([batch[4].cpu().detach().numpy() for batch in self.input])
            print(X_data.shape)
            print(target_data.shape)
            filter_models.fit(X_data, target_data)

        trainX = []
        data = []
        for batch_idx, data_batch in enumerate(self.input):
            if 'Tree' == filter_type:
                samp1 = np.random.choice(np.arange(len(X_data)),
                                         min(500, max(int(len(X_data) / 4), 200)), replace=True)
                samp2 = np.random.choice(np.arange(len(X_data)),
                                         min(100, max(int(len(X_data) / 16), 40)), replace=True)
                trainX = X_data[samp1, :]
                data = X_data[samp2, :]
                break
            else:
                if trainX != []:
                    data, _ = data_batch
                    break
                trainX, targets = data_batch

        print(trainX.shape)

        if filter_type == 'Tree':
            explainer = shap.TreeExplainer(filter_models, trainX)
        elif filter_type == 'shap':
            explainer = shap.DeepExplainer(filter_models, trainX)
        elif filter_type == 'grad':
            explainer = shap.GradientExplainer(filter_models, trainX)

        shap_values = explainer.shap_values(data)
        n_array = np.array(shap_values)

        print(n_array.shape)

        sum_array = []
        write_info_array = []
        flag = 1

        if 'Tree' == filter_type:
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

        for i in range(self.dim_dict['table_list_len']):
            if i not in sum_array:
                sum_array.insert(0, i)

        shap_values_array = sum_array

        with open(self.save_dir + "/" + filter_type + "_values_array.pickle", "wb") as f:
            pickle.dump(shap_values_array, f)

    def calculate_GREEDY(self, eval_dataset):

        self.test = True
        self.input = eval_dataset
        self.eval = True
        self.save = True
        self.model_train(0)

        # filter_type = 'Tree'
        filter_type = 'Net'

        if 'Tree' == filter_type:
            filter_models = RandomForestRegressor(max_depth=9, n_estimators=50, warm_start=False, random_state=1)
        else:
            filter_models = self.model

        if 'Tree' == filter_type:
            TrainX = np.vstack([batch[0].cpu().detach().numpy() for batch in self.input])
            TrainY = np.vstack([batch[1].cpu().detach().numpy() for batch in self.input])
            TrainX = TrainX.reshape(TrainX.shape[0], -1)
            target_data = np.hstack([batch[4].cpu().detach().numpy() for batch in self.input])
            filter_models.fit(TrainX, target_data)

        TrainX = []
        for batch_idx, data_batch in enumerate(self.input):
            if 'Tree' == filter_type:
                samp1 = np.random.choice(np.arange(len(TrainX)),
                                         min(500, max(int(len(TrainX) / 4), 200)), replace=True)
                TrainX = TrainX[samp1, :]
                TrainY = TrainY[samp1]
                break
            else:
                TrainX, TrainY = data_batch

        print(TrainX.shape)

        if 'Tree' == filter_type:
            R2 = TreeR2(filter_models, TrainX, TrainY)
        else:
            R2 = DeepR2(filter_models, TrainX, TrainY,
                        self.dim_dict["table_list_len"], self.dim_dict["maxlen_plan"])

        GREEDY_values_array = np.array(R2.filter_values)
        print(GREEDY_values_array.shape)

        with open(self.save_dir + "/greedy_values_array.pickle", "wb") as f:
            pickle.dump(GREEDY_values_array, f)

    @get_time
    def filterfunc(self, input_vec):
        trainXs = []
        targets = []
        input_vec = DataLoader(input_vec, batch_size=512)
        for batch_idx, data_batch in enumerate(input_vec):
            trainX, target = data_batch

            trainX = trainX[:, :, self.filter_list]
            if trainXs != []:
                trainXs = torch.vstack((trainXs, trainX))
                targets = torch.hstack((targets, target))
            else:
                trainXs = trainX
                targets = target

        return dataset.TensorDataset(trainXs, targets)

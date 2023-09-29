import copy

import numpy as np
import progressbar
import torch
from torch import nn
from torch.optim import lr_scheduler

import torch.nn.functional as F


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


class DeepR2():
    def __init__(self, model, TrainX, TrainY, table_list_len, plan_list_len):
        self.scheduler = None
        self.optimizer = None
        self.device = torch.device('cuda:0') \
            if torch.cuda.is_available() \
            else torch.device('cpu:0')
        self.TrainX = TrainX
        self.TrainY = TrainY
        self.model = model

        self.table_list_len = table_list_len
        self.plan_list_len = plan_list_len

        self.filter_values = [i for i in range(self.TrainX.shape[2])]
        self.default_eval_value = self.default()

        print("default_eval_value", self.default_eval_value)

        self.R2_func()

    def init_model(self, table_list_len, plan_list_len):
        self.model = MSCN(table_list_len, plan_list_len, 256).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

    def default(self):
        result = self.model(self.TrainX)
        default_eval_value = self.calculate(result, self.TrainY)
        return default_eval_value

    def R2_func(self):
        min_col = -1

        bar = progressbar.ProgressBar(widgets=[
            progressbar.Percentage(),
            ' (', progressbar.SimpleProgress(), ') ',
            ' (', progressbar.AbsoluteETA(), ') ', ])

        for i in bar(self.filter_values):
            self.col = copy.copy(self.filter_values)
            self.col.remove(i)

            table_list_len = copy.copy(self.table_list_len)
            plan_list_len = copy.copy(self.plan_list_len)
            try:
                table_list_len -= 1
            except:
                plan_list_len -= 1

            self.init_model(table_list_len, plan_list_len)
            self.R2_train()

            result = self.model(self.TrainX[:,:, self.col])
            temp_eval_value = self.calculate(result, self.TrainY)

            if temp_eval_value < self.default_eval_value:
                print(i, temp_eval_value)
                self.default_eval_value = temp_eval_value
                min_col = i
        if min_col != -1:
            self.filter_values.remove(min_col)
            try:
                self.table_list_len -= 1
            except:
                self.plan_list_len -= 1

            self.R2_func()
        else:
            return
        return

    def R2_train(self):
        for epoch in range(50):
            result = self.model(self.TrainX[:,:, self.col])
            loss = torch.sum(torch.abs(self.TrainY - result))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def calculate(self, pred_time, true_time):
        pred_time = np.array([item.cpu().detach().numpy() for item in pred_time])
        true_time = np.array([item.cpu().detach().numpy() for item in true_time])

        epsilon = 0.01
        q_error = []
        for idx, tt in enumerate(true_time):
            if pred_time[idx] < tt:
                q_error.append((tt + epsilon) / (pred_time[idx] + epsilon))
            else:
                q_error.append((pred_time[idx] + epsilon) / (tt + epsilon))

        return np.mean(q_error)

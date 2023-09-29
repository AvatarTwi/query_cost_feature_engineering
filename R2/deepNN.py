import copy

import numpy as np
import progressbar
import torch
from torch import nn
from torch.optim import lr_scheduler


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
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True


class DeepR2():
    def __init__(self, model, TrainX, TrainY, operator, dim_dict):
        self.scheduler = None
        self.optimizer = None
        self.device = torch.device('cuda:0') \
            if torch.cuda.is_available() \
            else torch.device('cpu:0')
        self.TrainX = TrainX
        self.TrainY = TrainY
        self.model = model
        self.dim_dict = dim_dict
        self.operator = operator
        self.filter_values = [i for i in range(self.TrainX.shape[1])]
        self.default_eval_value = self.default()

        print("default_eval_value", self.default_eval_value)

        self.R2_func()

    def default(self):
        result = self.model(self.TrainX)
        default_eval_value = self.calculate(result[:, 0], self.TrainY)
        return default_eval_value

    def R2_train(self):
        # sysbench
        # tpch 10
        # job 50
        for epoch in range(400):
            result = self.model(self.TrainX[:, self.col])
            loss = torch.sum(torch.abs(self.TrainY - result[:,0]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def init_model(self):
        self.dim_dict[self.operator] = len(self.col)
        self.model = NeuralUnit(self.operator, self.dim_dict).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

    def R2_func(self):
        min_col = -1

        bar = progressbar.ProgressBar(widgets=[
            progressbar.Percentage(),
            ' (', progressbar.SimpleProgress(), ') ',
            ' (', progressbar.AbsoluteETA(), ') ', ])

        for i in bar(self.filter_values):
            self.col = copy.copy(self.filter_values)
            self.col.remove(i)

            self.init_model()
            self.R2_train()

            result = self.model(self.TrainX[:, self.col])
            temp_eval_value = self.calculate(result[:, 0], self.TrainY)
            if temp_eval_value < self.default_eval_value:
                print(i,temp_eval_value)
                self.default_eval_value = temp_eval_value
                min_col = i
        if min_col != -1:
            self.filter_values.remove(min_col)
            self.R2_func()
        else:
            return
        return

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

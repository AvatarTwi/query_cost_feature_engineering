import copy

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class TreeR2():
    def __init__(self, model, TrainX, TrainY):
        self.TrainX = TrainX
        self.TrainY = TrainY
        self.model = model
        self.filter_values = [i for i in range(self.TrainX.shape[1])]
        self.default_eval_value = self.default()
        self.R2_func()

    def default(self):
        result = self.model.predict(self.TrainX)
        default_eval_value = self.calculate(result, self.TrainY)
        return default_eval_value

    def R2_func(self):
        min_col = -1
        self.model = RandomForestRegressor(max_depth=9, n_estimators=50, warm_start=False,
                                           random_state=1)
        for i in self.filter_values:
            temp = copy.copy(self.filter_values)
            temp.remove(i)
            self.model.fit(self.TrainX[:, temp], self.TrainY)
            result = self.model.predict(self.TrainX[:, temp])
            temp_eval_value = self.calculate(result, self.TrainY)
            if temp_eval_value < self.default_eval_value:
                self.default_eval_value = temp_eval_value
                min_col = i
        if min_col != -1:
            self.filter_values.remove(min_col)
            self.R2_func()
        else:
            return
        return

    def calculate(self, pred_time, true_time):
        epsilon = 0.01
        q_error = []
        for idx, tt in enumerate(true_time):
            if pred_time[idx] < tt:
                q_error.append((tt + epsilon) / (pred_time[idx] + epsilon))
            else:
                q_error.append((pred_time[idx] + epsilon) / (tt + epsilon))

        return np.mean(q_error)

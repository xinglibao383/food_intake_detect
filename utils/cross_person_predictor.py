import torch
import time
import torch.nn as nn
from datetime import datetime
from utils import commons, watch_glasses_dataset, create_model
from d2l import torch as d2l


class CrossPersonPredictor:
    def __init__(self, train_id, test_person_index=None):
        self.device = d2l.try_gpu(0)
        self.net = commons.load_the_best_weights(create_model.get_swin_transformer_v2_1d(), train_id, "post")
        self.data_iter = watch_glasses_dataset.load_data(None, cross_person=True,
                                                         use_for_final_test=True, test_person_index=test_person_index)

    def predict(self):
        self.net.to(self.device).eval()
        # 正确预测的数量, 总预测的数量
        metric = d2l.Accumulator(2)
        with torch.no_grad():
            for X, y in self.data_iter:
                X = commons.preprocess_inputs_experiment(*X)
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                y, y_hat = torch.argmax(y, dim=1), torch.argmax(y_hat, dim=1)
                metric.add(float((y_hat == y).type(y.dtype).sum()), y.shape[0])

        return metric[0] / metric[1]
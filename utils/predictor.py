import torch
import torch.nn as nn
from models import resnet, unet
from utils import commons
from d2l import torch as d2l

class Predictor:
    def __init__(self, config):
        self.config = config

        self.device = d2l.try_gpu(0)
        self.pre_model = commons.load_the_best_weights(resnet.resnet(), self.config['pre_model_train_id'], 'pre')
        self.post_model = commons.load_the_best_weights(unet.UNet(base_c=self.config['unet_base_c']), self.config['post_model_train_id'], 'post')
        self.pre_model.to(self.device).eval()
        self.post_model.to(self.device).eval()
        self.loss_function = nn.MSELoss(reduction='none')

    def predict(self, data, mask_percentage):
        with torch.no_grad():
            data = data.to(self.device)
            data_masked = commons.apply_random_mask(data.clone(), mask_percentage, self.device)
            pre_model_result = self.pre_model(data_masked)
            loss = self.loss_function(data, pre_model_result)
            result_mask = loss < self.config['threshold']
            data = data[result_mask]
            post_model_result = self.post_model(data)
            result = [self.config['categories'][i] for i in torch.argmax(post_model_result, dim=1)]
        return result


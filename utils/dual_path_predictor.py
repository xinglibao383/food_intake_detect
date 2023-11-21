import torch
import torch.nn as nn
from models import dual_path_unet, dual_path_resnet
from utils import commons
from d2l import torch as d2l

class DualPathPredictor:
    def __init__(self, config):
        self.config = config
        self.device = d2l.try_gpu(0)
        self.loss_function = nn.MSELoss(reduction='none')

        self.pre_model = commons.load_the_best_weights(dual_path_unet.DualPathUNet(self.config['base_c_watch'], self.config['base_c_glasses']),
                                                       self.config['pre_model_train_id'], 'pre').to(self.device).eval()
        self.post_model = commons.load_the_best_weights(dual_path_resnet.DualPathResNet(),
                                                        self.config['post_model_train_id'], 'post').to(self.device).eval()


    def predict(self, data):
        with torch.no_grad():
            data = [x.to(self.device) for x in data]
            data_masked = [commons.apply_random_mask(x.clone(), self.config['mask_percentage'], self.device) for x in data]
            pre_model_result = self.pre_model(data_masked)
            dual_loss = [self.loss_function(x_hat, x) for x_hat, x in zip(pre_model_result, data)]
            loss = dual_loss[0].sum(dim=(1, 2, 3)) + dual_loss[1].sum(dim=(1, 2, 3))

            result_mask = loss < self.config['threshold']
            data = [x[result_mask] for x in data]
            post_model_result = self.post_model(data)
            result = [self.config['category_names'][i] for i in torch.argmax(post_model_result, dim=1)]
        return result


if __name__ == '__main__':
    x1, x2 = torch.rand(size=(32, 1, 512, 6)), torch.rand(size=(32, 1, 102, 6))

    pre_model = dual_path_unet.DualPathUNet(2, 2)
    post_model = dual_path_resnet.DualPathResNet()
    predictor = DualPathPredictor(config=None)

    y = predictor.predict((x1, x2))
    print(y)
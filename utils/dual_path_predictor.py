import torch
import torch.nn as nn
from models import dual_path_unet, dual_path_resnet
from utils import commons, watch_glasses_dataset
from d2l import torch as d2l


class DualPathPredictor:
    def __init__(self, config):
        self.config = config

        self.device = d2l.try_gpu(0)
        self.loss_function = nn.MSELoss(reduction='none')
        self.data_iter = watch_glasses_dataset.load_data(None, True)
        pre_model = dual_path_unet.DualPathUNet(self.config['base_c_watch'], self.config['base_c_glasses'])
        pre_model = commons.load_the_best_weights(pre_model, self.config['pre_model_train_id'], 'pre').to(self.device).eval()
        self.pre_model = pre_model
        post_model = commons.choice_which_post_model(False, self.config['post_model_train_id'])
        post_model = commons.load_the_best_weights(post_model, self.config['post_model_train_id'], 'post').to(self.device).eval()
        self.post_model = post_model

        self.predict()


    def predict(self):
        # 进食样本成功预测数量, 非进食样本成功预测数量, 分类成功数量
        metric = d2l.Accumulator(3)
        # 样本总数, 进食样本数量, 非进食样本数量
        sample_count = d2l.Accumulator(3)
        for i, (X, y) in enumerate(self.data_iter):
            if_sample_is_eat = torch.ne(torch.sum(y, dim=1), 0)
            if_sample_is_not_eat = torch.eq(torch.sum(y, dim=1), 0)
            num_sample, num_eat_sample, num_not_eat_sample = X[0].shape[0], torch.sum(if_sample_is_eat).item(), torch.sum(if_sample_is_not_eat).item()
            sample_count.add(num_sample, num_eat_sample, num_not_eat_sample)
            print(f"batch_{i:03d}: 共 {num_sample} 个样本, 其中包含 {num_eat_sample} 个进食样本和 {num_not_eat_sample} 个非进食样本")

            pre_model_result_mask, post_model_result = [temp.to('cpu') if temp is not None else temp for temp in self.predict_batch(X)]
            batch_success1_eat = [b1 == b2 and b1 == True for b1, b2 in zip(if_sample_is_eat, pre_model_result_mask)].count(True)
            batch_success1_not_eat = [b1 == b2 and b1 == False for b1, b2 in zip(if_sample_is_eat, pre_model_result_mask)].count(True)

            if post_model_result is None:
                metric.add(batch_success1_eat, batch_success1_not_eat, 0)
                print(f"batch_{i:03d}: 成功区分 {batch_success1_eat} 个进食样本和 {batch_success1_not_eat} 个非进食样本, 模型成功分类 0 个进食样本")
            else:
                # pre_model_result_mask, post_model_result = pre_model_result_mask.to('cpu'), post_model_result.to('cpu')
                y = y[pre_model_result_mask]
                # real_labels = [self.config['category_names'][k] for k in torch.argmax(y, dim=1)]
                y_max_values, y_argmax_indices = torch.max(y, dim=1)
                y_argmax_indices[y_max_values == y.min(dim=1).values] = -1
                real_labels = [self.config['category_names'][k] if k != -1 else 'none' for k in y_argmax_indices]

                predicted_labels = [self.config['category_names'][k] for k in torch.argmax(post_model_result, dim=1)]
                # batch_success = 0
                # for real_label, predicted_label in zip(real_labels, predicted_labels):
                #     if real_label == predicted_label:
                #         batch_success += 1
                #     print(f"真实标签: {real_label:<15}, 预测标签：{predicted_label:<15}")
                batch_success2 = [l1 == l2 for l1, l2 in zip(real_labels, predicted_labels)].count(True)
                metric.add(batch_success1_eat, batch_success1_not_eat, batch_success2)
                print(f"batch_{i:03d}: 成功区分 {batch_success1_eat} 个进食样本和 {batch_success1_not_eat} 个非进食样本, 模型成功分类 {batch_success2} 个进食样本")

        print(f"共计 {int(sample_count[0])} 个样本, 其中 {int(sample_count[1])} 个进食样本, {int(sample_count[2])} 个非进食样本")
        print(f"M1 成功区分 {int(metric[0]) + int(metric[1])} 个样本, 准确率为 {(int(metric[0]) + int(metric[1])) / int(sample_count[0]) * 100:.2f}%")
        print(f"其中: 成功区分 {int(metric[0])} 个进食样本, 准确率为 {int(metric[0]) / int(sample_count[1]) * 100:.2f}%")
        print(f"其中: 成功区分 {int(metric[1])} 个非进食样本, 准确率为 {int(metric[1]) / int(sample_count[2]) * 100:.2f}%")
        print(f"M1 + M2 成功分类 {int(metric[1]) + int(metric[2])} 个样本, 准确率为 {(int(metric[1]) + int(metric[2])) / int(sample_count[0]) * 100:.2f}%")


    def predict_batch(self, data, mask_or_not=True):
        with torch.no_grad():
            data = [x.to(self.device) for x in data]
            if mask_or_not:
                data_masked = [commons.apply_random_mask(x.clone(), self.config['mask_percentage'], self.device) for x in data]
                pre_model_result = self.pre_model(data_masked)
            else:
                pre_model_result = self.pre_model(data)
            dual_loss = [self.loss_function(x_hat, x) for x_hat, x in zip(pre_model_result, data)]
            loss = dual_loss[0].mean(dim=(1, 2, 3)) + dual_loss[1].mean(dim=(1, 2, 3))
            pre_model_result_mask = (loss < self.config['threshold']).to(self.device)
            data = [x[pre_model_result_mask] for x in data]

            # 所有样本都是非进食样本
            if data[0].shape[0] == 0:
                return pre_model_result_mask, None

            post_model_result = self.post_model(data)
        return pre_model_result_mask, post_model_result


if __name__ == '__main__':
    x1, x2 = torch.rand(size=(32, 1, 512, 6)), torch.rand(size=(32, 1, 102, 6))

    pre_model = dual_path_unet.DualPathUNet(2, 2)
    post_model = dual_path_resnet.DualPathResNet()
    predictor = DualPathPredictor(config=None)

    y = predictor.predict((x1, x2))
    print(y)
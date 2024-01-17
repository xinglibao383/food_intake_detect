import torch
import time
import torch.nn as nn
from datetime import datetime
from utils import commons, watch_glasses_dataset, create_model
from d2l import torch as d2l


class Predictor:
    def __init__(self, train_id1, train_id2, mask_percentage, threshold):
        self.device = d2l.try_gpu(0)
        self.loss_function = nn.MSELoss(reduction='none')
        self.data_iter = watch_glasses_dataset.load_data(None, True)
        self.pre_model = commons.load_the_best_weights(create_model.get_unet_1d(base_c=128), train_id1, "pre").to(self.device).eval()
        self.mask_percentage = mask_percentage
        self.threshold = threshold
        """
            使用归一化 2024_01_08_12_28_18
            不使用归一化 2024_01_12_16_51_48
        """
        self.post_model = commons.load_the_best_weights(create_model.get_swin_transformer_v2_1d(), train_id2, "post").to(self.device).eval()
        self.category_names = commons.get_classify_category_names()

    def predict(self):
        # 进食样本成功预测数量, 非进食样本成功预测数量, 分类成功数量
        metric = d2l.Accumulator(3)
        # 样本总数, 进食样本数量, 非进食样本数量
        sample_count = d2l.Accumulator(3)
        # 样本总数, 推理时间
        inference_info_records = []
        for i, (X, y) in enumerate(self.data_iter):
            if_sample_is_eat = torch.ne(torch.sum(y, dim=1), 0)
            if_sample_is_not_eat = torch.eq(torch.sum(y, dim=1), 0)
            num_sample, num_eat_sample, num_not_eat_sample = X[0].shape[0], torch.sum(
                if_sample_is_eat).item(), torch.sum(if_sample_is_not_eat).item()
            sample_count.add(num_sample, num_eat_sample, num_not_eat_sample)
            # print(f"batch_{i:03d}: 共 {num_sample} 个样本, 其中包含 {num_eat_sample} 个进食样本和 {num_not_eat_sample} 个非进食样本")

            start_time = time.time()
            pre_model_result_mask, post_model_result = [temp.to('cpu') if temp is not None else temp for temp in self.predict_batch(X, True)]
            end_time = time.time()
            inference_info_records.append((num_sample, int((end_time - start_time) * 1000)))

            batch_success1_eat = [b1 == b2 and b1 == True for b1, b2 in zip(if_sample_is_eat, pre_model_result_mask)].count(True)
            batch_success1_not_eat = [b1 == b2 and b1 == False for b1, b2 in zip(if_sample_is_eat, pre_model_result_mask)].count(True)

            if post_model_result is None:
                metric.add(batch_success1_eat, batch_success1_not_eat, 0)
                # print(f"batch_{i:03d}: 成功区分 {batch_success1_eat} 个进食样本和 {batch_success1_not_eat} 个非进食样本, 模型成功分类 0 个进食样本")
            else:
                y = y[pre_model_result_mask]
                y_max_values, y_argmax_indices = torch.max(y, dim=1)
                y_argmax_indices[y_max_values == y.min(dim=1).values] = -1
                real_labels = [self.category_names[k] if k != -1 else 'none' for k in y_argmax_indices]

                predicted_labels = [self.category_names[k] for k in torch.argmax(post_model_result, dim=1)]
                batch_success2 = [l1 == l2 for l1, l2 in zip(real_labels, predicted_labels)].count(True)
                metric.add(batch_success1_eat, batch_success1_not_eat, batch_success2)
                # print(f"batch_{i:03d}: 成功区分 {batch_success1_eat} 个进食样本和 {batch_success1_not_eat} 个非进食样本, 模型成功分类 {batch_success2} 个进食样本")

        messages = [f"共计 {int(sample_count[0])} 个样本, 其中 {int(sample_count[1])} 个进食样本, {int(sample_count[2])} 个非进食样本",
                   f"M1 成功区分 {int(metric[0]) + int(metric[1])} 个样本, 准确率为 {(int(metric[0]) + int(metric[1])) / int(sample_count[0]) * 100:.2f}%",
                   f"其中: 成功区分 {int(metric[0])} 个进食样本, 准确率为 {int(metric[0]) / int(sample_count[1]) * 100:.2f}%",
                   f"其中: 成功区分 {int(metric[1])} 个非进食样本, 准确率为 {int(metric[1]) / int(sample_count[2]) * 100:.2f}%",
                   f"M1 + M2 成功分类 {int(metric[1]) + int(metric[2])} 个样本, 准确率为 {(int(metric[1]) + int(metric[2])) / int(sample_count[0]) * 100:.2f}%"]
        result_data = [round((int(metric[0]) + int(metric[1])) / int(sample_count[0]) * 100, 2),
                       round((int(metric[1]) + int(metric[2])) / int(sample_count[0]) * 100, 2)]

        # 时间戳 推理结果信息 推理结果数据(M1 Acc, M1+M2 Acc) 推理时间性能
        return datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), messages, result_data, inference_info_records


    def predict_batch(self, data, mask_or_not=True):
        with torch.no_grad():
            # data = commons.preprocess_inputs(*data, channel_num=3).to(self.device)
            # data = commons.preprocess_inputs_v2(*data).to(self.device)
            # todo
            data = commons.preprocess_inputs_v3(*data, normalize=False).to(self.device)
            if mask_or_not:
                data_masked = commons.apply_random_mask(data, self.mask_percentage, self.device)
                pre_model_result = self.pre_model(data_masked)
            else:
                pre_model_result = self.pre_model(data)

            # loss = self.loss_function(data, pre_model_result).mean(dim=(1, 2, 3))
            # todo
            loss = self.loss_function(data, pre_model_result).mean(dim=(1, 2))
            # print(loss)
            pre_model_result_mask = (loss < self.threshold).to(self.device)
            data = data[pre_model_result_mask]

            # 所有样本都是非进食样本
            if data.shape[0] == 0:
                return pre_model_result_mask, None
            # todo
            # data = commons.z_score_normalize_v2(data)
            post_model_result = self.post_model(data)

        return pre_model_result_mask, post_model_result

    """
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
    """

import os
import torch.nn as nn
import numpy as np
import train_simple_v2
from d2l import torch as d2l
from datetime import datetime
from utils import logger, data_loader_v2, create_model, predictor, commons, confusion_matrix, \
    watch_glasses_dataset, train_eval_model_v2, cross_person_predictor


def threshold_and_mask_percentage_experiment():
    pre_id, post_id = "2024_01_16_15_16_05", "2024_01_12_16_51_48"
    thresholds = [0.632, 2.515, 3.805, 4.748, 5.547, 6.477, 7.41, 8.311, 9.122, 9.808, 10.7, 11.576, 12.351, 13.242,
                  14.673, 17.069, 18.887, 21.551, 25.747, 33.53, 73.956]
    mask_percentages = [0.05 * i for i in range(0, 20)]

    commons.record_experiment_results("threshold_and_mask_percentage_experiment.txt", [f'{pre_id}, {post_id}'])

    for i, threshold in enumerate(thresholds):
        for j, mask_percentage in enumerate(mask_percentages):
            result = predictor.Predictor(pre_id, post_id, mask_percentage, threshold).predict()
            timestamp, result_data = result[0], result[2]

            infos = [f'{timestamp} {i * 5} {int(mask_percentage * 100)} {result_data[0]} {result_data[1]}']
            commons.record_experiment_results("threshold_and_mask_percentage_experiment.txt", infos)


def inference_time_performance_experiment():
    pre_id, post_id = "2024_01_16_15_16_05", "2024_01_12_16_51_48"
    result = predictor.Predictor(pre_id, post_id, 80, 4.748).predict()
    inference_info_records = result[3]
    commons.record_experiment_results("inference_time_performance_experiment.txt",
                                      [f'{pre_id}, {post_id}, 推理进食/非进食数据(100:124)'])
    sample_num_count, inference_time_count = 0, 0
    for record in inference_info_records:
        sample_num_count, inference_time_count = sample_num_count + record[0], inference_time_count + record[1]
        commons.record_experiment_results("inference_time_performance_experiment.txt",
                                          [f'{sample_num_count} {inference_time_count}'])


def data_fusion_experiment_train():
    modes = ["watch", "glasses", "up_sample", "down_sample"]

    for mode in modes:
        print(f'开始训练M2: {mode}')
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        net = create_model.get_unet_1d(base_c=128, mode=mode)
        train_iter, val_iter, test_iter = data_loader_v2.load_data(4, 0.125)
        weights_save_parent_path = os.path.join("./weights", 'pre_model', timestamp)
        os.makedirs(weights_save_parent_path, exist_ok=True)
        my_logger = logger.Logger("file",
                                  os.path.join("./logs", 'pre_model', timestamp + ".txt"),
                                  os.path.join("./logs", 'pre_model', timestamp + ".yaml"))
        my_logger.record_logs([mode])

        train_eval_model_v2.train_reconstruct(net, train_iter, val_iter, 60, 0.0001, 0.75, None,
                                              5, d2l.try_all_gpus(), my_logger, weights_save_parent_path, mode)

    for mode in modes:
        print(f'开始训练M1: {mode}')
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        net = create_model.get_swin_transformer_v2_1d_experiment(mode)
        train_iter, val_iter, test_iter = data_loader_v2.load_data(16, 0.125)
        weights_save_parent_path = os.path.join("./weights", "post_model", timestamp)
        os.makedirs(weights_save_parent_path, exist_ok=True)
        my_logger = logger.Logger("file",
                                  os.path.join("./logs", "post_model", timestamp + ".txt"),
                                  os.path.join("./logs", 'post_model', timestamp + ".yaml"))
        my_logger.record_logs([mode])

        train_eval_model_v2.train_classify(net, train_iter, val_iter, 100, 0.0001, 10,
                                           d2l.try_all_gpus(), my_logger, weights_save_parent_path, mode)


def data_fusion_experiment_test():
    modes = ["watch", "glasses", "up_sample", "down_sample"]
    pre_models = ["2024_01_20_21_07_09", "2024_01_20_21_32_41", "2024_01_20_21_40_21", "2024_01_20_22_37_36"]
    post_models = ["2024_01_20_23_12_32", "2024_01_20_23_27_28", "2024_01_20_23_37_51", "2024_01_20_23_48_20"]
    thresholds = [35.376, 20.835, 18.42, 71.807]

    for i, mode in enumerate(modes):
        result_data = predictor.Predictor(pre_models[i], post_models[i], 0.15, thresholds[i], mode=mode).predict()[2]
        infos = [f'mode={mode}, pre_model={pre_models[i]}, post_model={post_models[i]}',
                 f'mask_percentage=15%, threshold={thresholds[i]}, M1 Acc={result_data[0]}, M1+M2 Acc={result_data[1]}']
        commons.record_experiment_results("data_fusion_experiment.txt", infos)


def confusion_metrix_experiment():
    post_model = commons.load_the_best_weights(create_model.get_swin_transformer_v2_1d_experiment(),
                                               "2024_01_12_16_51_48", "post").to(d2l.try_gpu(0))
    test_loader = watch_glasses_dataset.load_data("post", use_for_confusion_matrix=True)
    confusion_matrix.plot_confusion_matrix(post_model, test_loader)


def m1_validity_experiment():
    post_model = commons.load_the_best_weights(create_model.get_swin_transformer_v2_1d_experiment(),
                                               "2024_01_12_16_51_48", "post").to(d2l.try_gpu(0))
    test_loader = watch_glasses_dataset.load_data(None, use_for_final_test=True)
    commons.record_experiment_results("m1_validity_experiment.txt", ['post_model=2024_01_12_16_51_48'])

    for threshold in np.arange(0.1, 1, 0.1):
        acc, _ = train_eval_model_v2.evaluate_acc_loss_classify(post_model, test_loader, nn.CrossEntropyLoss(),
                                                                mode="up_sample", experiment_threshold=threshold)
        commons.record_experiment_results("m1_validity_experiment.txt", [f'threshold={threshold}, acc={acc}'])


def cross_person_experiment():
    for i in range(1, 11):
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        train_simple_v2.train_post_model(timestamp, cross_person=True, test_person_index=i)
        acc = cross_person_predictor.CrossPersonPredictor(timestamp, test_person_index=i).predict()
        commons.record_experiment_results("cross_person_experiment.txt", [f"{i} {acc}"])


def backbone_comparison_experiment_train():
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # train_simple_v2.train_pre_model(timestamp)
    train_simple_v2.train_post_model(timestamp)


def backbone_comparison_experiment_test():
    pre_models = ["2024_01_16_15_16_05", "2024_01_25_12_29_59"]
    post_models = ["2024_01_12_16_51_48", "2024_01_25_12_55_29"]
    thresholds = [18.887, 62.422]

    pre_model, post_model, threshold = pre_models[0], post_models[1], thresholds[0]

    result_data = predictor.Predictor(pre_model, post_model, 0.15, threshold).predict()[2]
    infos = [f'pre_model={pre_model}, post_model={post_model}',
             f'mask_percentage=15%, threshold={threshold}, M1 Acc={result_data[0]}, M1+M2 Acc={result_data[1]}']
    commons.record_experiment_results("backbone_comparison_experiment.txt", infos)


if __name__ == '__main__':
    # threshold_and_mask_percentage_experiment()
    # inference_time_performance_experiment()
    # data_fusion_experiment_train()
    # data_fusion_experiment_test()
    # confusion_metrix_experiment()
    # m1_validity_experiment()
    # cross_person_experiment_train()
    # cross_person_experiment()
    # backbone_comparison_experiment_train()
    backbone_comparison_experiment_test()

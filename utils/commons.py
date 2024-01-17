import os
import shutil
import yaml
import torch
import torch.nn as nn

from models import dual_path_resnet, dual_path_cross_vit, dual_path_unet


def record_experiment_results(file_name, logs):
    with open(os.path.join("./logs/experiments", file_name), "a", encoding='utf-8') as log_file:
        for log in logs:
            log_file.write(log + "\n")


def read_txt_lines(file_path):
    """逐行读取txt文件"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines


def load_config_yaml(file_path="./config.yaml"):
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def get_classify_category_names():
    names = os.listdir("./data/original_food_intake_data")
    return [name[3:] for name in names]


# def load_the_best_weights(model, train_id, pre_or_post):
#     model = torch.nn.DataParallel(model)
#     lines = read_txt_lines(os.path.join("./logs", f"{pre_or_post}_model", f"{train_id}.txt"))
#     pth_file_name = lines[-1].split(": ")[-1]
#     state_dict_path = os.path.join("./weights", f"{pre_or_post}_model", train_id, pth_file_name)
#     state_dict = torch.load(state_dict_path)
#     model.load_state_dict(state_dict)
#     return model


def load_the_best_weights(model, train_id, pre_or_post):
    model = torch.nn.DataParallel(model)
    state_dict_path = os.path.join("./weights", f"{pre_or_post}_model", train_id, "best_model_weights.pth")
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    return model


def create_the_pre_model(train_id):
    config = load_config_yaml(os.path.join("./logs/pre_model", f"{train_id}.yaml"))

    base_c_watch = config['train']['pre_model']['base_c_watch']
    base_c_glasses = config['train']['pre_model']['base_c_glasses']
    model = dual_path_unet.DualPathUNet(base_c_watch, base_c_glasses)

    mask_percentage = config['train']['pre_model']['base_c_watch']

    return model, mask_percentage


def choice_which_post_model(for_train, train_id=None):
    config_path = "./config.yaml" if for_train else os.path.join("./logs", "post_model", f"{train_id}.yaml")
    config = load_config_yaml(config_path)

    post_model_name = config['train']['post_model']['model_name']
    if post_model_name == 'ResNet':
        model = dual_path_resnet.DualPathResNet()
    elif post_model_name == "CrossVit":
        embed_dim = config['post_models']['cross_vit']['embed_dim']
        num_heads = config['post_models']['cross_vit']['num_heads']
        num_classes = config['data']['original_data']['category_num']
        num_layers = config['post_models']['cross_vit']['num_layers']
        mlp_dim = config['post_models']['cross_vit']['mlp_dim']
        dropout = config['post_models']['cross_vit']['dropout']

        model = dual_path_cross_vit.CrossViT(1, 1, embed_dim, num_heads, num_classes, num_layers, mlp_dim, dropout)

    return model


def apply_random_mask(samples, mask_percentage=0.8, device=None):
    random_numbers = torch.rand(samples.shape)
    mask = (random_numbers > mask_percentage).float().to(device)
    masked_samples = samples * mask
    return masked_samples


def clean_all_logs_weights():
    """清除所有的日志和权重"""
    # logs_parent_path weights_parent_path
    parent_paths = (os.path.join("..", "logs"), os.path.join("..", "weights"))
    for parent_path in parent_paths:
        for model_path in ("pre_model", "post_model"):
            path = os.path.join(parent_path, model_path)
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)


def z_score_normalize(imu_data):
    reshaped_imu_data = imu_data.view(-1, 6)

    mean_vals = torch.mean(reshaped_imu_data, dim=0)
    std_vals = torch.std(reshaped_imu_data, dim=0)
    normalized_imu_data = (reshaped_imu_data - mean_vals) / std_vals

    return normalized_imu_data.view(imu_data.shape)


def min_max_normalize(imu_data):
    reshaped_imu_data = imu_data.view(-1, 6)

    min_vals, _ = torch.min(reshaped_imu_data, dim=0)
    max_vals, _ = torch.max(reshaped_imu_data, dim=0)
    normalized_imu_data = (reshaped_imu_data - min_vals) / (max_vals - min_vals)

    return normalized_imu_data.view(imu_data.shape)


def preprocess_inputs(input1, input2, channel_num=3):
    # input1 = min_max_normalize(input1)
    # input2 = min_max_normalize(input2)

    # input1 = z_score_normalize(input1)
    # input2 = z_score_normalize(input2)

    input1 = torch.cat([input1.repeat(1, 1, 1, 22)], dim=3).repeat(1, channel_num, 1, 1)[:, :, :, :128]
    input2 = torch.cat([input2.repeat_interleave(5, dim=-2).repeat(1, 1, 1, 22)], dim=3).repeat(1, channel_num, 1, 1)[:, :, :, :128]
    input2 = torch.cat([input2[:, :, 0:1, :], input2, input2[:, :, -1:, :]], dim=2)

    input = torch.cat((input1, input2), dim=-1)

    return input


def z_score_normalize_v2(imu_data):
    """imu_data batch_size * 12 * 512"""
    imu_data = imu_data.permute(0, 2, 1)
    reshaped_imu_data = imu_data.reshape(-1, 12)

    mean_vals = torch.mean(reshaped_imu_data, dim=0)
    std_vals = torch.std(reshaped_imu_data, dim=0)
    normalized_imu_data = (reshaped_imu_data - mean_vals) / std_vals

    return normalized_imu_data.reshape(imu_data.shape).permute(0, 2, 1)


def preprocess_inputs_v2(input1, input2, normalize=True):
    input1, input2 = input1.squeeze(1).permute(0, 2, 1), input2.squeeze(1).permute(0, 2, 1)
    input1 = input1[:, :, 1:input1.shape[2] - 1][:, :, ::5]
    input = torch.cat((input1, input2), dim=1)[:, :, 1:101]

    if normalize == True:
        return z_score_normalize_v2(input)
    return input


def preprocess_inputs_v3(input1, input2, normalize=True):
    input1, input2 = input1.squeeze(1).permute(0, 2, 1), input2.squeeze(1).permute(0, 2, 1)
    input2 = input2.repeat_interleave(5, dim=-1)
    input2 = torch.cat((input2[:, :, 0:1], input2, input2[:, :, -2:-1]), dim=-1)
    input = torch.cat((input1, input2), dim=1)

    if normalize == True:
        return z_score_normalize_v2(input)
    return input


if __name__ == '__main__':
    # config = load_config_yaml()
    # print(config)
    clean_all_logs_weights()
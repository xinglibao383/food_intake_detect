import os
import shutil
import yaml
import torch


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


def load_the_best_weights(model, train_id, pre_or_post):
    model = torch.nn.DataParallel(model)
    lines = read_txt_lines(os.path.join("./logs", f"{pre_or_post}_model", f"{train_id}.txt"))
    pth_file_name = lines[-1].split(": ")[-1]
    state_dict_path = os.path.join("./weights", f"{pre_or_post}_model", train_id, pth_file_name)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
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



if __name__ == '__main__':
    # config = load_config_yaml()
    # print(config)
    clean_all_logs_weights()
import os
import yaml
import torch
from utils import data_segment


def load_config_yaml(file_path="config.yaml"):
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def get_classify_categories():
    categories = os.listdir("../data/original_food_intake_data")
    return [category[3:] for category in categories]


def load_the_best_weights(model, train_id, pre_or_post):
    lines = data_segment.read_txt_lines(os.path.join("../logs", f"{pre_or_post}_model", f"{train_id}.txt"))
    pth_file_name = lines[-1].split(": ")[-1]
    model.load_state_dict(os.path.join("../weights", f"{pre_or_post}_model", train_id, pth_file_name))
    return model


def apply_random_mask(samples, mask_percentage=0.8, device=None):
    random_numbers = torch.rand(samples.shape)
    mask = (random_numbers > mask_percentage).float().to(device)
    masked_samples = samples * mask
    return masked_samples


if __name__ == '__main__':
    load_the_best_weights(None, "2023_11_18_22_25_16", 'pre')
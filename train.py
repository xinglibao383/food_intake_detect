import os

from datetime import datetime
from d2l import torch as d2l
from models import dual_path_resnet, dual_path_unet, dual_path_cross_vit
from utils import logger, commons, train_eval_post_model, train_eval_pre_model, watch_glasses_dataset


def train_pre_model(config, timestamp):
    logging_mode = config['train']['logging_mode']
    base_c_watch = config['train']['pre_model']['base_c_watch']
    base_c_glasses = config['train']['pre_model']['base_c_glasses']
    num_epochs = config['train']['pre_model']['num_epochs']
    learning_rate = config['train']['pre_model']['learning_rate']
    mask_percentage = config['train']['pre_model']['mask_percentage']
    patience = config['train']['pre_model']['patience']

    net = dual_path_unet.DualPathUNet(base_c_watch, base_c_glasses)
    train_iter, test_iter = watch_glasses_dataset.load_data('pre')
    weights_save_parent_path = os.path.join("./weights", 'pre_model', timestamp)
    os.makedirs(weights_save_parent_path, exist_ok=True)
    logs_file_save_path = os.path.join("./logs", 'pre_model', timestamp + ".txt")
    config_file_save_path = os.path.join("./logs", 'pre_model', timestamp + ".yaml")
    my_logger = logger.Logger(logging_mode, logs_file_save_path, config_file_save_path)

    train_eval_pre_model.train(net, train_iter, test_iter, num_epochs, learning_rate, mask_percentage, patience,
                               d2l.try_all_gpus(), my_logger, weights_save_parent_path)


def train_post_model(config, timestamp):
    logging_mode = config['train']['logging_mode']
    num_epochs = config['train']['post_model']['num_epochs']
    learning_rate = config['train']['post_model']['learning_rate']
    patience = config['train']['post_model']['patience']

    """
    # 以下代码封装到 commons.py 内部
    post_model_name = config['train']['post_model']['model_name']
    if post_model_name == 'ResNet':
        net = dual_path_resnet.DualPathResNet()
    elif post_model_name == "CrossVit":
        embed_dim = config['post_models']['cross_vit']['embed_dim']
        num_heads = config['post_models']['cross_vit']['num_heads']
        num_classes = config['data']['original_data']['category_num']
        num_layers = config['post_models']['cross_vit']['num_layers']
        mlp_dim = config['post_models']['cross_vit']['mlp_dim']
        dropout = config['post_models']['cross_vit']['dropout']

        net = dual_path_cross_vit.CrossViT(1, 1, embed_dim, num_heads, num_classes, num_layers, mlp_dim, dropout)
    """

    net = commons.choice_which_post_model(True)
    train_iter, test_iter = watch_glasses_dataset.load_data('post')
    weights_save_parent_path = os.path.join("./weights", "post_model", timestamp)
    os.makedirs(weights_save_parent_path, exist_ok=True)
    logs_file_save_path = os.path.join("./logs", "post_model", timestamp + ".txt")
    config_file_save_path = os.path.join("./logs", 'post_model', timestamp + ".yaml")
    my_logger = logger.Logger(logging_mode, logs_file_save_path, config_file_save_path)

    train_eval_post_model.train(net, train_iter, test_iter, num_epochs, learning_rate, patience,
                                d2l.try_all_gpus(), my_logger, weights_save_parent_path)


if __name__ == "__main__":
    config = commons.load_config_yaml()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # train_pre_model(config, timestamp)
    train_post_model(config, timestamp)
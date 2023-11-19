import os

from datetime import datetime
from d2l import torch as d2l
from models import resnet, unet
from utils import logger, commons, train_eval_post_model, train_eval_pre_model, dataset


def train_pre_model(config, timestamp):
    logging_mode = config['train']['logging_mode']

    base_c = config['train']['post_model']['basc_c']
    num_epochs = config['train']['post_model']['num_epochs']
    learning_rate = config['train']['post_model']['learning_rate']
    mask_percentage = config['train']['post_model']['mask_percentage']
    patience = config['train']['post_model']['patience']

    net = unet.UNet(base_c=base_c)
    train_iter, test_iter = dataset.load_data('pre')
    weights_save_parent_path = os.path.join("./weights", 'pre_model', timestamp)
    os.makedirs(weights_save_parent_path, exist_ok=True)
    logs_file_save_path = os.path.join("./logs", 'pre_model', timestamp + ".txt")
    my_logger = logger.Logger(logging_mode, logs_file_save_path)

    train_eval_pre_model.train(net, train_iter, test_iter, num_epochs, learning_rate, mask_percentage, patience,
                               d2l.try_all_gpus(), my_logger, weights_save_parent_path)


def train_post_model(config, timestamp):
    logging_mode = config['train']['logging_mode']

    num_epochs = config['train']['pre_model']['num_epochs']
    learning_rate = config['train']['pre_model']['learning_rate']
    patience = config['train']['pre_model']['patience']

    net = resnet.resnet()
    train_iter, test_iter = dataset.load_data('post')
    weights_save_parent_path = os.path.join("./weights", "post_model", timestamp)
    os.makedirs(weights_save_parent_path, exist_ok=True)
    logs_file_save_path = os.path.join("./logs", "post_model", timestamp + ".txt")
    my_logger = logger.Logger(logging_mode, logs_file_save_path)

    train_eval_post_model.train(net, train_iter, test_iter, num_epochs, learning_rate, patience,
                     d2l.try_all_gpus(), my_logger, weights_save_parent_path)


if __name__ == "__main__":
    config = commons.load_config_yaml()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    train_post_model(config, timestamp)
    # train_pre_model(config, timestamp)
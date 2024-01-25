import os
from datetime import datetime
from d2l import torch as d2l
from utils import logger, data_loader_v2, train_eval_model_v2, create_model
from models.Autoencoder_1D import Autoencoder
from models.resnet_1D import *


def train_pre_model(timestamp, cross_person=False):
    # net = create_model.get_segformer()
    # net = unet_v2.UNet()
    net = Autoencoder(dropout=0.1)
    # net = create_model.get_unet_1d(base_c=128)
    train_iter, val_iter, test_iter = data_loader_v2.load_data(8, 0.125, cross_person)
    weights_save_parent_path = os.path.join("./weights", 'pre_model', timestamp)
    os.makedirs(weights_save_parent_path, exist_ok=True)
    my_logger = logger.Logger("file",
                              os.path.join("./logs", 'pre_model', timestamp + ".txt"),
                              os.path.join("./logs", 'pre_model', timestamp + ".yaml"))

    train_eval_model_v2.train_reconstruct(net, train_iter, val_iter, 50, 0.0001, 0.75, None,
                                          10, d2l.try_all_gpus(), my_logger, weights_save_parent_path)


def train_post_model(timestamp, cross_person=False, test_person_index=None):
    # net = create_model.get_swin_transformer_1d()
    # net = create_model.get_swin_transformer_v2_1d()
    net = resnet()
    train_iter, val_iter, test_iter = data_loader_v2.load_data(16, 0.125, cross_person, test_person_index)
    weights_save_parent_path = os.path.join("./weights", "post_model", timestamp)
    os.makedirs(weights_save_parent_path, exist_ok=True)
    my_logger = logger.Logger("file",
                              os.path.join("./logs", "post_model", timestamp + ".txt"),
                              os.path.join("./logs", 'post_model', timestamp + ".yaml"))

    train_eval_model_v2.train_classify(net, train_iter, val_iter, 50, 0.0001, 5,
                                       d2l.try_all_gpus(), my_logger, weights_save_parent_path)


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    train_pre_model(timestamp)
    # train_post_model(timestamp)
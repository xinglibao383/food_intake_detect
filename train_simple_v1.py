import os
from datetime import datetime
from d2l import torch as d2l
from utils import logger, watch_glasses_dataset, train_eval_model_v2, create_model


def train_pre_model(timestamp):
    # net = create_model.get_segformer()
    net = create_model.get_swin_transformer_with_segmentation_head()
    train_iter, test_iter = watch_glasses_dataset.load_data('pre')
    weights_save_parent_path = os.path.join("./weights", 'pre_model', timestamp)
    os.makedirs(weights_save_parent_path, exist_ok=True)
    my_logger = logger.Logger("file",
                              os.path.join("./logs", 'pre_model', timestamp + ".txt"),
                              os.path.join("./logs", 'pre_model', timestamp + ".yaml"))

    train_eval_model_v2.train_reconstruct(net, train_iter, test_iter, 50, 0.0001, 0.75, 10,
                                          d2l.try_all_gpus(), my_logger, weights_save_parent_path)


def train_post_model(timestamp):
    net = create_model.get_swin_transformer()
    train_iter, test_iter = watch_glasses_dataset.load_data('post')
    weights_save_parent_path = os.path.join("./weights", "post_model", timestamp)
    os.makedirs(weights_save_parent_path, exist_ok=True)
    my_logger = logger.Logger("file",
                              os.path.join("./logs", "post_model", timestamp + ".txt"),
                              os.path.join("./logs", 'post_model', timestamp + ".yaml"))

    train_eval_model_v2.train_classify(net, train_iter, test_iter, 3, 0.0003, 5,
                                       d2l.try_all_gpus(), my_logger, weights_save_parent_path)


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # train_pre_model(timestamp)
    train_post_model(timestamp)
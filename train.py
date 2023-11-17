import os

from datetime import datetime
from d2l import torch as d2l
from models import resnet
from utils import logger, load_config, train_eval, dataset


def main():
    config = load_config.load_config_yaml()
    num_epochs = config['model']['train']['num_epochs']
    learning_rate = config['model']['train']['learning_rate']
    patience = config['model']['train']['patience']
    logging_mode = config['model']['train']['logging_mode']
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    net = resnet.resnet()
    train_iter, test_iter = dataset.load_data()
    logs_file_save_path = os.path.join("./logs", timestamp + ".txt")
    weights_save_parent_path = os.path.join("./weights", timestamp)
    os.makedirs(weights_save_parent_path, exist_ok=True)
    my_logger = logger.Logger(logging_mode, logs_file_save_path)

    train_eval.train(net, train_iter, test_iter, num_epochs, learning_rate, patience,
                     d2l.try_all_gpus(), my_logger, weights_save_parent_path)


if __name__ == "__main__":
    main()

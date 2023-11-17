import os
import numpy as np
import scipy.io
from torch.utils.data import Dataset, DataLoader
from utils import data_segment, load_config

class SelfDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y


def load_data_from_disk(dir_root_path, train):
    acc_gyr_data, label_data = [], []
    category_dirs = [f.name for f in os.scandir(dir_root_path) if f.is_dir()]

    for dir in category_dirs:
        if train:
            mats_dir = os.path.join(dir_root_path, dir, "train", "eat")
        else:
            mats_dir = os.path.join(dir_root_path, dir, "test", "eat")
        if not (os.path.exists(mats_dir) and os.listdir(mats_dir)):
            continue
        mats = [os.path.join(mats_dir, f) for f in os.listdir(mats_dir) if os.path.isfile(os.path.join(mats_dir, f))]
        for mat in mats:
            mat_data = scipy.io.loadmat(mat)

            acc = np.array(mat_data["accData"])
            gyr = np.array(mat_data["gyrData"])
            acc_gyr = np.concatenate((acc, gyr), axis=1).reshape((1, 512, 6))

            one_hot_label = np.zeros(11)
            one_hot_label[mat_data["label"][0][0] - 1] = 1

            acc_gyr_data.append(acc_gyr)
            label_data.append(one_hot_label)

    return np.array(acc_gyr_data), np.array(label_data)


def load_data():
    config = load_config.load_config_yaml()

    generated_data_save_path = config['data']['generated_data']['save_path']
    sample_length = config['model']['dataset']['sample_length']
    stride = config['model']['dataset']['stride']
    cross_person = config['model']['dataset']['cross_person']
    train_person_num = config['model']['dataset']['train_person_num']
    train_ratio = config['model']['dataset']['train_ratio']
    batch_size = config['model']['train']['data_loader']['batch_size']
    shuffle = config['model']['train']['data_loader']['shuffle']
    num_workers = config['model']['train']['data_loader']['num_workers']

    if cross_person:
        load_parent_path = os.path.join(generated_data_save_path,
                                        "{}_{}_{}_cross_person".format(sample_length, stride, train_person_num))
        if not os.path.exists(load_parent_path):
            data_segment.save_all_data(sample_length, stride, cross_person, train_person_num, train_ratio)
    else:
        load_parent_path = os.path.join(generated_data_save_path,
                                        "{}_{}_{}_not_cross_person".format(sample_length, stride, train_ratio))
        if not os.path.exists(load_parent_path):
            data_segment.save_all_data(sample_length, stride, cross_person, train_person_num, train_ratio)

    train_acc_gyr_data, train_label_data = load_data_from_disk(load_parent_path, True)
    test_acc_gyr_data, test_label_data = load_data_from_disk(load_parent_path, False)
    print(f"共载入 {train_acc_gyr_data.shape[0]} 条训练数据, {test_acc_gyr_data.shape[0]} 条测试数据")

    return (DataLoader(SelfDataset(train_acc_gyr_data, train_label_data), batch_size, shuffle=shuffle, num_workers=num_workers),
            DataLoader(SelfDataset(test_acc_gyr_data, test_label_data), batch_size, shuffle=shuffle, num_workers=num_workers))
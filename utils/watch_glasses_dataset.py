import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from utils import data_segment_v2, commons


class WatchGlassesDataset(Dataset):
    def __init__(self, data_watch, data_glasses, targets, transform=None):
        self.data_watch = data_watch
        self.data_glasses = data_glasses
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x_watch = self.data_watch[index]
        x_glasses = self.data_glasses[index]
        if self.transform:
            x_watch = self.transform(x_watch)
            x_glasses = self.transform(x_glasses)

        x = (x_watch, x_glasses)
        y = self.targets[index]

        return x, y


def load_data_from_disk(dir_root_path, sample_length, train, eat=True):
    data_watch, data_glasses, data_label = [], [], []
    category_dirs = [f.name for f in os.scandir(dir_root_path) if f.is_dir()]
    for dir in category_dirs:
        for device in ["watch", "glasses"]:
            train_or_not = "train" if train else "test"
            eat_or_not = "eat" if eat else "not_eat"
            watch_mats_dir = os.path.join(dir_root_path, dir, device, train_or_not, eat_or_not)

            if not (os.path.exists(watch_mats_dir) and os.listdir(watch_mats_dir)):
                continue
            mats = [os.path.join(watch_mats_dir, f) for f in os.listdir(watch_mats_dir) if os.path.isfile(os.path.join(watch_mats_dir, f))]
            for mat in mats:
                # print(mat)
                mat_data = scipy.io.loadmat(mat)

                acc = np.array(mat_data["accData"])
                gyr = np.array(mat_data["gyrData"])
                # if acc.shape[0] != sample_length and acc.shape[0] != (sample_length // 5):
                #     continue
                new_shape = (1, sample_length, 6) if device == "watch" else (1, sample_length // 5, 6)
                acc_gyr = np.concatenate((acc, gyr), axis=1).reshape(new_shape)

                if device == "watch":
                    data_watch.append(acc_gyr)

                    one_hot_label = np.zeros(11)
                    one_hot_label[mat_data["label"][0][0] - 1] = 1
                    data_label.append(one_hot_label)
                else:
                    data_glasses.append(acc_gyr)

    return np.array(data_watch), np.array(data_glasses), np.array(data_label)


def load_data(pre_or_post, use_for_final_test=False):
    config = commons.load_config_yaml()

    generated_data_save_path = config['data']['generated_data']['save_path']
    sample_length = config['train']['dataset']['sample_length']
    stride = config['train']['dataset']['stride']
    cross_person = config['train']['dataset']['cross_person']
    train_person_num = config['train']['dataset']['train_person_num']
    train_ratio = config['train']['dataset']['train_ratio']

    if_pre = 'pre_model' if pre_or_post == 'pre' else 'post_model'
    batch_size = config['train'][if_pre]['data_loader']['batch_size']
    shuffle = config['train'][if_pre]['data_loader']['shuffle']
    num_workers = config['train'][if_pre]['data_loader']['num_workers']

    if cross_person:
        load_parent_path = os.path.join(generated_data_save_path, f"{sample_length}_{stride}_{train_person_num}_cross_person")
    else:
        load_parent_path = os.path.join(generated_data_save_path, f"{sample_length}_{stride}_{train_ratio}_not_cross_person")
    if not os.path.exists(load_parent_path):
        data_segment_v2.save_all_data(load_parent_path, sample_length, stride, cross_person, train_person_num, train_ratio)

    if use_for_final_test:
        # test_eat_data_watch, test_eat_data_glasses, test_eat_data_label = load_data_from_disk(load_parent_path, sample_length, False, True)
        # test_not_eat_data_watch, test_not_eat_data_glasses, test_not_eat_data_label = load_data_from_disk(load_parent_path, sample_length, False, False)
        test_eat_data = load_data_from_disk(load_parent_path, sample_length, False, True)
        test_not_eat_data = load_data_from_disk(load_parent_path, sample_length, False, False)

        test_data = [torch.cat((torch.tensor(data1), torch.tensor(data2)), dim=0) for data1, data2 in zip(test_eat_data, test_not_eat_data)]
        print(f"共载入 {test_data[0].shape[0]} 条进食和非进食数据用于最终测试, 其中包含 {test_eat_data[0].shape[0]} 条进食数据")
        return DataLoader(WatchGlassesDataset(*test_data), batch_size, shuffle=False, num_workers=num_workers)

    # train_data_watch, train_data_glasses, train_data_label
    train_data = load_data_from_disk(load_parent_path, sample_length, True, True)
    # test_data_watch, test_data_glasses, test_data_label
    test_data = load_data_from_disk(load_parent_path, sample_length, False, True)

    print(f"共载入 {train_data[0].shape[0]} 条训练数据, {test_data[0].shape[0]} 条测试数据")

    return (DataLoader(WatchGlassesDataset(*train_data), batch_size, shuffle=shuffle, num_workers=num_workers),
            DataLoader(WatchGlassesDataset(*test_data), batch_size, shuffle=shuffle, num_workers=num_workers))


if __name__ == '__main__':
    # for data in load_data_from_disk("../data/512_128_0.8_not_cross_person", 512, True, False):
    #     print(data.shape)

    train_iter, test_iter = load_data('post')
    for X, y in train_iter:
        print(isinstance(X, list), len(X), isinstance(X[0], list), X[0].shape, X[1].shape)


    # data_iter = load_data(None, True)
    # for X, y in data_iter:
    #     print(isinstance(X, list), len(X), isinstance(X[0], list), X[0].shape, X[1].shape, y.shape)
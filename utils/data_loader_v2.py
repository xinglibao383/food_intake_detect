import os
import torch
from torch.utils.data import DataLoader
from utils import watch_glasses_dataset, data_segment_v2


def load_data(batch_size, val_ratio=0.125, cross_person=False, test_person_index=None):
    if cross_person:
        load_parent_path = os.path.join("./data", f"512_128_{test_person_index}_cross_person")
    else:
        load_parent_path = os.path.join("./data", "512_128_0.8_not_cross_person")
    if not os.path.exists(load_parent_path):
        data_segment_v2.save_all_data(load_parent_path, 512, 128, cross_person, test_person_index, 0.125)

    train_val_data = watch_glasses_dataset.load_data_from_disk(load_parent_path, 512, True, True)
    shuffled_indices = torch.randperm(train_val_data[0].shape[0])
    train_val_data = [temp[shuffled_indices] for temp in train_val_data]

    train_data_len = int(len(train_val_data[0]) * (1 - val_ratio))
    train_data = [temp[:train_data_len] for temp in train_val_data]
    val_data = [temp[train_data_len:] for temp in train_val_data]
    test_data = watch_glasses_dataset.load_data_from_disk(load_parent_path, 512, False, True)

    print(f"共载入 {train_data[0].shape[0]} 条训练数据, {val_data[0].shape[0]} 条验证数据, {test_data[0].shape[0]} 条测试数据")

    return (DataLoader(watch_glasses_dataset.WatchGlassesDataset(*train_data), batch_size, shuffle=True, num_workers=4),
            DataLoader(watch_glasses_dataset.WatchGlassesDataset(*val_data), batch_size, shuffle=True, num_workers=4),
            DataLoader(watch_glasses_dataset.WatchGlassesDataset(*test_data), batch_size, shuffle=True, num_workers=4))

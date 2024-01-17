import os.path

import pandas as pd
import numpy as np


def read_data(file_path):
    grouped_data = pd.read_csv(file_path, sep='\t').groupby('设备编号')

    data_dict = {}
    for device_id, group in grouped_data:
        numpy_array = group[['加速度X', '加速度Y', '加速度Z', '角速度X', '角速度Y', '角速度Z']].values.astype(float)
        data_dict[device_id] = numpy_array

    return data_dict


def split_data(data, length, interval):
    rows, cols = data.shape
    num_samples = (rows - length) // interval + 1
    last_sample_start = (num_samples - 1) * interval
    if last_sample_start + length > rows:
        num_samples -= 1

    result = np.zeros((num_samples, length, cols), dtype=data.dtype)
    for i in range(num_samples):
        start = i * interval
        result[i] = data[start:start + length, :]

    return result


def load(parent_path='./data/supplementary_raw_data'):
    file_paths = ['daily_activities_xlb.txt', 'daily_activities_lb.txt',
                  'daily_activities_xhm.txt', 'daily_activities_yzk.txt']
    file_paths = [os.path.join(parent_path, file_name) for file_name in file_paths]

    watch_samples, glasses_samples = np.empty((0, 512, 6)), np.empty((0, 102, 6))

    for file_path in file_paths:
        data_dict = read_data(file_path)

        watch_data = split_data(data_dict['WT5300006766'], 512, 128)
        glasses_data = split_data(data_dict['WT5300006727'], 102, 26)
        num_watch_samples, num_glasses_samples = watch_data.shape[0], glasses_data.shape[0]
        num_samples = num_watch_samples if num_watch_samples < num_glasses_samples else num_glasses_samples
        watch_data, glasses_data = watch_data[:num_samples], glasses_data[:num_samples]

        watch_samples = np.concatenate((watch_samples, watch_data), axis=0)
        glasses_samples = np.concatenate((glasses_samples, glasses_data), axis=0)

    watch_samples, glasses_samples = watch_samples[:, np.newaxis, :, :], glasses_samples[:, np.newaxis, :, :]
    labels = np.zeros((watch_samples.shape[0], 11))

    return watch_samples, glasses_samples, labels

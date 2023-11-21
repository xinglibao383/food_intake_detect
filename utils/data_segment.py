import os
import numpy as np
import pandas as pd
import scipy.io
import concurrent.futures
import time
from datetime import datetime
from utils import commons


def get_start_end_timestamp_index_eat(file_path):
    file = pd.read_csv(file_path)
    video_start_frame_eat = file['video_start_frame']
    video_end_frame_eat = file['video_end_frame']
    start_end_timestamp_index_eat = list(zip(video_start_frame_eat, video_end_frame_eat))
    return start_end_timestamp_index_eat


def get_start_end_timestamp_eat(range_csv_path, timestamp_txt_path):
    timestamps = commons.read_txt_lines(timestamp_txt_path)
    start_end_timestamp_index_eat = get_start_end_timestamp_index_eat(range_csv_path)
    start_end_timestamp_eat = []
    for i in range(len(start_end_timestamp_index_eat)):
        start_index = start_end_timestamp_index_eat[i][0]
        end_index = start_end_timestamp_index_eat[i][1]
        start_timestamp = datetime.strptime(timestamps[start_index].replace(' ', 'T').strip(), '%Y-%m-%dT%H:%M:%S.%f')
        end_timestamp = datetime.strptime(timestamps[end_index].replace(' ', 'T').strip(), '%Y-%m-%dT%H:%M:%S.%f')
        start_end_timestamp_eat.append((start_timestamp, end_timestamp))
    return start_end_timestamp_eat


def get_single_sensor_data(start_end_timestamp_eat, sensor_path):
    """获取一类中一个人的一个传感器数据"""
    with open(sensor_path, 'r') as file:
        data = [line.split() for line in file]

    for i in range(0, len(data)):
        # 0.899 0.512 -0.035 2022-10-15T12:21:20.337
        # 0.895 0.493 -0.016  2022-10-15 12:16:05.340
        if len(data[i]) == 5:
            data[i] = [data[i][0], data[i][1], data[i][2], data[i][3] + 'T' + data[i][4]]
        data[i][3] = datetime.strptime(data[i][3], '%Y-%m-%dT%H:%M:%S.%f')

    data_eat, data_not_eat = [], []
    for row in data:
        flag = 0
        for i in range(len(start_end_timestamp_eat)):
            if row[3] < start_end_timestamp_eat[i][0] or row[3] > start_end_timestamp_eat[i][1]:
                flag += 1
        if flag != len(start_end_timestamp_eat):
            data_eat.append(row)
        else:
            data_not_eat.append(row)

    return data_eat, data_not_eat


def data_split(data_acc, data_gyr, sample_length, stride):
    result = []
    length = min(len(data_acc), len(data_gyr)) - sample_length

    for i in range(0, length, stride):
        align_index, align_time_diff = i, abs((data_acc[i][3] - data_gyr[i][3]).total_seconds() * 1000)
        for j in range(0, length):
            if abs((data_acc[i][3] - data_gyr[j][3]).total_seconds() * 1000) < align_time_diff:
                align_index = j
                align_time_diff = abs((data_acc[i][3] - data_gyr[j][3]).total_seconds() * 1000)

        row_batch = []
        for k in range(sample_length):
            acc_row = data_acc[i + k]
            gyr_row = data_gyr[align_index + k]
            new_row = []
            new_row.extend(acc_row[0: 3])
            new_row.extend(gyr_row[0: 3])
            row_batch.append(new_row)
        result.append(row_batch)


    print("处理结果: acc数据总条数: {}, gyr数据总条数: {}, 切割得到样本数量: {}".format(str(len(data_acc)).ljust(5), str(len(data_gyr)).ljust(5), str(len(result)).ljust(3)))

    return result


def get_split_sensor_data(range_csv_path, timestamp_txt_path, sensor_acc_path, sensor_gyr_path, sample_length, stride):
    start_end_timestamp_eat = get_start_end_timestamp_eat(range_csv_path, timestamp_txt_path)

    data_eat_acc, data_not_eat_acc = get_single_sensor_data(start_end_timestamp_eat, sensor_acc_path)
    data_eat_gyr, data_not_eat_gyr = get_single_sensor_data(start_end_timestamp_eat, sensor_gyr_path)

    eat_result = data_split(data_eat_acc, data_eat_gyr, sample_length, stride)
    not_eat_result = data_split(data_not_eat_acc, data_not_eat_gyr, sample_length, stride)

    return eat_result, not_eat_result


def save_111_data(parent_path, data, category_index, person_index, train, eat):
    """保存一个类别下一个人的进食或非进食的训练/测试数据"""
    data_len = len(data)
    zfill_size = 3 if data_len > 100 else 2

    parent_path = os.path.join(parent_path, str(category_index).zfill(2), "watch")
    parent_path = os.path.join(parent_path, "train") if train else os.path.join(parent_path, "test")
    parent_path = os.path.join(parent_path, "eat") if eat else os.path.join(parent_path, "not_eat")
    os.makedirs(os.path.dirname(parent_path), exist_ok=True)

    for i in range(0, data_len):
        acc_data = np.array([row[0: 3] for row in data[i]], dtype=np.single)
        gyr_data = np.array([row[3: 6] for row in data[i]], dtype=np.single)

        label_value = category_index if eat else 0
        label_data = np.full(len(acc_data), label_value, dtype=np.int32).reshape((len(acc_data), 1))

        person_data = np.full(len(acc_data), person_index, dtype=np.int32).reshape((len(acc_data), 1))

        mat_file_name = "{}_{}_{}.mat".format(str(category_index).zfill(2), str(person_index).zfill(2),
                                              str(i).zfill(zfill_size))
        mat_file_path = os.path.join(parent_path, mat_file_name)
        os.makedirs(os.path.dirname(mat_file_path), exist_ok=True)
        scipy.io.savemat(mat_file_path, {"accData": acc_data, "gyrData": gyr_data, "label": label_data, "person": person_data})


def save_112_data(parent_path, eat_data, not_eat_data, category_index, person_index, train):
    """保存一个类别下一个人的进食和非进食的训练/测试数据"""
    # save_111_data(parent_path, eat_data, category_index, person_index, train, True)
    # save_111_data(parent_path, not_eat_data, category_index, person_index, train, False)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        start_time = time.time()

        task1 = executor.submit(save_111_data, parent_path, eat_data, category_index, person_index, train, True)
        task2 = executor.submit(save_111_data, parent_path, not_eat_data, category_index, person_index, train, False)

        concurrent.futures.wait([task1, task2], return_when=concurrent.futures.ALL_COMPLETED)

        end_time = time.time()
        execution_time = end_time - start_time
        train_or_test = "训练" if train else "测试"
        print(f"第 {category_index} 类的第 {person_index} 个人的的进食和非进食的{train_or_test}数据保存成功. 耗时: {execution_time} 秒")
        if task1.result() is not None:
            print(f"任务1执行失败, 结果: {task1.result()}")
        if task2.result() is not None:
            print(f"任务2执行失败, 结果: {task2.result()}")

    executor.shutdown()


def save_all_data(sample_length, stride, cross_person, train_person_num, train_ratio):
    config = commons.load_config_yaml()

    original_data_root_path = config['data']['original_data']['root_path']
    generated_data_save_path = config['data']['generated_data']['save_path']
    category_num = config['data']['original_data']['category_num']
    person_num = config['data']['original_data']['person_num_per_category']

    if cross_person:
        data_save_root_path = os.path.join(generated_data_save_path, f"{sample_length}_{stride}_{train_person_num}_cross_person")
    else:
        data_save_root_path = os.path.join(generated_data_save_path, f"{sample_length}_{stride}_{train_ratio}_not_cross_person")
    category_dirs = [entry.name for entry in os.scandir(original_data_root_path) if entry.is_dir()]

    # 根据类别分别处理并保存数据
    for i in range(1, category_num + 1):
        # 获取处理并保存当前类别数据所必须的文件夹目录
        category_dir = os.path.join(original_data_root_path, category_dirs[i - 1])
        person_dirs = [f.name for f in os.scandir(category_dir) if f.is_dir()]
        for j in range(1, person_num + 1):
            print(f"正在处理第 {i} 类的第 {j} 个人的数据...")

            temp = os.path.join(category_dir, person_dirs[j - 1])
            range_csv_path = os.path.join(temp, "data.csv")
            timestamp_txt_path = os.path.join(temp, "time.txt")
            sensor_acc_path = os.path.join(temp, "Others", "accData.txt")
            sensor_gyr_path = os.path.join(temp, "Others", "gyrData.txt")

            # 一个类别下一个人的进食和非进食数据
            eat_data, not_eat_data = get_split_sensor_data(range_csv_path, timestamp_txt_path, sensor_acc_path,
                                                           sensor_gyr_path, sample_length, stride)

            if cross_person:
                # 划分训练数据与验证数据
                is_train_person = j <= train_person_num
                save_112_data(data_save_root_path, eat_data, not_eat_data, i, j, is_train_person)
            else:
                # 划分训练数据与验证数据
                # random.shuffle(eat_data), random.shuffle(not_eat_data)
                eat_data_len, not_eat_data_len = len(eat_data), len(not_eat_data)
                eat_data_partition_num, not_eat_data_partition_num = int(eat_data_len * train_ratio), int(
                    not_eat_data_len * train_ratio)
                eat_data_train, eat_data_test = eat_data[:eat_data_partition_num], eat_data[eat_data_partition_num:]
                not_eat_data_train, not_eat_data_test = not_eat_data[:not_eat_data_partition_num], not_eat_data[not_eat_data_partition_num:]

                save_112_data(data_save_root_path, eat_data_train, not_eat_data_train, i, j, True)
                save_112_data(data_save_root_path, eat_data_test, not_eat_data_test, i, j, False)


"""
def save_all_data_cross_person(original_data_root_path, generated_data_save_path,
                               category_num, person_num, sample_length, stride, train_person_num):
    data_save_root_path = os.path.join(generated_data_save_path, f"{sample_length}_{stride}_{train_person_num}_cross_person")
    category_dirs = [entry.name for entry in os.scandir(original_data_root_path) if entry.is_dir()]

    # 根据类别分别处理并保存数据
    for i in range(1, category_num + 1):
        # 获取处理并保存当前类别数据所必须的文件夹目录
        category_dir = os.path.join(original_data_root_path, category_dirs[i - 1])
        person_dirs = [f.name for f in os.scandir(category_dir) if f.is_dir()]
        for j in range(1, person_num + 1):
            print(f"正在处理第 {i} 类的第 {j} 个人的数据...")

            temp = os.path.join(category_dir, person_dirs[j - 1])
            range_csv_path = os.path.join(temp, "data.csv")
            timestamp_txt_path = os.path.join(temp, "time.txt")
            sensor_acc_path = os.path.join(temp, "Others", "accData.txt")
            sensor_gyr_path = os.path.join(temp, "Others", "gyrData.txt")

            # 一个类别下一个人的进食和非进食数据
            eat_data, not_eat_data = get_split_sensor_data(range_csv_path, timestamp_txt_path, sensor_acc_path,
                                                           sensor_gyr_path, sample_length, stride)

            # 划分训练数据与验证数据
            is_train_person = j <= train_person_num
            save_112_data(data_save_root_path, eat_data, not_eat_data, i, j, is_train_person)


def save_all_data_not_cross_person(original_data_root_path, generated_data_save_path,
                                   category_num, person_num, sample_length, stride, train_ratio):
    data_save_root_path = os.path.join(generated_data_save_path, f"{sample_length}_{stride}_{train_ratio}_not_cross_person")
    category_dirs = [entry.name for entry in os.scandir(original_data_root_path) if entry.is_dir()]

    # 根据类别分别处理并保存数据
    for i in range(1, category_num + 1):
        # 获取处理并保存当前类别数据所必须的文件夹目录
        category_dir = os.path.join(original_data_root_path, category_dirs[i - 1])
        person_dirs = [f.name for f in os.scandir(category_dir) if f.is_dir()]
        for j in range(1, person_num + 1):
            print(f"正在处理第 {i} 类的第 {j} 个人的数据...")

            temp = os.path.join(category_dir, person_dirs[j - 1])
            range_csv_path = os.path.join(temp, "data.csv")
            timestamp_txt_path = os.path.join(temp, "time.txt")
            sensor_acc_path = os.path.join(temp, "Others", "accData.txt")
            sensor_gyr_path = os.path.join(temp, "Others", "gyrData.txt")

            # 一个类别下一个人的进食和非进食数据
            eat_data, not_eat_data = get_split_sensor_data(range_csv_path, timestamp_txt_path, sensor_acc_path, sensor_gyr_path,
                                                           sample_length, stride)

            # 划分训练数据与验证数据
            random.shuffle(eat_data), random.shuffle(not_eat_data)
            eat_data_len, not_eat_data_len = len(eat_data), len(not_eat_data)
            eat_data_partition_num, not_eat_data_partition_num = int(eat_data_len * train_ratio), int(not_eat_data_len * train_ratio)
            eat_data_train, eat_data_test = eat_data[:eat_data_partition_num], eat_data[eat_data_partition_num:]
            not_eat_data_train, not_eat_data_test = not_eat_data[:not_eat_data_partition_num], not_eat_data[not_eat_data_partition_num:]

            save_112_data(data_save_root_path, eat_data_train, not_eat_data_train, i, j, True)
            save_112_data(data_save_root_path, eat_data_test, not_eat_data_test, i, j, False)


def save_all_data(sample_length, stride, cross_person, train_person_num, train_ratio):
    config = commons.load_config_yaml()

    original_data_root_path = config['data']['original_data']['root_path']
    generated_data_save_path = config['data']['generated_data']['save_path']
    category_num = config['data']['original_data']['category_num']
    person_num = config['data']['original_data']['person_num_per_category']

    if cross_person:
        save_all_data_cross_person(original_data_root_path, generated_data_save_path,
                                   category_num, person_num, sample_length, stride, train_person_num)
    else:
        save_all_data_not_cross_person(original_data_root_path, generated_data_save_path,
                                       category_num, person_num, sample_length, stride, train_ratio)
"""

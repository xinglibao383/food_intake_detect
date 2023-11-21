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


def get_watch_single_data(start_end_timestamp_eat, sensor_path):
    """获取一类中一个人的手表的一个传感器数据"""
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

    return np.array(data_eat), np.array(data_not_eat)


def get_glasses_data(start_end_timestamp_eat, sensor_path):
    """获取一类中一个人的眼镜传感器数据"""
    table = pd.read_table(sensor_path, sep='\t')
    table['接收时间'] = table['接收时间'].str.strip().str.replace(' ', 'T')
    table['接收时间'] = pd.to_datetime(table['接收时间'], format='%Y-%m-%dT%H:%M:%S.%f')
    selected_columns = ['接收时间', '加速度X', '加速度Y', '加速度Z', '角速度X', '角速度Y', '角速度Z']
    data = table[selected_columns].to_numpy()

    data_eat, data_not_eat = [], []
    for row in data:
        # array([Timestamp('2022-10-15 12:16:05.340000'), 0.895, 0.493, -0.016, -0.732, -0.488, 0.915], dtype=object)
        flag = 0
        for i in range(len(start_end_timestamp_eat)):
            if row[0] < start_end_timestamp_eat[i][0] or row[0] > start_end_timestamp_eat[i][1]:
                flag += 1
        if flag != len(start_end_timestamp_eat):
            data_eat.append(row)
        else:
            data_not_eat.append(row)
    return np.array(data_eat), np.array(data_not_eat)


def data_split(data_watch_acc, data_watch_gyr, data_glasses, sample_length, stride):
    splited_data_watch, splited_data_glasses = [], []
    watch_length = min(len(data_watch_acc), len(data_watch_gyr)) - sample_length
    glasses_length = len(data_glasses) - sample_length // 5

    for i in range(0, watch_length, stride):
        # 将手表陀螺仪数据与手表加速度计数据对齐
        align_watch_gyr_index, align_watch_gyr_time_diff = i, abs((data_watch_acc[i][3] - data_watch_gyr[0][3]).total_seconds() * 1000)
        for j in range(0, watch_length):
            if abs((data_watch_acc[i][3] - data_watch_gyr[j][3]).total_seconds() * 1000) < align_watch_gyr_time_diff:
                align_watch_gyr_index = j
                align_watch_gyr_time_diff = abs((data_watch_acc[i][3] - data_watch_gyr[j][3]).total_seconds() * 1000)

        data_watch_acc_sub_matrix = data_watch_acc[i: i + sample_length, 0: 3]
        data_watch_gyr_sub_matrix = data_watch_gyr[align_watch_gyr_index: align_watch_gyr_index + sample_length, 0: 3]
        data_watch_sub_matrix = np.concatenate((data_watch_acc_sub_matrix, data_watch_gyr_sub_matrix), axis=1)
        splited_data_watch.append(data_watch_sub_matrix)

        # 将眼镜数据与手表加速度计数据对齐
        align_glasses_index, align_glasses_time_diff = i, abs((data_watch_acc[i][3] - data_glasses[0][0]).total_seconds() * 1000)
        for k in range(0, glasses_length):
            if abs((data_watch_acc[i][3] - data_glasses[k][0]).total_seconds() * 1000) < align_glasses_time_diff:
                align_glasses_index = k
                align_glasses_time_diff = abs((data_watch_acc[i][3] - data_glasses[k][0]).total_seconds() * 1000)
        splited_data_glasses.append(np.array(data_glasses[align_glasses_index: align_glasses_index + sample_length // 5, 1:], dtype=np.single))

    print("手表处理结果: 数据总条数 {}, 切割得到样本数量 {}".format(str(min(len(data_watch_acc), len(data_watch_gyr))).ljust(5),
                                                                    str(len(splited_data_watch)).ljust(3)))
    print("眼镜处理结果: 数据总条数 {}, 切割得到样本数量 {}".format(str(len(data_glasses)).ljust(5),
                                                                    str(len(splited_data_glasses)).ljust(3)))

    return splited_data_watch, splited_data_glasses


def get_split_sensor_data(range_csv_path, timestamp_txt_path, data_origin_path, sample_length, stride):
    start_end_timestamp_eat = get_start_end_timestamp_eat(range_csv_path, timestamp_txt_path)

    # data_origin_path (watch_acc_path, watch_gyr_path, glasses_path)
    data_eat_watch_acc, data_not_eat_watch_acc = get_watch_single_data(start_end_timestamp_eat, data_origin_path[0])
    data_eat_watch_gyr, data_not_eat_watch_gyr = get_watch_single_data(start_end_timestamp_eat, data_origin_path[1])
    data_eat_glasses, data_not_eat_glasses = get_glasses_data(start_end_timestamp_eat, data_origin_path[2])


    watch_eat_data, glasses_eat_data = data_split(data_eat_watch_acc, data_eat_watch_gyr, data_eat_glasses, sample_length, stride)
    watch_not_eat_data, glasses_not_eat_data = data_split(data_not_eat_watch_acc, data_not_eat_watch_gyr, data_not_eat_glasses, sample_length, stride)

    return watch_eat_data, watch_not_eat_data, glasses_eat_data, glasses_not_eat_data


def save_111_data(parent_path, data, category_index, person_index, watch, train, eat):
    """保存一个类别下一个人的进食或非进食的训练/测试数据"""
    data_len = len(data)
    zfill_size = 3 if data_len > 100 else 2

    parent_path = os.path.join(parent_path, str(category_index).zfill(2), "watch" if watch else "glasses")
    parent_path = os.path.join(parent_path, "train") if train else os.path.join(parent_path, "test")
    parent_path = os.path.join(parent_path, "eat") if eat else os.path.join(parent_path, "not_eat")
    os.makedirs(os.path.dirname(parent_path), exist_ok=True)

    for i, sample in enumerate(data):
        acc_data = np.array(sample[:, :3], dtype=np.single)
        gyr_data = np.array(sample[:, 3:], dtype=np.single)

        label_value = category_index if eat else 0
        label_data = np.full(len(acc_data), label_value, dtype=np.int32).reshape((len(acc_data), 1))

        person_data = np.full(len(acc_data), person_index, dtype=np.int32).reshape((len(acc_data), 1))

        mat_file_name = "{}_{}_{}.mat".format(str(category_index).zfill(2), str(person_index).zfill(2),
                                              str(i).zfill(zfill_size))
        mat_file_path = os.path.join(parent_path, mat_file_name)
        os.makedirs(os.path.dirname(mat_file_path), exist_ok=True)
        scipy.io.savemat(mat_file_path, {"accData": acc_data, "gyrData": gyr_data, "label": label_data, "person": person_data})


def save_114_data(parent_path, data, category_index, person_index, train):
    """保存一个类别下一个人的进食和非进食的训练/测试数据"""

    start_time = time.time()

    # watch_eat_data, watch_not_eat_data, glasses_eat_data, glasses_not_eat_data
    # parent_path, data, category_index, person_index, watch, train, eat
    save_111_data(parent_path, data[0], category_index, person_index, True, train, True)
    save_111_data(parent_path, data[1], category_index, person_index, True, train, False)
    save_111_data(parent_path, data[2], category_index, person_index, False, train, True)
    save_111_data(parent_path, data[3], category_index, person_index, False, train, False)

    end_time = time.time()
    execution_time = end_time - start_time
    train_or_test = "训练" if train else "测试"
    print(f"第 {category_index} 类的第 {person_index} 个人的{train_or_test}数据保存成功, 耗时: {execution_time} 秒")


    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     start_time = time.time()
    #
    #     # watch_eat_data, watch_not_eat_data, glasses_eat_data, glasses_not_eat_data
    #     # parent_path, data, category_index, person_index, watch, train, eat
    #     task0 = executor.submit(save_111_data, parent_path, data[0], category_index, person_index, True, train, True)
    #     task1 = executor.submit(save_111_data, parent_path, data[1], category_index, person_index, True, train, False)
    #     task2 = executor.submit(save_111_data, parent_path, data[2], category_index, person_index, False, train, True)
    #     task3 = executor.submit(save_111_data, parent_path, data[3], category_index, person_index, False, train, False)
    #
    #     concurrent.futures.wait([task0, task1, task2, task3], return_when=concurrent.futures.ALL_COMPLETED)
    #
    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #     train_or_test = "训练" if train else "测试"
    #     print(f"第 {category_index} 类的第 {person_index} 个人的{train_or_test}数据保存成功, 耗时: {execution_time} 秒")
    #     for task in [task0, task1, task2, task3]:
    #         if task.result() is not None:
    #             print(f"任务1执行失败, 结果: {task.result()}")
    #
    # executor.shutdown()



def save_all_data(data_save_root_path, sample_length, stride, cross_person, train_person_num, train_ratio):
    # 加载配置文件并读取配置信息
    config = commons.load_config_yaml()
    original_data_root_path = config['data']['original_data']['root_path']
    generated_data_save_path = config['data']['generated_data']['save_path']
    category_num = config['data']['original_data']['category_num']
    person_num = config['data']['original_data']['person_num_per_category']

    # 创建数据存储根目录 512_128_0.7_not_cross_person
    # if cross_person:
    #     data_save_root_path = os.path.join(generated_data_save_path, f"{sample_length}_{stride}_{train_person_num}_cross_person")
    # else:
    #     data_save_root_path = os.path.join(generated_data_save_path, f"{sample_length}_{stride}_{train_ratio}_not_cross_person")

    # 获取原始数据存储根目录下的文件夹名
    category_dirs = [entry.name for entry in os.scandir(original_data_root_path) if entry.is_dir()]

    # 根据类别分别处理并保存数据
    for i in range(1, category_num + 1):
        category_dir = os.path.join(original_data_root_path, category_dirs[i - 1])
        person_dirs = [f.name for f in os.scandir(category_dir) if f.is_dir()]
        # 根据类别下的每个人分别处理并保存数据
        for j in range(1, person_num + 1):
            print(f"正在处理第 {i} 类的第 {j} 个人的数据...")

            temp = os.path.join(category_dir, person_dirs[j - 1])
            range_csv_path = os.path.join(temp, "data.csv")
            timestamp_txt_path = os.path.join(temp, "time.txt")
            watch_acc_path = os.path.join(temp, "Others", "accData.txt")
            watch_gyr_path = os.path.join(temp, "Others", "gyrData.txt")
            glasses_path = os.path.join(temp, "sensor.txt")
            data_origin_path = (watch_acc_path, watch_gyr_path, glasses_path)

            # watch_eat_data, watch_not_eat_data, glasses_eat_data, glasses_not_eat_data
            data = get_split_sensor_data(range_csv_path, timestamp_txt_path,
                                         data_origin_path, sample_length, stride)
            if cross_person:
                # 划分训练数据与验证数据
                is_train_person = j <= train_person_num
                save_114_data(data_save_root_path, data, i, j, is_train_person)
            else:
                # 划分训练数据与验证数据
                train_data, test_data = [], []
                for sub_data in data:
                    sub_data_len = len(sub_data)
                    sub_data_partition_num = int(sub_data_len * train_ratio)
                    sub_data_train, sub_data_test = sub_data[:sub_data_partition_num], sub_data[sub_data_partition_num:]
                    train_data.append(sub_data_train)
                    test_data.append(sub_data_test)
                save_114_data(data_save_root_path, train_data, i, j, True)
                save_114_data(data_save_root_path, test_data, i, j, False)


if __name__ == '__main__':
    save_all_data(512, 128, False, 8, 0.8)
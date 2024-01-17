from utils import predictor, commons


def threshold_and_mask_percentage_experiment():
    pre_id, post_id = "2024_01_16_15_16_05", "2024_01_12_16_51_48"
    thresholds = [0.632, 2.515, 3.805, 4.748, 5.547, 6.477, 7.41, 8.311, 9.122, 9.808, 10.7, 11.576, 12.351, 13.242,
                  14.673, 17.069, 18.887, 21.551, 25.747, 33.53, 73.956]
    mask_percentages = [0.05 * i for i in range(0, 20)]

    commons.record_experiment_results("threshold_and_mask_percentage_experiment.txt", [f'{pre_id}, {post_id}'])

    for i, threshold in enumerate(thresholds):
        for j, mask_percentage in enumerate(mask_percentages):
            result = predictor.Predictor(pre_id, post_id, mask_percentage, threshold).predict()
            timestamp, result_data = result[0], result[2]

            infos = [f'{timestamp} {i * 5} {int(mask_percentage * 100)} {result_data[0]} {result_data[1]}']
            commons.record_experiment_results("threshold_and_mask_percentage_experiment.txt", infos)


def inference_time_performance_experiment():
    pre_id, post_id = "2024_01_16_15_16_05", "2024_01_12_16_51_48"
    result = predictor.Predictor(pre_id, post_id, 80, 4.748).predict()
    inference_info_records = result[3]
    commons.record_experiment_results("inference_time_performance_experiment.txt", [f'{pre_id}, {post_id}, 推理进食/非进食数据(100:124)'])
    sample_num_count, inference_time_count = 0, 0
    for record in inference_info_records:
        sample_num_count, inference_time_count =  sample_num_count + record[0], inference_time_count + record[1]
        commons.record_experiment_results("inference_time_performance_experiment.txt", [f'{sample_num_count} {inference_time_count}'])


if __name__ == '__main__':
    # threshold_and_mask_percentage_experiment()
    inference_time_performance_experiment()
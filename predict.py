from utils import commons, dual_path_predictor


if __name__ == '__main__':
    category_names = commons.get_classify_category_names(),

    config_yaml = commons.load_config_yaml()
    pre_model_base_c_watch = config_yaml['deploy']['pre_model']['base_c_watch']
    pre_model_base_c_glasses = config_yaml['deploy']['pre_model']['base_c_glasses']
    pre_model_mask_percentage = config_yaml['deploy']['pre_model']['mask_percentage']
    pre_model_threshold = config_yaml['deploy']['pre_model']['threshold']
    pre_model_train_id = config_yaml['deploy']['pre_model']['train_id']
    post_model_train_id = config_yaml['deploy']['post_model']['train_id']

    config_map = {"category_names": category_names,
                  "base_c_watch": pre_model_base_c_watch,
                  "base_c_glasses": pre_model_base_c_glasses,
                  "mask_percentage": pre_model_mask_percentage,
                  "threshold": pre_model_threshold,
                  "pre_model_train_id": pre_model_train_id,
                  "post_model_train_id": post_model_train_id}

    my_predictor = dual_path_predictor.DualPathPredictor(config_map)
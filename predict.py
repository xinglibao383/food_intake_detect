from utils import predictor, commons


if __name__ == '__main__':
    config_yaml = commons.load_config_yaml()
    categories = commons.get_classify_categories(),
    pre_model_base_c = config_yaml['deploy']['pre_model']['base_c']
    pre_model_threshold = config_yaml['deploy']['pre_model']['threshold']
    pre_model_train_id = config_yaml['deploy']['pre_model']['train_id']
    post_model_train_id = config_yaml['deploy']['post_model']['train_id']
    config_map = {"categories": categories,
                  "unet_base_c": pre_model_base_c,
                  "threshold": pre_model_threshold,
                  "pre_model_train_id": pre_model_train_id,
                  "post_model_train_id": post_model_train_id}
    my_predictor = predictor.Predictor(config_map)
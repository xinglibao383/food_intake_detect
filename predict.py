from utils import commons, dual_path_predictor, predictor


if __name__ == '__main__':
    """
    category_names = commons.get_classify_category_names()

    config_yaml = commons.load_config_yaml()
    pre_model_threshold = config_yaml['deploy']['pre_model']['threshold']
    pre_model_train_id = config_yaml['deploy']['pre_model']['train_id']
    post_model_train_id = config_yaml['deploy']['post_model']['train_id']

    config_map = {"category_names": category_names,
                  "threshold": pre_model_threshold,
                  "pre_model_train_id": pre_model_train_id,
                  "post_model_train_id": post_model_train_id}

    dual_path_predictor.DualPathPredictor(config_map)
    """
    
    predictor.Predictor()



    """
    # 以下代码迁移到 DualPathPredictor() 内部
    data_iter = watch_glasses_dataset.load_data(None, True)
    metric = d2l.Accumulator(2)
    for i, (X, y) in enumerate(data_iter):
        pre_model_result_mask, post_model_result = predictor.predict(X)
        if pre_model_result_mask is None:
            print(f"batch_{i:02d}: 共 {X[0].shape[0]} 个样本, 模型全部预测为非进食样本")
            metric.add(X[0].shape[0], 0)
            continue
        pre_model_result_mask, post_model_result = pre_model_result_mask.to('cpu'), post_model_result.to('cpu')
        y = y[pre_model_result_mask]
        real_labels = [category_names[k] for k in torch.argmax(y, dim=1)]
        predicted_labels = [category_names[k] for k in torch.argmax(post_model_result, dim=1)]
        # batch_success = 0
        # for real_label, predicted_label in zip(real_labels, predicted_labels):
        #     if real_label == predicted_label:
        #         batch_success += 1
        #     print(f"真实标签: {real_label:<15}, 预测标签：{predicted_label:<15}")
        batch_success = [l1 == l2 for l1, l2 in zip(real_labels, predicted_labels)].count(True)
        print(f"batch_{i:02d}: 共 {X[0].shape[0]} 个样本, 模型成功预测 {batch_success} 个样本")
        metric.add(X[0].shape[0], batch_success)
    print(f"共计 {int(metric[0])} 个样本, 模型成功预测 {int(metric[1])} 个样本")
    """
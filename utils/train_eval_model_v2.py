import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from d2l import torch as d2l
from utils import commons


def evaluate_loss_reconstruct(net, data_iter, loss_function, mask_percentage, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    # 测试损失之和, 测试样本的总数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # todo
                X = commons.preprocess_inputs_v3(*X, normalize=False)
            X = X.to(device)
            X_masked = commons.apply_random_mask(X, mask_percentage, device)
            X_hat = net(X_masked)
            loss = loss_function(X, X_hat)
            metric.add(loss * X.shape[0], X.shape[0])
    return metric[0] / metric[1]


def evaluate_loss_with_threshold_reconstruct(net, data_iter, mask_percentage, q_list, device=None, mode="up_sample"):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    q_list = [p for p in range(0, 101, 5)] if q_list == None else q_list
    loss_function, loss_values = nn.MSELoss(reduction='none'), []
    metric = d2l.Accumulator(2)  # 测试损失之和, 测试样本的总数量

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # todo
                # X = commons.preprocess_inputs_v3(*X, normalize=False)
                X = commons.preprocess_inputs_experiment(*X, mode=mode)
            X = X.to(device)
            X_masked = commons.apply_random_mask(X, mask_percentage, device)
            X_hat = net(X_masked)
            loss = loss_function(X, X_hat)
            metric.add(torch.sum(loss.mean(dim=(1, 2))), X.shape[0])
            loss_values.extend(loss.mean(dim=(1, 2)).cpu().numpy())

    average_loss = metric[0] / metric[1]
    threshold_list = [np.percentile(loss_values, q) for q in q_list]

    return average_loss, threshold_list


def train_reconstruct(net, train_iter, test_iter, num_epochs, learning_rate, mask_percentage, q_list,
                      patience, devices, logger, weights_save_parent_path, mode="up_sample"):
    animator = None if len(devices) > 1 else d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                                                          legend=['train loss', 'test loss'])

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    # 在多个GPU上并行训练模型
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    # 使用均方误差损失函数
    loss_function = nn.MSELoss()
    timer, num_batches = d2l.Timer(), len(train_iter)

    best_weights = net.state_dict()
    best_test_loss = float('inf')
    best_test_loss_epoch = 0
    current_patience = 0

    # test_loss_values = []

    for epoch in range(num_epochs):
        # 训练损失之和, 样本数
        metric = d2l.Accumulator(2)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            if isinstance(X, list):
                # todo
                # X = commons.preprocess_inputs_v3(*X, normalize=False)
                X = commons.preprocess_inputs_experiment(*X, mode=mode)
            X = X.to(devices[0])
            X_masked = commons.apply_random_mask(X, mask_percentage, devices[0])
            X_hat = net(X_masked)
            loss = loss_function(X, X_hat)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * X.shape[0], X.shape[0])
            timer.stop()
            train_loss = metric[0] / metric[1]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                if animator != None:
                    animator.add(epoch + (i + 1) / num_batches, (train_loss, None))
                logger.record_logs([f'epoch: {epoch + 1}, data iter: {i + 1}, train loss: {train_loss:.3f}'])
        # test_loss = evaluate_loss_reconstruct(net, test_iter, loss_function, mask_percentage)
        test_loss, threshold_list = evaluate_loss_with_threshold_reconstruct(net, test_iter, mask_percentage, None, mode=mode)

        # test_loss_values.append(test_loss)
        if animator != None:
            animator.add(epoch + 1, (None, test_loss))
        logger.record_logs(
            [f'epoch: {epoch + 1}, current patience: {current_patience + 1}, test average loss: {test_loss:.3f}',
             f'threshold_list: {np.round(threshold_list, decimals=3)}'])
        weights_save_path = os.path.join(weights_save_parent_path, f"epoch_{epoch + 1}.pth")
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            torch.save(net.state_dict(), weights_save_path)
        if test_loss < best_test_loss:
            best_weights = net.state_dict()
            best_test_loss = test_loss
            best_test_loss_epoch = epoch + 1
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                logs = [f'Early stopping after {epoch + 1} epochs']
                logger.record_logs(logs)
                break

    torch.save(best_weights, os.path.join(weights_save_parent_path, "best_model_weights.pth"))
    # threshold = np.percentile(test_loss_values[:best_test_loss_epoch], 25)
    logs = [f"The best testing loss occurred in the {best_test_loss_epoch} epoch",
            # f"The threshold is {threshold}",
            f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}']
    logger.record_logs(logs)


def accuracy_classify_for_experiment(y_hat, y, threshold=0.5):
    max_values, max_indices = torch.max(F.softmax(y_hat, dim=1), dim=1)
    mask = max_values > threshold
    predictions = torch.where(mask, max_indices, torch.tensor(-1, dtype=torch.long))

    row_sum = torch.sum(y, dim=1)
    mask = row_sum != 0
    labels = torch.where(mask, torch.argmax(y, dim=1), torch.tensor(-1, dtype=torch.long))

    cmp = labels == predictions
    return float(cmp.type(y.dtype).sum()) / y.shape[0]


def accuracy_classify(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    cmp = y_hat == torch.argmax(y, dim=1)
    return float(cmp.type(y.dtype).sum()) / y.shape[0]


def evaluate_acc_loss_classify(net, data_iter, loss, device=None, mode=None, experiment_threshold=None):
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量, 测试损失之和, 总预测的数量
    metric = d2l.Accumulator(3)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # todo
                # X = commons.preprocess_inputs_v3(*X, normalize=False)
                X = commons.preprocess_inputs_experiment(*X, mode=mode)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            if experiment_threshold is not None:
                print(f"threshold={experiment_threshold}")
                metric.add(accuracy_classify_for_experiment(y_hat, y, experiment_threshold) * y.shape[0], loss(y_hat, y) * y.shape[0], y.shape[0])
            else:
                metric.add(accuracy_classify(y_hat, y) * y.shape[0], loss(y_hat, y) * y.shape[0], y.shape[0])
    return metric[0] / metric[2], metric[1] / metric[2]


def train_classify(net, train_iter, test_iter, num_epochs, learning_rate, patience,
                   devices, logger, weights_save_parent_path, mode="up_sample"):
    animator = None if len(devices) > 1 else d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                                                          legend=['train loss', 'train acc', 'test loss', 'test acc'])

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # net.apply(init_weights)

    # 在多个GPU上并行训练模型
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)

    best_weights = net.state_dict()
    best_test_loss = float('inf')
    best_test_loss_epoch = 0
    current_patience = 0

    for epoch in range(num_epochs):
        # 训练损失之和, 正确预测的数量, 样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            if isinstance(X, list):
                # todo
                # X = commons.preprocess_inputs_v3(*X, normalize=False)
                X = commons.preprocess_inputs_experiment(*X, mode=mode)
            X = X.to(devices[0])
            y = y.to(devices[0])
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * y.shape[0], accuracy_classify(y_hat, y) * y.shape[0], y.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                if animator != None:
                    animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None, None))
                logger.record_logs(
                    [f'epoch: {epoch + 1}, data iter: {i + 1}, train loss: {train_l:.3f}, train acc: {train_acc:.3f}'])
        # test_acc, test_loss = evaluate_acc_loss_classify(net, test_iter, loss)
        test_acc, test_loss = evaluate_acc_loss_classify(net, test_iter, loss, mode=mode)
        if animator != None:
            animator.add(epoch + 1, (None, None, test_loss, test_acc))
        logger.record_logs([f'epoch: {epoch + 1}, current patience: {current_patience + 1}, test loss: {test_loss:.3f}, test acc: {test_acc:.3f}'])
        weights_save_path = os.path.join(weights_save_parent_path, f"epoch_{epoch + 1}.pth")
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            torch.save(net.state_dict(), weights_save_path)
        if test_loss < best_test_loss:
            best_weights = net.state_dict()
            best_test_loss = test_loss
            best_test_loss_epoch = epoch + 1
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                logs = [f'Early stopping after {epoch + 1} epochs']
                logger.record_logs(logs)
                break

    torch.save(best_weights, os.path.join(weights_save_parent_path, "best_model_weights.pth"))
    logs = [f"The best testing loss occurred in the {best_test_loss_epoch} epoch",
            f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}']
    logger.record_logs(logs)

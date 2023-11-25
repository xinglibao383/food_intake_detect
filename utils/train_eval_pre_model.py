import os

import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
from utils import commons


def evaluate_loss(net, data_iter, loss_function, mask_percentage, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 测试损失之和, 测试样本的总数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
                X_masked = [commons.apply_random_mask(x.clone(), mask_percentage, device) for x in X]
            else:
                X = X.to(device)
                X_masked = commons.apply_random_mask(X.clone(), mask_percentage, device)
            X_hat = net(X_masked)
            if isinstance(X, list):
                loss = torch.sum(torch.stack([loss_function(x_hat, x) for x_hat, x in zip(X_hat, X)]))
                metric.add(loss * X[0].shape[0], X[0].shape[0])
            else:
                loss = loss_function(X_hat, X)
                metric.add(loss * X.shape[0], X.shape[0])
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, num_epochs, learning_rate, mask_percentage, patience, devices, logger, weights_save_parent_path):
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

    # 阈值
    test_loss_values = []

    for epoch in range(num_epochs):
        # 训练损失之和, 样本数
        metric = d2l.Accumulator(2)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            if isinstance(X, list):
                X = [x.to(devices[0]) for x in X]
                X_masked = [commons.apply_random_mask(x.clone(), mask_percentage, devices[0]) for x in X]
            else:
                X = X.to(devices[0])
                X_masked = commons.apply_random_mask(X.clone(), mask_percentage, devices[0])
            X_hat = net(X_masked)
            if isinstance(X, list):
                loss = torch.sum(torch.stack([loss_function(x_hat, x) for x_hat, x in zip(X_hat, X)]))
            else:
                loss = loss_function(X_hat, X)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if isinstance(X, list):
                    metric.add(loss * X[0].shape[0], X[0].shape[0])
                else:
                    metric.add(loss * X.shape[0], X.shape[0])
            timer.stop()
            train_loss = metric[0] / metric[1]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                if animator != None:
                    animator.add(epoch + (i + 1) / num_batches, (train_loss, None))
                logger.record_logs([f'epoch: {epoch + 1}, data iter: {i + 1}, train loss: {train_loss:.3f}'])
        test_loss = evaluate_loss(net, test_iter, loss_function, mask_percentage)
        test_loss_values.append(test_loss)
        if animator != None:
            animator.add(epoch + 1, (None, test_loss))
        logger.record_logs([f'epoch: {epoch + 1}, test loss: {test_loss:.3f}'])
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
                logs = [f'Early stopping after {epoch} epochs.']
                logger.record_logs(logs)
                break

    torch.save(best_weights, os.path.join(weights_save_parent_path, "best_model_weights.pth"))
    threshold = np.percentile(test_loss_values[:best_test_loss_epoch], 25)
    logs = [f"The threshold is {threshold}",
            f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}']
    logger.record_logs(logs)
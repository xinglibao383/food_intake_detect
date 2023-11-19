import os
import torch
from torch import nn
from d2l import torch as d2l


def accuracy(y_hat, y):
    """计算正确预测的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1)
    cmp = y_hat == torch.argmax(y, dim=1)
    return float(cmp.type(y.dtype).sum())


def evaluate_acc_loss(net, data_iter, loss, device=None):
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量, 测试损失之和, 总预测的数量
    metric = d2l.Accumulator(3)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            metric.add(accuracy(y_hat, y), loss(y_hat, y) * X.shape[0], y.shape[0])
    return metric[0] / metric[2], metric[1]


def train(net, train_iter, test_iter, num_epochs, learning_rate, patience, devices, logger, weights_save_parent_path):
    animator = None if len(devices) > 1 else d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                                                          legend=['train loss', 'train acc', 'test acc'])

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    # 在多个GPU上并行训练模型
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)

    best_test_loss = float('inf')
    best_test_loss_epoch = 0
    current_patience = 0

    for epoch in range(num_epochs):
        # 训练损失之和, 训练准确率之和, 样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                if animator != None:
                    animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
                logger.record_logs([f'epoch: {epoch + 1}, data iter: {i + 1}, train loss: {train_l:.3f}, train acc: {train_acc:.3f}'])
        test_acc, test_loss = evaluate_acc_loss(net, test_iter, loss)
        if animator != None:
            animator.add(epoch + 1, (None, None, test_acc))
        logger.record_logs([f'epoch: {epoch + 1}, test acc: {test_acc:.3f}'])
        weights_save_path = os.path.join(weights_save_parent_path, f"epoch_{epoch + 1}.pth")
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs or test_loss < best_test_loss:
            torch.save(net.state_dict(), weights_save_path)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_loss_epoch = epoch + 1
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                logs = [f'Early stopping after {epoch} epochs.',
                        f'The best test loss occurs in the {best_test_loss_epoch} epoch.']
                logger.record_logs(logs)
                break
    logs = [f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}',
            f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}',
            f"The File name for saving the best model weights: epoch_{best_test_loss_epoch}.pth"]
    logger.record_logs(logs)
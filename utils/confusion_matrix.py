import torch
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from models import dual_path_resnet
from utils import commons, watch_glasses_dataset


def plot(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """绘制混淆矩阵"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrix(model, test_iter):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_iter:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    conf_mat = confusion_matrix(all_labels, all_preds)
    class_names = commons.get_classify_category_names()

    plt.figure(figsize=(8, 8))
    plot(conf_mat, classes=class_names, normalize=True)
    plt.show()


if __name__ == '__main__':
    model = torch.nn.DataParallel(dual_path_resnet.DualPathResNet())
    weight_file_path = r"F:\PyCharmProjects\food_intake_detect\weights\post_model\2023_11_22_16_24_01\epoch_10.pth"
    state_dict = torch.load(weight_file_path)
    model.load_state_dict(state_dict)
    _, test_iter = watch_glasses_dataset.load_data('post')

    plot_confusion_matrix(model, test_iter)

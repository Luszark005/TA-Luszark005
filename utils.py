import logging
import os
import random
import shutil

import numpy as np
import torch

# NEW IMPORTS
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class Logger():
    def __init__(self, logfile):
        if os.path.exists(os.path.dirname(logfile)) != True:
            os.makedirs(os.path.dirname(logfile))
        
        self.logfile = logfile
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.INFO,
            filename=self.logfile
        )

    def info(self, msg, *args):
        msg = str(msg)
        if args:
            print(msg % args)
            self.logger.info(msg, *args)
        else:
            print(msg)
            self.logger.info(msg)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ================= NEW METRICS ================= #

def calculate_metrics(outputs, targets):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu()

    preds = torch.argmax(outputs, dim=1)

    y_true = targets.numpy()
    y_pred = preds.numpy()

    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        'f1_weighted': f1,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'mae': mae,
        'rmse': rmse
    }


def plot_confusion_matrix(outputs, targets, trait_name, save_path):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu()

    preds = torch.argmax(outputs, dim=1)

    y_true = targets.numpy()
    y_pred = preds.numpy()

    class_labels = ['Very Low', 'Low', 'High', 'Very High']

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {trait_name}')

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'cm_{trait_name}.png'))
    plt.close()


# ================= EXISTING FUNCTIONS ================= #

def generate_adaptive_LD(outputs, targets, num_classes, threshold, sharpen, T):
    LD = []
    for i in range(5):
        device = outputs[i].device
        LD.append(torch.zeros(num_classes, num_classes).to(device))

    for i in range(5):
        outputs_l = outputs[i][1:, :]
        targets_l = targets[i][1:]

        probs = torch.softmax(outputs_l, dim=1)

        for j in range(num_classes):
            idxs = np.where(targets_l.cpu().numpy()==j)[0]
            if len(idxs) > 0 and torch.mean(probs[idxs], dim=0)[j] >= threshold:
                LD[i][j] = torch.mean(probs[idxs], dim=0)
            else:
                LD[i][j] = torch.zeros(num_classes).fill_(0.4/(num_classes-1)).scatter_(0, torch.tensor(j), threshold)

        if sharpen:
            LD[i] = torch.pow(LD[i], 1/T) / torch.sum(torch.pow(LD[i], 1/T), dim=1)
        
    return LD


def get_accuracy(outputs, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)

    _, preds = outputs.topk(maxk, 1, True, True)
    preds = preds.t()
    correct = preds.eq(targets.view(1, -1).expand_as(preds))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, checkpoint='./checkpoints', filename='checkpoint.pth.tar'):
    os.makedirs(checkpoint, exist_ok=True)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
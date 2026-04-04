import logging
import os
import random
import shutil

import numpy as np
import torch

def set_random_seed(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

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
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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
            if torch.mean(probs[idxs], dim=0)[j] >= threshold:
                LD[i][j] = torch.mean(probs[idxs], dim=0)
            else:
                LD[i][j] = torch.zeros(num_classes).fill_(0.4/(num_classes-1)).scatter_(0, torch.tensor(j), threshold)

        if sharpen == True:
            LD[i] = torch.pow(LD[i], 1/T) / torch.sum(torch.pow(LD[i], 1/T), dim=1)
        
    return LD

def generate_average_weights(weights, targets, num_classes, max_weight, min_weight):
    weights_avg = []
    weights_max = []
    weights_min = []
    for j in range(5):
        weights_l = weights[j][1:]
        targets_l = targets[j][1:]

        weights_l = ((weights_l - weights_l.min()) / (weights_l.max() - weights_l.min())) * (max_weight-min_weight) + min_weight

        
        w_avg = []
        w_max = []
        w_min = []

        for i in range(num_classes):
            idxs = np.where(targets_l.cpu().numpy()==i)[0]
            weights_i = weights_l[idxs]
            w_avg.append(weights_i.mean().item())
            w_max.append(weights_i.max().item())
            w_min.append(weights_i.min().item())
        
        weights_avg.append(w_avg)
        weights_max.append(w_max)
        weights_min.append(w_min)

    return weights_avg, weights_max, weights_min

def get_accuracy(outputs, targets, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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
    if os.path.exists(checkpoint) != True:
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
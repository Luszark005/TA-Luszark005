import argparse
import math
import os
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from model import create_model
from utils import set_random_seed, Logger, AverageMeter, generate_adaptive_LD, generate_average_weights, get_accuracy, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch Training Multi-Frame')
# train configs
parser.add_argument('--epochs', default=75, type=int)
parser.add_argument('--batch_size', default=4, type=int) # TURUN DRASTIS: Mencegah OOM di GPU
parser.add_argument('--accumulation_steps', default=4, type=int) # BARU: Gradient Accumulation
parser.add_argument('--lr', default=0.0001, type=float) # TURUN DRASTIS: Standar untuk Transformer
parser.add_argument('--num_classes', default=4, type=int)
# method configs
parser.add_argument('--threshold', default=0.7, type=float)
parser.add_argument('--sharpen', action=argparse.BooleanOptionalAction)
parser.add_argument('--T', default=1.2, type=float)
parser.add_argument('--alpha', default=None, type=float)
parser.add_argument('--beta', default=3, type=int)
parser.add_argument('--max_weight', default=1.0, type=float)
parser.add_argument('--min_weight', default=0.2, type=float)
parser.add_argument('--drop_rate', default=0.2, type=float) # Sesuaikan dengan model Swin
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)
parser.add_argument('--tops', default=0.7, type=float)
parser.add_argument('--margin_1', default=0.07, type=float)
# common configs
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--data_path', default='./dataset', type=str)
parser.add_argument('--num_workers', default=4, type=int) # Sesuaikan core CPU
parser.add_argument('--device_id', default=0, type=int)

args = parser.parse_args()

best_acc=0
best_epoch = 0

# set device
device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

if args.seed is not None:
    set_random_seed(args.seed)

def main():
    global best_acc
    global best_epoch
    global device

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    logger = Logger('./results_emotions/log-'+timestamp+'.txt')
    logger.info(args)

    writer = SummaryWriter()

    # initialization LD untuk Ada-DF
    LD = [torch.zeros(args.num_classes, args.num_classes).to(device) for _ in range(5)]
    LD_maxes = []

    for i in range(5):
        for j in range(args.num_classes):
            LD[i][j] = torch.zeros(args.num_classes).fill_((1-args.threshold)/(args.num_classes-1)).scatter_(0, torch.tensor(j), args.threshold)
        if args.sharpen == True:
            LD[i] = torch.pow(LD[i], 1/args.T) / torch.sum(torch.pow(LD[i], 1/args.T), dim=1)
        LD_maxes.append(torch.max(LD[i], dim=1))

    nan = float('nan')
    weights_avg = [[nan for i in range(args.num_classes)] for _ in range(5)]
    weights_max = [[nan for i in range(args.num_classes)] for _ in range(5)]
    weights_min = [[nan for i in range(args.num_classes)] for _ in range(5)]

    patience = 5
    counter = 0

    logger.info('Load model Swin Transformer Multi-Frame...')
    model = create_model(args.num_classes, args.drop_rate, emotion=True).to(device)   

    train_loader, test_loader = get_dataloader(args.data_path, args.batch_size, args.num_workers)

    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smoothing)
    criterion_kld = nn.KLDivLoss(reduction='none')

    # BARU: Menggunakan AdamW (Wajib untuk Swin Transformer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    logger.info('Start training.')

    for epoch in range(1, args.epochs+1):
        logger.info('----------------------------------------------------------')
        logger.info('Epoch: %d, Learning Rate: %f', epoch, optimizer.param_groups[0]['lr'])

        # train (sekarang menerima accumulation_steps)
        _, train_loss_ce, train_loss_kld, alpha_1, alpha_2 = train(train_loader, model, criterion, criterion_kld, optimizer, LD, epoch, args.accumulation_steps)
        train_loss, train_acc, outputs_new, targets_new, weights_new = validate(train_loader, model, criterion, epoch, phase='train')

        LD = generate_adaptive_LD(outputs_new, targets_new, args.num_classes, args.threshold, args.sharpen, args.T)
        LD_maxes = [torch.max(LD[i], dim=1) for i in range(5)]

        weights_avg, weights_max, weights_min = generate_average_weights(weights_new, targets_new, args.num_classes, args.max_weight, args.min_weight)

        val_loss, val_acc, _, _, _ = validate(test_loader, model, criterion, epoch, phase='val')

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch   
            counter = 0
            is_best = True
        else:
            counter += 1
            is_best = False

        logger.info(f'Alpha_1: {alpha_1:.2f}, Alpha_2: {alpha_2:.2f}')
        logger.info(f'Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f}')
        logger.info(f'Best Acc: {best_acc:.2f} (Epoch {best_epoch})')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'class_distributions': [LD[i].detach() for i in range(5)],
            }, is_best, checkpoint='./checkpoints_emotion')

        if counter >= patience:
            logger.info('Early stopping trigger...')
            break 
        
        scheduler.step()

def train(train_loader, model, criterion, criterion_kld, optimizer, LD, epoch, accumulation_steps):
    if args.alpha is not None:
        alpha_1, alpha_2 = args.alpha, 1 - args.alpha
    else:
        if epoch <= args.beta:
            alpha_1, alpha_2 = math.exp(-(1-epoch/args.beta)**2), 1
        else:
            alpha_1, alpha_2 = 1, math.exp(-(1-args.beta/epoch)**2)
    
    losses, losses_ce, losses_kld, losses_rr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar.set_description(f'Epoch [{epoch}/{args.epochs}]')

    model.train()
    optimizer.zero_grad() # Pindahkan zero_grad ke luar loop untuk gradient accumulation

    for i, (images, labels, emotions, indexes) in pbar:
        images = images.to(device)
        labels = labels.permute(1, 0)
        labels = [label.to(device) for label in labels]
        emotions = emotions.to(device)

        outputs_1, outputs_2, attention_weights = model(images, emotions)
        
        total_loss, total_loss_ce, total_loss_kld, total_RR_loss = 0, 0, 0, 0

        for j in range(5):
            tops = int(args.batch_size * args.tops)

            _, top_idx = torch.topk(attention_weights[j].squeeze(), tops)
            _, down_idx = torch.topk(attention_weights[j].squeeze(), args.batch_size - tops, largest = False)

            high_group = attention_weights[j][top_idx]
            low_group = attention_weights[j][down_idx]
            diff = torch.mean(low_group) - torch.mean(high_group) + args.margin_1

            RR_loss = diff if diff > 0 else torch.tensor(0.0, device=device)
            loss_ce = criterion(outputs_1[j], labels[j]).mean()

            attention_weights[j] = attention_weights[j].squeeze(1)
            attention_weights[j] = ((attention_weights[j] - attention_weights[j].min()) / (attention_weights[j].max() - attention_weights[j].min() + 1e-8)) * (args.max_weight-args.min_weight) + args.min_weight
            attention_weights[j] = attention_weights[j].unsqueeze(1)

            targets = (1 - attention_weights[j]) * F.softmax(outputs_1[j], dim=1) + attention_weights[j] * LD[j][labels[j]]
            loss_kld = criterion_kld(F.log_softmax(outputs_2[j], dim=1), targets).sum() / args.batch_size

            loss = alpha_2 * loss_ce + alpha_1 * loss_kld + RR_loss
            
            total_loss += loss
            total_loss_ce += loss_ce
            total_loss_kld += loss_kld
            total_RR_loss += RR_loss
        
        total_loss /= 5.0
        
        # BARU: Pembagian loss untuk Gradient Accumulation
        loss_to_backward = total_loss / accumulation_steps
        loss_to_backward.backward()

        # BARU: Eksekusi optimizer hanya setiap N steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        losses.update(total_loss.item(), images.size(0))
        pbar.set_postfix(loss=losses.avg)

    return losses.avg, total_loss_ce.item()/5.0, total_loss_kld.item()/5.0, alpha_1, alpha_2

def validate(test_loader, model, criterion, epoch, phase='train'):
    losses, accs = AverageMeter(), AverageMeter()
    model.eval()

    outputs_new = [torch.ones(1, 4).to(device) for _ in range(5)]
    targets_new = [torch.ones(1).long().to(device) for _ in range(5)]
    weights_new = [torch.ones(1, 1).float().to(device) for _ in range(5)]

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    
    with torch.no_grad():
        for i, (inputs, targets, emotions, indexes) in pbar:
            inputs = inputs.to(device)
            targets = [target.to(device) for target in targets.permute(1, 0)]
            emotions = emotions.to(device)

            if phase == 'train':
                outputs, _, attention_weights = model(inputs, emotions)
            else:
                _, outputs, attention_weights = model(inputs, emotions)

            total_loss, total_acc = 0, 0
            for j in range(5):
                loss = criterion(outputs[j], targets[j]).mean()

                outputs_new[j] = torch.cat((outputs_new[j], outputs[j]), dim=0)
                targets_new[j] = torch.cat((targets_new[j], targets[j]), dim=0)
                weights_new[j] = torch.cat((weights_new[j], attention_weights[j]), dim=0)

                top1, _ = get_accuracy(outputs[j], targets[j], topk=(1, 4))
                total_loss += loss
                total_acc += top1
            
            losses.update((total_loss/5.0).item(), inputs.size(0))
            accs.update((total_acc/5.0).item(), inputs.size(0))

    return losses.avg, accs.avg, outputs_new, targets_new, weights_new
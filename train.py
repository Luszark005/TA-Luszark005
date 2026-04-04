import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from model import create_model
from utils import set_random_seed, Logger, AverageMeter, generate_adaptive_LD, generate_average_weights, get_accuracy, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch Training')
# train configs
parser.add_argument('--epochs', default=75, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--num_classes', default=4, type=int)

# method configs
parser.add_argument('--threshold', default=0.7, type=float)
parser.add_argument('--sharpen', action=argparse.BooleanOptionalAction)
parser.add_argument('--T', default=1.2, type=float)
parser.add_argument('--alpha', default=None, type=float)
parser.add_argument('--beta', default=3, type=int)
parser.add_argument('--max_weight', default=1.0, type=float)
parser.add_argument('--min_weight', default=0.2, type=float)
parser.add_argument('--drop_rate', default=0.0, type=float)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--label_smoothing', default=0.0, type=float)
parser.add_argument('--tops', default=0.7, type=float)
parser.add_argument('--margin_1', default=0.07, type=float)
# common configs
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--data_path', default='../dataset', type=str)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--device_id', default=0, type=int)

args = parser.parse_args()

best_acc=0
best_epoch = 0

# set device
device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

# set random seed
if args.seed is not None:
    set_random_seed(args.seed)

def main():
    global best_acc
    global best_epoch
    global device

    # log file
    # logger = Logger('./results/log-'+time.strftime('%b%d_%H-%M-%S')+'.txt')
    # logger = Logger('./results/log-'+str(args.beta)+'-'+str(args.threshold)+'-'+time.strftime('%b%d_%H-%M-%S')+'.txt')
    logger = Logger('./results/'+str(len(os.listdir('./results')) + 1)+'.txt') 
    logger.info(args)

    # TensorBoard writer
    writer = SummaryWriter()

    # initialization
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

    # stopping criteria
    patience = 5
    counter = 0

    # model
    logger.info('Load model...')
    model = create_model(args.num_classes, args.drop_rate).to(device)   

    # dataloaders
    train_loader, val_loader = get_dataloader(args.data_path, args.batch_size, args.num_workers)

    # loss & optimizer
    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smoothing)
    criterion_kld = nn.KLDivLoss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    logger.info('Start training.')

    for epoch in range(1, args.epochs+1):
        logger.info('----------------------------------------------------------')
        logger.info('Epoch: %d, Learning Rate: %f', epoch, optimizer.param_groups[0]['lr'])

        for j in range(5):
            logger.info(f'Personality {j + 1}')
            logger.info(f'Maximums of LD: {[round(LD_maxes[j][0].cpu().tolist()[i], 4) for i in range(args.num_classes)]}')
            logger.info(f'Average weights: {[round(weights_avg[j][i], 4) for i in range(args.num_classes)]}')
            logger.info(f'Max weights: {[round(weights_max[j][i], 4) for i in range(args.num_classes)]}')
            logger.info(f'Min weights: {[round(weights_min[j][i], 4) for i in range(args.num_classes)]}')

        # train
        _, train_loss_ce, train_loss_kld, alpha_1, alpha_2 = train(train_loader, model, criterion, criterion_kld, optimizer, LD, epoch)
        train_loss, train_acc, outputs_new, targets_new, weights_new = validate(train_loader, model, criterion, epoch, phase='train')

        LD = generate_adaptive_LD(outputs_new, targets_new, args.num_classes, args.threshold, args.sharpen, args.T)
        LD_maxes = [torch.max(LD[i], dim=1) for i in range(5)]

        weights_avg, weights_max, weights_min = generate_average_weights(weights_new, targets_new, args.num_classes, args.max_weight, args.min_weight)

        # test
        val_loss, val_acc, _, _, _ = validate(val_loader, model, criterion, epoch, phase='val')

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch   
            counter = 0
        else:
            counter += 1

        logger.info('')
        logger.info('Alpha_1, Alpha_2: %.2f, %.2f Beta: %.2f', alpha_1, alpha_2, args.beta)
        logger.info('Train Acc: %.2f', train_acc)
        logger.info('Validate Acc: %.2f', val_acc)
        logger.info('Train Loss: %.2f', train_loss)
        logger.info('Validate Loss: %.2f', val_loss)
        logger.info('Best Acc: %.2f (%d)', best_acc, best_epoch)

        is_best = (best_epoch==epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'class_distributions': [LD[i].detach() for i in range(5)],
            }, 
            is_best)
        
        if counter >= patience:
            logger.info('Early stopping...')
            break 

        scheduler.step()

def train(train_loader, model, criterion, criterion_kld, optimizer, LD, epoch):
    if args.alpha is not None:
        alpha_1 = args.alpha
        alpha_2 = 1 - args.alpha
    else:
        if epoch <= args.beta:
            alpha_1 = math.exp(-(1-epoch/args.beta)**2)
            alpha_2 = 1
        else:
            alpha_1 = 1
            alpha_2 = math.exp(-(1-args.beta/epoch)**2)
    
    # losses
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_kld = AverageMeter()
    losses_rr = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    pbar.set_description(f'Epoch [{epoch}/{args.epochs}]')

    # training
    model.train()

    # iterate through each batch
    for i, (images, labels, indexes) in pbar:
        images = images.to(device)
        labels = labels.permute(1, 0)
        labels = [label.to(device) for label in labels]

        outputs_1, outputs_2, attention_weights = model(images)
        
        total_loss = 0
        total_loss_ce = 0
        total_loss_kld = 0
        total_RR_loss = 0

        # iterate for each personality trait (5 in total)
        for i in range(5):
            tops = int(args.batch_size * args.tops)

            _, top_idx = torch.topk(attention_weights[i].squeeze(), tops)
            _, down_idx = torch.topk(attention_weights[i].squeeze(), args.batch_size - tops, largest = False)

            high_group = attention_weights[i][top_idx]
            low_group = attention_weights[i][down_idx]
            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
            diff = low_mean - high_mean + args.margin_1

            if diff > 0:
                RR_loss = diff
            else:
                RR_loss = torch.tensor(0.0, device=device)

            loss_ce = criterion(outputs_1[i], labels[i]).mean()

            # label fusion
            attention_weights[i] = attention_weights[i].squeeze(1)
            attention_weights[i] = ((attention_weights[i] - attention_weights[i].min()) / (attention_weights[i].max() - attention_weights[i].min())) * (args.max_weight-args.min_weight) + args.min_weight
            attention_weights[i] = attention_weights[i].unsqueeze(1)

            targets = (1 - attention_weights[i]) * F.softmax(outputs_1[i], dim=1) + attention_weights[i] * LD[i][labels[i]]
            loss_kld = criterion_kld(F.log_softmax(outputs_2[i], dim=1), targets).sum() / args.batch_size

            loss = alpha_2 * loss_ce + alpha_1 * loss_kld + RR_loss
            
            total_loss += loss
            total_loss_ce += loss_ce
            total_loss_kld += loss_kld
            total_RR_loss += RR_loss
        
        total_loss /= 5.0
        total_loss_ce /= 5.0
        total_loss_kld /= 5.0
        total_RR_loss /= 5

        losses.update(total_loss.item(), images.size(0))
        losses_ce.update(total_loss_ce.item(), images.size(0))
        losses_kld.update(total_loss_kld.item(), images.size(0))
        losses_rr.update(total_RR_loss.item(), images.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=losses.avg, loss_ce=losses_ce.avg, loss_kld=losses_kld.avg, loss_rr = losses_rr.avg)

    return losses.avg, losses_ce.avg, losses_kld.avg, alpha_1, alpha_2

def validate(val_loader, model, criterion, epoch, phase='train'):
    losses = AverageMeter()
    accs = AverageMeter()

    model.eval()
    
    # 4 classes, 5 personalities
    outputs_new = [torch.ones(1, 4).to(device) for _ in range(5)]
    targets_new = [torch.ones(1).long().to(device) for _ in range(5)]
    weights_new = [torch.ones(1, 1).float().to(device) for _ in range(5)]

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    pbar.set_description(f'Epoch [{epoch}/{args.epochs}]')

    with torch.no_grad():
        for i, (inputs, targets, indexes) in pbar:
            inputs, targets = inputs.to(device), [target.to(device) for target in targets.permute(1, 0)]

            if phase == 'train':
                outputs, _, attention_weights = model(inputs)
            else:
                _, outputs, attention_weights = model(inputs)

            total_loss = 0
            total_acc = 0

            # loop through each personality
            for i in range(5):
                loss = criterion(outputs[i], targets[i]).mean()

                outputs_new[i] = torch.cat((outputs_new[i], outputs[i]), dim=0)
                targets_new[i] = torch.cat((targets_new[i], targets[i]), dim=0)
                weights_new[i] = torch.cat((weights_new[i], attention_weights[i]), dim=0)

                top1, _ = get_accuracy(outputs[i], targets[i], topk=(1, 4))

                total_loss += loss
                total_acc += top1
            
            total_loss /= 5.0
            total_acc /= 5.0
            
            losses.update(total_loss.item(), inputs.size(0))
            accs.update(total_acc.item(), inputs.size(0))

            pbar.set_postfix(loss=losses.avg, acc=accs.avg)

    return (losses.avg, accs.avg, outputs_new, targets_new, weights_new) 

if __name__ == '__main__':
    main()

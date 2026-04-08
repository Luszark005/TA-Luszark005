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

parser = argparse.ArgumentParser(description='PyTorch Training Swin-Emotion Ada-DF')
# train configs
parser.add_argument('--epochs', default=75, type=int)
parser.add_argument('--batch_size', default=4, type=int) 
parser.add_argument('--accumulation_steps', default=4, type=int) # Efektif batch size = 16
parser.add_argument('--lr', default=0.00005, type=float) # LR lebih kecil untuk Swin
parser.add_argument('--num_classes', default=4, type=int)
parser.add_argument('--num_frames', default=8, type=int)

# Ada-DF method configs
parser.add_argument('--threshold', default=0.7, type=float)
parser.add_argument('--sharpen', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--T', default=1.2, type=float)
parser.add_argument('--beta', default=3, type=int)
parser.add_argument('--max_weight', default=1.0, type=float)
parser.add_argument('--min_weight', default=0.2, type=float)
parser.add_argument('--drop_rate', default=0.2, type=float)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--margin_1', default=0.07, type=float)
parser.add_argument('--tops', default=0.7, type=float)

# common configs
parser.add_argument('--data_path', default='./dataset', type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--device_id', default=0, type=int)

args = parser.parse_args()

device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

def main():
    best_acc = 0
    best_epoch = 0
    set_random_seed(42)

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    logger = Logger(f'./results_emotions/log-{timestamp}.txt')
    logger.info(args)
    writer = SummaryWriter(f'./runs/Swin_Emotion_{timestamp}')

    # 1. Inisialisasi Label Distribution (LD) untuk 5 Trait
    LD = [torch.zeros(args.num_classes, args.num_classes).to(device) for _ in range(5)]
    for i in range(5):
        for j in range(args.num_classes):
            LD[i][j] = torch.zeros(args.num_classes).fill_((1-args.threshold)/(args.num_classes-1)).scatter_(0, torch.tensor(j), args.threshold)
        if args.sharpen:
            LD[i] = torch.pow(LD[i], 1/args.T) / torch.sum(torch.pow(LD[i], 1/args.T), dim=1)

    # 2. Load Model & Data
    model = create_model(num_classes=args.num_classes, drop_rate=args.drop_rate).to(device)
    train_loader, val_loader = get_dataloader(args.data_path, args.batch_size, args.num_workers, num_frames=args.num_frames)

    # 3. Loss & Optimizer (AdamW untuk Swin Transformer)
    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    criterion_kld = nn.KLDivLoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    logger.info('Mulai Pelatihan Swin Transformer + Ada-DF...')

    for epoch in range(1, args.epochs + 1):
        # TRAIN
        tr_loss, tr_ce, tr_kld, a1, a2 = train(train_loader, model, criterion_ce, criterion_kld, optimizer, LD, epoch, args.accumulation_steps)
        
        # VALIDATE (Mendapatkan output untuk update LD)
        val_loss, val_acc, outputs_val, targets_val, weights_val = validate(val_loader, model, criterion_ce, epoch)

        # 4. Update Adaptive Distribution (Ada-DF)
        LD = generate_adaptive_LD(outputs_val, targets_val, args.num_classes, args.threshold, args.sharpen, args.T)

        # Logging
        logger.info(f'Epoch {epoch} | Val Acc: {val_acc:.2f}% | Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)

        # Save Checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint='./checkpoints_swin')

        if epoch - best_epoch > 10: # Early Stopping sederhana
            logger.info("Early stopping...")
            break
        
        scheduler.step()

def train(train_loader, model, criterion_ce, criterion_kld, optimizer, LD, epoch, accumulation_steps):
    model.train()
    meters = {k: AverageMeter() for k in ['loss', 'ce', 'kld', 'rr']}
    
    # Dinamis Alpha untuk Ada-DF
    if epoch <= args.beta:
        alpha_1, alpha_2 = math.exp(-(1 - epoch / args.beta)**2), 1
    else:
        alpha_1, alpha_2 = 1, math.exp(-(1 - args.beta / epoch)**2)

    optimizer.zero_grad()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")

    for i, (images, labels, emotions, _) in pbar:
        images, emotions = images.to(device), emotions.to(device)
        labels = labels.to(device).permute(1, 0) # [5, B]

        out_aux, out_target, attention_weights = model(images, emotions)

        # DEBUG DI SINI
        if epoch == 1 and i == 0:
            print("Jumlah output (aux):", len(out_aux))
            print("Jumlah output (target):", len(out_target))

    for j in range(len(out_target)):
        print(f"Trait {j} shape:", out_target[j].shape)
        
        loss_batch = 0
        for j in range(5): # Loop OCEAN traits
            # Rank Regularization Loss
            tops = int(args.batch_size * args.tops)
            _, top_idx = torch.topk(attention_weights[j].squeeze(), tops)
            _, down_idx = torch.topk(attention_weights[j].squeeze(), args.batch_size - tops, largest=False)
            rr_loss = torch.clamp(torch.mean(attention_weights[j][down_idx]) - torch.mean(attention_weights[j][top_idx]) + args.margin_1, min=0)

            # Cross Entropy untuk Auxiliary Branch
            l_ce = criterion_ce(out_aux[j], labels[j]).mean()

            # KL Divergence untuk Target Branch
            # Adaptive Label Fusion: d_fused = w * d_class + (1-w) * d_aux
            soft_aux = F.softmax(out_aux[j], dim=1)
            targets_fused = (1 - attention_weights[j]) * soft_aux + attention_weights[j] * LD[j][labels[j]]
            l_kld = criterion_kld(F.log_softmax(out_target[j], dim=1), targets_fused).sum() / args.batch_size

            loss_batch += (alpha_2 * l_ce + alpha_1 * l_kld + rr_loss)

        loss_batch /= 5.0
        (loss_batch / accumulation_steps).backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        meters['loss'].update(loss_batch.item(), images.size(0))
        pbar.set_postfix(loss=meters['loss'].avg)

    return meters['loss'].avg, 0, 0, alpha_1, alpha_2

def validate(loader, model, criterion, epoch):
    model.eval()
    meters = {'loss': AverageMeter(), 'acc': AverageMeter()}
    
    # Store outputs for Ada-DF update
    outputs_all = [torch.tensor([]).to(device) for _ in range(5)]
    targets_all = [torch.tensor([], dtype=torch.long).to(device) for _ in range(5)]
    weights_all = [torch.tensor([]).to(device) for _ in range(5)]

    with torch.no_grad():
        for images, labels, emotions, _ in tqdm(loader, desc=f"Epoch {epoch} [Val]"):
            images, emotions = images.to(device), emotions.to(device)
            labels_list = labels.to(device).permute(1, 0)

            _, out_target, weights = model(images, emotions)

            batch_loss = 0
            batch_acc = 0
            for j in range(5):
                batch_loss += criterion(out_target[j], labels_list[j]).mean()
                acc, _ = get_accuracy(out_target[j], labels_list[j], topk=(1, 4))
                batch_acc += acc
                
                outputs_all[j] = torch.cat([outputs_all[j], out_target[j]])
                targets_all[j] = torch.cat([targets_all[j], labels_list[j]])
                weights_all[j] = torch.cat([weights_all[j], weights[j]])

            meters['loss'].update((batch_loss/5).item(), images.size(0))
            meters['acc'].update((batch_acc/5).item(), images.size(0))

    return meters['loss'].avg, meters['acc'].avg, outputs_all, targets_all, weights_all

if __name__ == '__main__':
    main()

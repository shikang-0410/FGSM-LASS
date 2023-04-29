import argparse

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from models import resnet18, WideResNet, PreActResNet18
import time
import logging
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import std, get_loaders, evaluate_pgd, evaluate_standard, label_smoothing, LabelSmoothLoss, clamp, \
    lower_limit, upper_limit, get_loaders_cifar100, get_loaders_tiny_imagenet

logger = logging.getLogger('logger')
CUDA_LAUNCH_BLOCKING = '0,1'


def get_args():
    parser = argparse.ArgumentParser('FGSM-CKPT')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--target_model', default='PreActResNet18',
                        choices=['ResNet18', 'PreActResNet18', 'WideResNet'], type=str, help='model name')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--factor', default=0.9, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=int)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--data_dir', default='/data/yangshikang/cifar10', type=str)
    parser.add_argument('--out_dir', default='FGSM_CKPT_output', type=str, help='Output directory')
    parser.add_argument('--start_epoch', default=1, type=float)
    parser.add_argument('--c', default=3, type=float)
    parser.add_argument('--dataset', default='tiny-imagenet', type=str,
                        choices=['cifar10', 'cifar100', 'tiny-imagenet'])

    arguments = parser.parse_args()
    return arguments


args = get_args()
output_path = os.path.join(args.out_dir, str(args.dataset))
output_path = os.path.join(output_path, 'epochs_' + str(args.epochs))
output_path = os.path.join(output_path, 'factor_' + str(args.factor))
output_path = os.path.join(output_path, 'target_model_' + args.target_model)
output_path = os.path.join(output_path, 'lr_' + str(args.lr))
output_path = os.path.join(output_path, 'epsilon_' + str(args.epsilon))
output_path = os.path.join(output_path, 'alpha_' + str(args.alpha))
output_path = os.path.join(output_path, 'c_' + str(args.c))

print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

summary_log_dir = os.path.join(output_path, "runs")
summary_writer = SummaryWriter(log_dir=summary_log_dir)

logfile = os.path.join(output_path, 'output.log')
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(output_path, 'output.log'))
logger.info(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std

if args.target_model == "ResNet18":
    target_model = resnet18(10 if args.dataset == 'cifar10' else (100 if args.dataset == 'cifar100' else 200))
elif args.target_model == "PreActResNet18":
    target_model = PreActResNet18(10 if args.dataset == 'cifar10' else (100 if args.dataset == 'cifar100' else 200))
else:
    target_model = WideResNet(
        num_classes=10 if args.dataset == 'cifar10' else (100 if args.dataset == 'cifar100' else 200))

target_model = torch.nn.DataParallel(target_model)
target_model.cuda()

target_model_optimizer = torch.optim.SGD(target_model.parameters(), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)


def lr_schedule(t):
    if t < 100:
        return args.lr
    elif t < 105:
        return args.lr / 10.
    else:
        return args.lr / 100.


best_result = 0
checkpoint_path = os.path.join(output_path, 'checkpoint.pth')
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    target_model.load_state_dict(checkpoint['target_model'])
    best_result = checkpoint['best_result']
    args.start_epoch = checkpoint['start_epoch']
    print(f'resuming... epoch: {args.start_epoch}')
    print(f'best pgd10 acc: {best_result}')

if args.dataset == 'cifar10':
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
elif args.dataset == 'cifar100':
    train_loader, test_loader = get_loaders_cifar100(args.data_dir, args.batch_size)
else:
    train_loader, test_loader = get_loaders_tiny_imagenet(args.data_dir, args.batch_size)


def train(epoch):
    target_model.train()
    epoch_time = 0
    train_loss = 0
    train_acc = 0
    train_n = 0

    lr = lr_schedule(epoch)
    target_model_optimizer.param_groups[0].update(lr=lr)
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time = time.time()
        data, target = data.cuda(), target.cuda()

        batch_size = len(data)
        delta = torch.zeros_like(data).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - data, upper_limit - data)
        delta.requires_grad = True
        logit_clean = target_model(data + delta)
        grad = torch.autograd.grad(F.cross_entropy(logit_clean, target), [delta])[0]
        delta.requires_grad = False
        delta.data = delta + alpha * torch.sign(grad)
        data_adv = data + delta

        _, pre_clean = torch.max(logit_clean.data, dim=1)
        correct = pre_clean.eq(target)
        correct_idx = torch.masked_select(torch.arange(batch_size).cuda(), correct)
        wrong_idx = torch.masked_select(torch.arange(batch_size).cuda(), ~correct)

        data_adv[wrong_idx] = data[wrong_idx].detach()

        datas = (torch.cat([data] * (args.c - 1)) +
                 torch.cat([torch.arange(1, args.c).cuda().view(-1, 1)] * batch_size, dim=1).
                 view(-1, 1, 1, 1) * torch.cat([delta / args.c] * (args.c - 1)))

        targets = torch.cat([target] * (args.c - 1))

        # Inference checkpoints for correct images.
        idx = correct_idx
        idxs = []
        target_model.eval()
        with torch.no_grad():
            for k in range(args.c - 1):
                # Stop iterations if all checkpoints are correctly classified.
                if len(idx) == 0:
                    break
                # Stack checkpoints for inference.
                elif 1024 >= (len(idxs) + 1) * len(idx):
                    idxs.append(idx + k * batch_size)
                else:
                    pass

                # Do inference.
                if (1024 < (len(idxs) + 1) * len(idx)) or (k == args.c - 2):
                    # Inference selected checkpoints.
                    idxs = torch.cat(idxs).cuda()
                    pre = target_model(datas[idxs]).detach()
                    _, pre = torch.max(pre.data, 1)
                    correct = (pre.eq(targets[idxs])).view(-1, len(idx))

                    # Get index of misclassified images for selected checkpoints.
                    max_idx = idxs.max() + 1
                    wrong_idxs = (idxs.view(-1, len(idx)) * (1 - correct * 1)) + (max_idx * (correct * 1))
                    wrong_idx, _ = wrong_idxs.min(dim=0)

                    wrong_idx = torch.masked_select(wrong_idx, wrong_idx < max_idx)
                    update_idx = wrong_idx % batch_size
                    data_adv[update_idx] = datas[wrong_idx]

                    # Set new indexes by eliminating updated indexes.
                    idx = torch.tensor(list(set(idx.cpu().data.numpy().tolist()) \
                                            - set(update_idx.cpu().data.numpy().tolist())))
                    idxs = []

        target_model.train()
        output = target_model(data_adv.detach().cuda())
        if args.factor < 1:
            smooth_label = Variable(
                torch.tensor(label_smoothing(target, args.factor, 10 if args.dataset == 'cifar10' else (
                    100 if args.dataset == 'cifar100' else 200))).cuda())
            loss = LabelSmoothLoss(output, smooth_label.float())
        else:
            loss = F.cross_entropy(output, target)
        target_model_optimizer.zero_grad()
        loss.backward()
        target_model_optimizer.step()

        train_loss += loss.item() * target.size(0)
        train_acc += (output.max(1)[1].eq(target)).sum().item()
        train_n += target.size(0)

        batch_end_time = time.time()
        epoch_time += batch_end_time - batch_start_time

    logger.info('Epoch \t Seconds \t LR \t Train Loss \t Train Acc')
    logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
                epoch, epoch_time, lr, train_loss / train_n, train_acc / train_n)

    print('Epoch \t Seconds \t LR \t Train Loss \t Train Acc')
    print(f'{epoch} \t {epoch_time:.1f} \t {lr:.4f} \t '
          f'{train_loss / train_n:.4f} \t {train_acc / train_n:.4f}')

    summary_writer.add_scalar('1_train_acc', train_acc / train_n, epoch)

    return epoch_time


def main():
    global best_result
    total_time = 0.
    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_time = train(epoch)
        total_time += epoch_time

        # test
        target_model.eval()
        pgd_loss, pgd_acc = evaluate_pgd(test_loader, target_model, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, target_model)
        summary_writer.add_scalar('0_pgd_acc', pgd_acc, epoch)
        summary_writer.add_scalar('2_test_acc', test_acc, epoch)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        print('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        print(f'{test_loss:.4f} \t {test_acc:.4f} \t {pgd_loss:.4f} \t {pgd_acc:.4f}')

        if best_result <= pgd_acc:
            best_result = pgd_acc
            torch.save(target_model.state_dict(), os.path.join(output_path, 'best_model.pth'))
        torch.save(target_model.state_dict(), os.path.join(output_path, 'final_model.pth'))

        torch.save({
            'target_model': target_model.state_dict(),
            'best_result': best_result,
            'start_epoch': epoch + 1
        }, os.path.join(output_path, 'checkpoint.pth'))

        print(f'best pgd-10-1 acc: {best_result}')

    print(f'total_time: {total_time:.1f} (s)')
main()

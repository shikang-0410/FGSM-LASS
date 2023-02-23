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

from utils import std, get_loaders, evaluate_pgd, evaluate_standard, label_smoothing, LabelSmoothLoss, attack_fgsm, \
    get_loaders_cifar100, get_loaders_tiny_imagenet

logger = logging.getLogger('logger')


def get_args():
    parser = argparse.ArgumentParser('FGSM-RS')

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
    parser.add_argument('--data_dir', default='/data/yangshikang/tiny-imagenet-200', type=str)
    parser.add_argument('--out_dir', default='FGSM_RS_output', type=str, help='Output directory')
    parser.add_argument('--start_epoch', default=1, type=float)
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


checkpoint_path = os.path.join(output_path, 'checkpoint.pth')

best_result = 0
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
        delta = attack_fgsm(target_model, data, target, epsilon, alpha, 1, early_stop=False, is_init=True)
        output = target_model(data + delta)
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

    print(total_time)


main()

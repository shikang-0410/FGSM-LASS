import argparse

from alpha_predictor import AlphaPredictor
from models import *
import time
import logging
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import std, get_loaders, evaluate_pgd, evaluate_standard, clamp, lower_limit, upper_limit, calculate_loss, \
    get_loaders_cifar100, get_loaders_tiny_imagenet

logger = logging.getLogger('logger')
CUDA_LAUNCH_BLOCKING = '0,1'


def get_args():
    parser = argparse.ArgumentParser('FGSM-LASS')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--target_model', default='PreActResNet18',
                        choices=['ResNet18', 'PreActResNet18', 'WideResNet'], type=str, help='model name')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument('--lr_ap', type=float, default=6e-5, help='AlphaPredictor learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--factor', default=0.9, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--data_dir', default='/data/yangshikang/tiny-imagenet-200', type=str)
    parser.add_argument('--out_dir', default='FGSM_LASS_output', type=str, help='Output directory')
    parser.add_argument('--iter_num', default=20, type=int, help='number of iterations for AlphaPredictor update')
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--dataset', default='tiny-imagenet', type=str,
                        choices=['cifar10', 'cifar100', 'tiny-imagenet'])

    arguments = parser.parse_args()
    return arguments


args = get_args()
output_path = os.path.join(args.out_dir, str(args.dataset))
output_path = os.path.join(output_path, 'epochs_' + str(args.epochs))
output_path = os.path.join(output_path, 'target_model_' + args.target_model)
output_path = os.path.join(output_path, 'lr_' + str(args.lr))
output_path = os.path.join(output_path, 'lr_ap_' + str(args.lr_ap))
output_path = os.path.join(output_path, 'iter_num_' + str(args.iter_num))
output_path = os.path.join(output_path, 'factor_' + str(args.factor))
output_path = os.path.join(output_path, 'epsilon_' + str(args.epsilon))

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

alpha_predictor = AlphaPredictor(3, 64)
alpha_predictor = torch.nn.DataParallel(alpha_predictor)
alpha_predictor.cuda()

optimizer_ap = torch.optim.SGD(alpha_predictor.parameters(), lr=args.lr_ap, momentum=args.momentum,
                               weight_decay=args.weight_decay)


def lr_schedule(t):
    if t < 100:
        return args.lr, args.lr_ap
    elif t < 105:
        return args.lr / 10., args.lr_ap / 10.
    else:
        return args.lr / 100., args.lr_ap / 100.


checkpoint_path = os.path.join(output_path, 'checkpoint.pth')

best_result = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    target_model.load_state_dict(checkpoint['target_model'])
    alpha_predictor.load_state_dict(checkpoint['alpha_predictor'])
    best_result = checkpoint['best_result']
    args.start_epoch = checkpoint['start_epoch']
    print(f'resuming... epoch: {args.start_epoch}')
    print(f'best pgd acc: {best_result}')

if args.dataset == 'cifar10':
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
elif args.dataset == 'cifar100':
    train_loader, test_loader = get_loaders_cifar100(args.data_dir, args.batch_size)
else:
    train_loader, test_loader = get_loaders_tiny_imagenet(args.data_dir, args.batch_size)

epsilon = (args.epsilon / 255.) / std


def train(epoch):
    target_model.train()
    epoch_time = 0
    train_loss = 0
    train_acc = 0
    train_n = 0

    lr, lr_ap = lr_schedule(epoch)
    target_model_optimizer.param_groups[0].update(lr=lr)
    # optimizer_ap.param_groups[0].update(lr=lr_ap)
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time = time.time()
        data, target = data.cuda(), target.cuda()

        delta = torch.zeros_like(data).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - data, upper_limit - data)
        delta.requires_grad = True
        output_adv = target_model(data + delta)
        loss_nature = F.cross_entropy(output_adv, target)
        grad = torch.autograd.grad(loss_nature, [delta])[0]
        delta.requires_grad = False

        if batch_idx % args.iter_num == 0:
            # 训练alpha预测模型
            _, loss = calculate_loss(target_model, grad, args.factor, alpha_predictor, data, target, delta.detach(),
                                     epsilon, 1.25 * epsilon, 1, summary_writer, logger=logger,
                                     class_num=10 if args.dataset == 'cifar10' else (
                                         100 if args.dataset == 'cifar100' else 200))
            loss = -loss
            target_model_optimizer.zero_grad()
            optimizer_ap.zero_grad()
            loss.backward()
            optimizer_ap.step()

        # 训练目标模型
        output_adv, loss = calculate_loss(target_model, grad, args.factor, alpha_predictor, data, target,
                                          delta.detach(), epsilon, 1.25 * epsilon, 0,
                                          class_num=10 if args.dataset == 'cifar10' else (
                                              100 if args.dataset == 'cifar100' else 200))

        target_model_optimizer.zero_grad()
        optimizer_ap.zero_grad()
        loss.backward()
        target_model_optimizer.step()
        train_loss += loss.item() * target.size(0)
        train_acc += (output_adv.max(1)[1].eq(target)).sum().item()
        train_n += target.size(0)

        batch_end_time = time.time()
        epoch_time += batch_end_time - batch_start_time

    logger.info('Epoch \t Seconds \t LR \t Train Loss \t Train Acc')
    logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
                epoch, epoch_time, lr, train_loss / train_n, train_acc / train_n)
    print('Epoch \t Seconds \t LR \t Train Loss \t Train Acc')
    print(f'{epoch} \t {epoch_time:.1f} \t {lr:.4f} \t {train_loss / train_n:.4f} \t {train_acc / train_n:.4f}')

    summary_writer.add_scalar('2_train_acc', train_acc / train_n, epoch)

    return epoch_time


def main():
    global best_result
    total_time = 0.
    for epoch in range(args.start_epoch, args.epochs + 1):
        total_time += train(epoch)

        target_model.eval()
        pgd_loss, pgd_acc = evaluate_pgd(test_loader, target_model, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, target_model)
        summary_writer.add_scalar('1_pgd_acc', pgd_acc, epoch)
        summary_writer.add_scalar('3_test_acc', test_acc, epoch)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        print('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        print(f'{test_loss:.4f} \t {test_acc:.4f} \t {pgd_loss:.4f} \t {pgd_acc:.4f}')

        if best_result <= pgd_acc:
            best_result = pgd_acc
            torch.save(target_model.state_dict(), os.path.join(output_path, 'best_model.pth'))
        torch.save(target_model.state_dict(), os.path.join(output_path, 'final_model.pth'))

        torch.save({
            'target_model': target_model.state_dict(),
            'alpha_predictor': alpha_predictor.state_dict(),
            'best_result': best_result,
            'start_epoch': epoch + 1
        }, os.path.join(output_path, 'checkpoint.pth'))

        print(f'best pgd-10-1 acc: {best_result}')

    print(f'total time {total_time} (s)')


if __name__ == '__main__':
    main()

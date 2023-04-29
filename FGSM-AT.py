import argparse

from torch.autograd import Variable

from models import *
import time
import logging
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import std, get_loaders, evaluate_pgd, evaluate_standard, label_smoothing, LabelSmoothLoss, attack_fgsm

logger = logging.getLogger('logger')
CUDA_LAUNCH_BLOCKING = '0,1'


def get_args():
    parser = argparse.ArgumentParser('FGSM-AT')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--target_model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--factor', default=0.9, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=int)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--data_dir', default='/data/yangshikang/cifar10', type=str)
    parser.add_argument('--out_dir', default='FGSM_AT_output', type=str, help='Output directory')
    parser.add_argument('--start_epoch', default=1, type=float)

    arguments = parser.parse_args()
    return arguments


args = get_args()
output_path = os.path.join(args.out_dir, 'epochs_' + str(args.epochs))
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
    target_model = resnet18()
else:
    target_model = WideResNet()
device_id = range(torch.cuda.device_count())
if len(device_id) > 1:
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

train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)


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
        delta = attack_fgsm(target_model, data, target, epsilon, alpha, 1, early_stop=False, is_init=False)

        output = target_model(data + delta)
        if args.factor < 1:
            smooth_label = Variable(torch.tensor(label_smoothing(target, args.factor)).cuda())
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

    summary_writer.add_scalar('2_train_acc', train_acc / train_n, epoch)


def main():
    global best_result
    for epoch in range(args.start_epoch, args.epochs + 1):
        train(epoch)

        target_model.eval()
        pgd_loss, pgd_acc = evaluate_pgd(test_loader, target_model, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, target_model)
        summary_writer.add_scalar('0_pgd_acc', pgd_acc, epoch)
        summary_writer.add_scalar('1_test_acc', test_acc, epoch)
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


main()

import os.path

import torch
import torch.nn.functional as F
import numpy as np
from advertorch.attacks import CarliniWagnerL2Attack
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import transforms

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)
cifar10_mean = (0., 0., 0.)
cifar10_std = (1., 1., 1.)
mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def clamp(X, lower_lim, upper_lim):
    return torch.max(torch.min(X, upper_lim), lower_lim)


################################## loss function of target model ##################################
def label_smoothing(label, factor, class_num=10):
    one_hot = np.eye(class_num)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))

    return result


def LabelSmoothLoss(y, target):
    log_prob = F.log_softmax(y, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss


################################## load datasets ##################################
def get_loaders(dir_, batch_size, shuffle=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    num_workers = 0
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def get_loaders_cifar100(dir_, batch_size, shuffle=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 0
    train_dataset = datasets.CIFAR100(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def get_loaders_tiny_imagenet(dir_, batch_size, shuffle=True):
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 0
    train_dataset = datasets.ImageFolder(root=os.path.join(dir_, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(dir_, 'val'), transform=test_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


################################## attack ##################################
def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, early_stop=False, is_init=True):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if is_init:
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1].eq(y))[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            grad = torch.autograd.grad(loss, [delta])[0]
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index, :, :, :], upper_limit - X[index, :, :, :])
            delta.data[index, :, :, :] = d
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def attack_fgsm(model, X, y, epsilon, alpha, restarts, early_stop=False, is_init=True):
    attack_iters = 1
    return attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, early_stop, is_init)


################################ GradAlign loss #######################################
def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta


def get_input_grad(model, X, y, eps, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError('wrong delta init')

    output = model(X + delta)
    loss = F.cross_entropy(output, y)

    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]

    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad


def grad_align_loss(model, X, y, eps, lamd):
    grad1 = get_input_grad(model, X, y, eps, delta_init='none', backprop=False)
    grad2 = get_input_grad(model, X, y, eps, delta_init='random_uniform', backprop=True)
    grad1, grad2 = grad1.reshape(len(grad1), -1), grad2.reshape(len(grad2), -1)
    cos = torch.nn.functional.cosine_similarity(grad1, grad2, 1)
    reg = lamd * (1.0 - cos.mean())

    return reg


################################ AutoAttack #######################################
def AutoAttack(model, norm='Linf', epsilon=8 / 255., data_dir='/data/yangshikang/cifar10', dataset='cifar10'):
    model.eval()
    if dataset == 'cifar10':
        _, test_loader = get_loaders(data_dir, batch_size=1000)
    elif dataset == 'cifar100':
        _, test_loader = get_loaders_cifar100(data_dir, batch_size=1000)
    else:
        _, test_loader = get_loaders_tiny_imagenet(data_dir, 128)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    # load attack
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=norm, eps=epsilon, version='standard')

    # run attack and save images
    with torch.no_grad():
        adversary.run_standard_evaluation(x_test, y_test, bs=1000)


def evaluate_cw(test_loader, model, num_classes):
    attacker = CarliniWagnerL2Attack(model, num_classes, max_iterations=10)
    loss = 0
    acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        X_adv = attacker.perturb(X, y)
        with torch.no_grad():
            output = model(X_adv)
            loss = F.cross_entropy(output, y)
            loss += loss.item() * y.size(0)
            acc += (output.max(1)[1].eq(y)).sum().item()
            n += y.size(0)
    return loss / n, acc / n


def evaluate_pgd(test_loader, model, attack_iters, restarts, epsilon=(8 / 255.) / std, alpha=(2 / 255.) / std):
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, early_stop=True)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1].eq(y)).sum().item()
            n += y.size(0)
    return pgd_loss / n, pgd_acc / n


def evaluate_fgsm(test_loader, model, restarts):
    attack_iters = 1
    alpha = (10 / 255.) / std
    return evaluate_pgd(test_loader, model, attack_iters, restarts, alpha=alpha)


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1].eq(y)).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n


step = 1


################################## loss calculate ##################################
def calculate_loss(model,
                   grad,
                   factor,
                   alpha_predictor,
                   X,
                   y,
                   init_delta,
                   epsilon=(8 / 255.) / std,
                   alpha=(10 / 255.) / std,
                   for_ap=0,
                   summary_writer=None,
                   adv_init_network=None,
                   logger=None,
                   class_num=10):
    global step
    if for_ap == 1:
        model.eval()
        if adv_init_network is not None:
            adv_init_network.eval()
        alpha_predictor.train()
    elif for_ap == 2:
        model.eval()
        adv_init_network.train()
        alpha_predictor.eval()
    else:
        model.train()
        if adv_init_network is not None:
            adv_init_network.eval()
        alpha_predictor.eval()

    # 计算对抗初始化，并计算对抗初始化后的梯度
    if adv_init_network is not None:
        ain_input = torch.cat([X, 1.0 * (torch.sign(grad))], 1).detach()
        init_delta = adv_init_network(ain_input)
        x_adv = X + init_delta
        x_adv.requires_grad_()
        loss_adv = F.cross_entropy(model(x_adv), y)
        grad_adv = torch.autograd.grad(loss_adv, [x_adv])[0]
    else:
        grad_adv = grad

    # 计算对抗扰动
    ap_input = X.detach()
    ap_output = alpha_predictor(ap_input)
    alpha = alpha[None, :, :, :] * ap_output[:, :, None, None]
    if for_ap == 1:
        alpha_ = torch.mean(alpha, dim=1)
        alpha_ = alpha_[:, 0, 0] * 255. * torch.mean(std)
        min_alpha_ = torch.min(alpha_)
        mean_alpha_ = torch.mean(alpha_)
        max_alpha_ = torch.max(alpha_)
        logger.info(f'min={min_alpha_}, mean={mean_alpha_}, max={max_alpha_}')
        summary_writer.add_scalar('0_alpha_mean', mean_alpha_, step)
        summary_writer.add_scalar('1_alpha_gap', max_alpha_ - min_alpha_, step)
        step += 1
    delta = clamp(init_delta + alpha * torch.sign(grad_adv), -epsilon, epsilon)
    delta.data = clamp(delta, lower_limit - X, upper_limit - X)

    # 计算损失
    output_adv = model(X + delta)
    smooth_label = Variable(torch.tensor(label_smoothing(y, factor, class_num)).cuda())
    loss = LabelSmoothLoss(output_adv, smooth_label.float())

    return output_adv, loss


################################## plot ##################################
def plot(datas, labels, offset, end, file):
    datas = np.array(datas)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    # 绘图
    for i in range(offset, end + 1):
        heights = datas[:, i]
        if i == 3:
            ax.bar(labels, heights, width=0.3, label=i)
        else:
            ax.bar(labels, heights - datas[:, i - 1], width=0.3, bottom=datas[:, i - 1], label=i)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Percentage')
    plt.savefig(file, dpi=900, bbox_inches='tight')
    plt.show()

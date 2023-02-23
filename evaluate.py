import os.path

import torch

import models
from models import resnet18
from utils import evaluate_pgd, get_loaders, AutoAttack, evaluate_cw, evaluate_standard, get_loaders_cifar100, \
    get_loaders_tiny_imagenet

target_model = models.PreActResNet18(num_classes=200)
target_model = torch.nn.DataParallel(target_model)
target_model.cuda()

_, test_loader = get_loaders_tiny_imagenet('/data/yangshikang/tiny-imagenet-200', 128)

# path = f'FGSM_RS_output/cifar100/epochs_110/factor_0.9/target_model_ResNet18/lr_0.1/epsilon_8/alpha_10'
path = f'FGSM_GA_output/tiny-imagenet/epochs_110/factor_0.9/target_model_PreActResNet18/lr_0.01/epsilon_8/alpha_10'
# path = f'FGSM_CKPT_output/cifar100/epochs_110/factor_0.9/target_model_ResNet18/lr_0.1/epsilon_8/alpha_10/c_3'
# path = f'ATAS_output/cifar100/epochs_110/factor_0.9/target_model_ResNet18/lr_0.1/epsilon_8/beta_0.6/c_0.01'
# path = f'FGSM_LASS_output/tiny-imagenet/epochs_110/target_model_PreActResNet18/lr_0.01/lr_ap_6e-05/iter_num_20/factor_0.9/epsilon_8'
# path = f'PGD_AT_output/cifar100/epochs_110/factor_0.9/target_model_ResNet18/lr_0.1/epsilon_8/alpha_2/iters_10'
for name in ['best_model.pth', 'final_model.pth']:
    file = os.path.join(path, name)
    target_model.load_state_dict(torch.load(file))
    target_model.eval()

    _, clean_acc = evaluate_standard(test_loader, target_model)
    _, pgd_10_acc = evaluate_pgd(test_loader, target_model, 10, 1)
    _, pgd_20_acc = evaluate_pgd(test_loader, target_model, 20, 1)
    _, pgd_50_acc = evaluate_pgd(test_loader, target_model, 50, 1)
    print(clean_acc, pgd_10_acc, pgd_20_acc, pgd_50_acc)
    _, cw_acc = evaluate_cw(test_loader, target_model, num_classes=200)
    print(cw_acc)

    adversary = AutoAttack(target_model, data_dir='/data/yangshikang/tiny-imagenet-200', dataset='tiny-imagenet')

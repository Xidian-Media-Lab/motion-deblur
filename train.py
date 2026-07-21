import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data_loader.data_load import JAI_data
from loss.common import gradient, clamp
from net.MyNet import MyNet

import matplotlib.pyplot as plt

def init_seeds(seed=0):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Motion Deblur')
    parser.add_argument('--dataset_path',
                        default='train_data/JAI/Crop256_128',
                        help='path to training dataset')
    parser.add_argument('--model_save_path', default='model')
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('--loss_weight', default='[13, 1, 3]', type=str, help='loss weight')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool, help='use GPU or not.')
    parser.add_argument('--lr_start', default='0.0001', type=float)
    parser.add_argument('--lr_decay', default='0.85', type=float)
    parser.add_argument('--experiment_path', default='experiment_path', type=str)
    parser.add_argument('--num_workers', '--workers', default=4, type=int, help='number of data loading workers ')
    args = parser.parse_args()

    init_seeds(args.seed)
    train_dataset = JAI_data(args.dataset_path) # Load JAI Dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    device = torch.device('cuda:{}'.format(0))
    model = MyNet().to(device) # Load Fusion model
    optimizer = optim.Adam(model.parameters(), lr=args.lr_start)
    lr = StepLR(optimizer=optimizer, step_size=1, gamma=args.lr_decay) # After each iteration, the learning rate becomes 0.85 times of the original
    log_name = os.path.join(args.experiment_path, "log.txt") #  Save the output of the port to log.txt
    sys.stdout = utils.ExperimentLogger(filename=log_name, stream=sys.stdout)
    stat_dict = utils.get_stat_dict()
    n, m, t = eval(args.loss_weight) # Obtain the weight of three loss functions
    print(f'n={n} m={m} t={t} lr_start ={args.lr_start} decay={args.lr_decay}')
    best_loss = 10000.

    train_losses = []
    for epoch in range(args.epochs):
        model.train()
        train_tqdm = tqdm(train_loader, total=len(train_loader))
        epoch_loss = 0.
        epoch_l1 = 0.
        epoch_grad = 0.
        epoch_aux = 0.
        iteration = 0
        for i, (_, vis_y_image, _, _, inf_image, _, vistruth_y_image, _, _, mask_image, _) in enumerate(train_tqdm):
            iteration += 1
            vis_y_image = vis_y_image.cuda()
            inf_image = inf_image.cuda()
            vistruth_y_image = vistruth_y_image.cuda()
            mask_image = mask_image.cuda()
            mask_background = 1 - mask_image

            optimizer.zero_grad()

            fused_image = model(vis_image=vis_y_image, nir_image=inf_image)
            fused_image = clamp(fused_image)
            #Acquire front and rear scenes of NIR image and VIS image respectively
            inf_foreground = mask_image * inf_image
            vis_y_foreground = mask_image * vis_y_image
            vistruth_foreground = mask_image * vistruth_y_image
            fused_foreground = mask_image * fused_image

            inf_background = mask_background * inf_image
            vis_y_background = mask_background * vis_y_image
            vistruth_background = mask_background * vistruth_y_image
            fused_background = mask_background * fused_image

            # loss 1 -> L1 loss
            loss_l1_foreground = F.l1_loss(fused_foreground, inf_foreground)
            loss_l1_background = F.l1_loss(fused_background, vis_y_background)

            # loss 2 -> grad loss
            loss_sobelgrad_foreground = F.l1_loss(gradient(fused_foreground),
                                                  torch.max(gradient(vistruth_foreground), gradient(inf_foreground)))
            loss_sobelgrad_background = F.l1_loss(gradient(fused_background),
                                                  torch.max(gradient(vistruth_background), gradient(inf_background)))

            # loss 3 -> auxiliary intensity  loss
            loss_aux_foreground = F.l1_loss(fused_foreground, torch.max(vistruth_foreground, inf_foreground))
            loss_aux_background = F.l1_loss(fused_background, torch.max(vistruth_background, inf_background))

            loss_l1 = n * (loss_l1_foreground + loss_l1_background)
            loss_grad = m * (loss_sobelgrad_foreground + loss_sobelgrad_background)
            loss_aux = t * (loss_aux_foreground + loss_aux_background)

            loss = loss_l1 + loss_grad + loss_aux

            train_tqdm.set_postfix(epoch=epoch,
                                   loss_l1=loss_l1.item(),
                                   loss_grad=loss_grad.item(),
                                   loss_aux=loss_aux.item(),
                                   loss_total=loss.item())

            epoch_l1 += loss_l1.item()
            epoch_grad += loss_grad.item()
            epoch_aux += loss_aux.item()
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        num_epochs = args.epochs
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")

        if best_loss > epoch_loss / iteration:
            best_loss = epoch_loss / iteration
            best_epoch = epoch
        print(
            "lr: {:.5}".format(optimizer.param_groups[0]['lr']),
            "best_epoch: {} ".format(best_epoch),
            "epoch_l1: {:.8} ".format(epoch_l1 / iteration),
            "epoch_grad: {:.8} ".format(epoch_grad / iteration),
            "epoch_aux: {:.8} ".format(epoch_aux / iteration),
            "epoch_loss: {:.8} ".format(epoch_loss / iteration),
        )
        sys.stdout.flush()
        lr.step()
        torch.save(model.state_dict(), f'{args.model_save_path}/model_epoch_{epoch}.pth')

# Training Loss Curve
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()

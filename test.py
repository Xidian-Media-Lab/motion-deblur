import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from data_loader.data_load import JAI_testdata2
from loss.common import YCrCb2RGB
from net.MyNet import MyNet

# test_transforms = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(400)])
test_transforms = transforms.Compose([transforms.ToTensor()])


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
    parser.add_argument('--dataset_path', default='test_data/test')  # location of test data
    parser.add_argument('--save_path', default='results')  # location of fusion results
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--model_save_path', default='model/EFDN_model_epoch_29.pth')
    argument = parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--cuda', default=True, type=bool, help='use GPU or not.')
    parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

    args = parser.parse_args()

    init_seeds(args.seed)

    test_dataset = JAI_testdata2(data_dir=args.dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    model = MyNet() # Load Fusion model
    model = model
    model.load_state_dict(torch.load(args.model_save_path))
    model.eval()
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    for i, (_, vis_y_image, cb, cr, inf_image, name) in enumerate(test_tqdm):
        # if i > 0: break
        with torch.no_grad():
            vis_y_image = vis_y_image
            fused_image = model(vis_image=vis_y_image, nir_image=inf_image).clamp(0.0, 1.0)

        r = 1.3 # Brightness correction coefficient of Y channel after fusion
        rgb_fused_image = YCrCb2RGB(fused_image[0]**r, cb[0], cr[0])
        rgb_fused_image = transforms.ToPILImage()(rgb_fused_image.cpu())
        rgb_fused_image.save(f'{args.save_path}/{name[0]}')

import os
import random
import time
import imageio
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from loss.common import RGB2YCrCb
import cv2

to_tensor = transforms.Compose([transforms.ToTensor()])

def unevenLightCompensate(gray, blockSize):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # vis = vis.convert('L')
    gray = np.array(gray)
    average = np.mean(gray)
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))
    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]
            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)

            blockImage[r, c] = temaver

    blockImage = blockImage - 0.7*average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst[dst > 255] = 255
    dst[dst < 0] = 0
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    # dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst

def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

class JAI_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'nir':
                self.inf_path = temp_path
            if sub_dir == 'vis_blur':
                self.vis_path = temp_path
            if sub_dir == 'vis':
                self.vistruth_path = temp_path
            if sub_dir == 'mask':
                self.mask_path = temp_path

        self.name_list = os.listdir(self.inf_path)
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')
        mask_image = Image.open(os.path.join(self.mask_path, name)).convert('L')
        vis_image = Image.open(os.path.join(self.vis_path, name))
        vistruth_image = Image.open(os.path.join(self.vistruth_path, name))

        inf_image = self.transform(inf_image)
        mask_image= self.transform(mask_image)
        vis_image = self.transform(vis_image)
        vistruth_image = self.transform(vistruth_image)

        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        vistruth_y_image, vistruth_cb_image, vistruth_cr_image = RGB2YCrCb(vistruth_image)

        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, \
               vistruth_image, vistruth_y_image, vistruth_cb_image, vistruth_cr_image, mask_image, name

    def __len__(self):
        return len(self.name_list)


class JAI_data2(data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'nir':
                self.inf_path = temp_path
            if sub_dir == 'vis_blur':
                self.vis_path = temp_path
            if sub_dir == 'vis':
                self.vistruth_path = temp_path
            if sub_dir == 'mask':
                self.mask_path = temp_path

        self.name_list = os.listdir(self.inf_path)

    def __getitem__(self, index):
        name = self.name_list[index]

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')
        mask_image = Image.open(os.path.join(self.mask_path, name)).convert('L')
        vis_image = Image.open(os.path.join(self.vis_path, name))
        vistruth_image = Image.open(os.path.join(self.vistruth_path, name))

        p1, p2 = np.random.choice([0, 1]), np.random.choice([0, 1])
        my_transforms = transforms.Compose([
            # transforms.RandomResizedCrop([128]),
            transforms.RandomHorizontalFlip(p1),
            transforms.RandomVerticalFlip(p2),
            transforms.ToTensor()
        ])
        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        inf_image = my_transforms(inf_image)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        mask_image = my_transforms(mask_image)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        vis_image = my_transforms(vis_image)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        vistruth_image = my_transforms(vistruth_image)

        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        vistruth_y_image, vistruth_cb_image, vistruth_cr_image = RGB2YCrCb(vistruth_image)

        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, \
               vistruth_image, vistruth_y_image, vistruth_cb_image, vistruth_cr_image, mask_image, name

    def __len__(self):
        return len(self.name_list)
class JAI_testdata(data.Dataset):
    def __init__(self, data_dir, transform = to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'nir':
                self.inf_path = temp_path
            if sub_dir == 'vis_blur':
                self.vis_path = temp_path

        self.name_list = os.listdir(self.inf_path)
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')
        vis_image = Image.open(os.path.join(self.vis_path, name))
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)

        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name

    def __len__(self):
        return len(self.name_list)


class JAI_testdata2(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'after_HSV_blance':
                self.inf_path = temp_path
            if sub_dir == 'vis_blur':
                self.vis_path = temp_path

        self.name_list = os.listdir(self.inf_path)
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')
        vis_image = Image.open(os.path.join(self.vis_path, name))
        # inf_image = unevenLightCompensate(inf_image, 400)

        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)

        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name

    def __len__(self):
        return len(self.name_list)
class JAI_cropdata(data.Dataset):
    def __init__(self, data_dir, patch_size=96, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'nir':
                self.inf_path = temp_path
            if sub_dir == 'vis_blur':
                self.vis_path = temp_path
            if sub_dir == 'vis':
                self.vistruth_path = temp_path

        self.name_list = os.listdir(self.inf_path)
        self.patch_size = patch_size
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]

        inf_image = imageio.imread(os.path.join(self.inf_path, name), pilmode="L")
        vis_image = imageio.imread(os.path.join(self.vis_path, name), pilmode="RGB")
        vistruth_image = imageio.imread(os.path.join(self.vistruth_path, name), pilmode="RGB")

        # inf_image = cv2.imread(os.path.join(self.inf_path, name), 0)
        # vis_image = cv2.imread(os.path.join(self.vis_path, name))
        # vistruth_image = cv2.imread(os.path.join(self.vistruth_path, name))

        # crop patch randomly
        H, W = inf_image.shape
        # print(f'H = {H} , W = {W}')
        patch_size = self.patch_size
        lx, ly = random.randrange(0, W - patch_size + 1), random.randrange(0, H - patch_size + 1)
        inf_image_patch = inf_image[ly:ly + patch_size, lx:lx + patch_size]
        vis_image_patch = vis_image[ly:ly + patch_size, lx:lx + patch_size, :]
        vistruth_image_patch = vistruth_image[ly:ly + patch_size, lx:lx + patch_size, :]

        # augment data
        # print("data augmentation!")
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip: inf_image_patch, vis_image_patch, vistruth_image_patch = \
            inf_image_patch[:, ::-1], vis_image_patch[:, ::-1, :], vistruth_image_patch[:, ::-1, :]
        if vflip: inf_image_patch, vis_image_patch, vistruth_image_patch = \
            inf_image_patch[::-1, :], vis_image_patch[::-1, :, :], vistruth_image_patch[::-1, :, :]
        if rot90: inf_image_patch, vis_image_patch, vistruth_image_patch = \
            inf_image_patch.transpose(1, 0), vis_image_patch.transpose(1, 0, 2), vistruth_image_patch.transpose(1, 0, 2)

        # numpy to tensor
        inf_image_patch = np.ascontiguousarray(inf_image_patch)
        # inf_image_patch = torch.from_numpy(inf_image_patch).float()
        # inf_image_patch = torch.unsqueeze(inf_image_patch, dim=0)
        inf_image_patch = self.transform(inf_image_patch)
        vistruth_image_patch = self.transform(np.ascontiguousarray(vistruth_image_patch))
        vis_image_patch = self.transform(np.ascontiguousarray(vis_image_patch))

        # vis_image_patch, vistruth_image_patch = ndarray2tensor(vis_image_patch), ndarray2tensor(vistruth_image_patch)

        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image_patch)
        vistruth_y_image, vistruth_cb_image, vistruth_cr_image = RGB2YCrCb(vistruth_image_patch)

        # print(inf_image_patch.shape, vis_image_patch.shape, vistruth_image_patch.shape)

        return vis_image_patch, vis_y_image, vis_cb_image, vis_cr_image, inf_image_patch, \
               vistruth_image_patch, vistruth_y_image, vistruth_cb_image, vistruth_cr_image, name

    def __len__(self):
        return len(self.name_list)


if __name__ == '__main__':

    data_dir = '../test_data/test'
    dataset = JAI_testdata2(data_dir)
    start = time.time()

    for idx in range(1):
        vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name = dataset[idx]
        print(vis_image.shape, vis_y_image.shape, vis_cb_image.shape, vis_cr_image.shape, inf_image.shape, name)
    end = time.time()
    print(end - start)

import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import to_pil_image, to_tensor

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

# 將原始影像和標籤進行隨機旋轉、翻轉、縮放等變換的操作，並將其轉換為 PyTorch 張量的類
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # order=3 表示使用三次多項式插值。插值的次數越高，插值的精確性通常越高
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        # 確保 image 的大小是 (224, 224)
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = zoom(image, (224 / image.shape[0], 224 / image.shape[1]), order=3)
            label = zoom(label, (224 / label.shape[0], 224 / label.shape[1]), order=3)
        # print("image.shape:", image.shape)
        # print("label.shape:", label.shape)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # 將原始影像和標籤進行隨機旋轉、翻轉、縮放等變換的操作
        self.split = split          # 指定數據集的切分（例如，訓練集、驗證集或測試集）
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()  # list_dir: 數據樣本列表所在的目錄
        self.data_dir = base_dir    # 數據集的基本目錄，用於構造完整的數據路徑

    def __len__(self):
        return len(self.sample_list)

    # 取得數據集中指定索引（idx）的單個樣本
    def __getitem__(self, idx):
        
        if self.split == "train":  # train
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        
        else:   # test
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

            # 將 NumPy 陣列轉換為 PyTorch 張量
            image_tensor = to_tensor(image)
            label_tensor = to_tensor(label)

            # 指定目標大小
            target_size = (224, 224)

            # 進行插值
            resized_image_tensor = resize(to_pil_image(image_tensor), target_size)
            resized_label_tensor = resize(to_pil_image(label_tensor), target_size)

            # 將插值後的張量轉換回 NumPy 陣列
            image = to_tensor(resized_image_tensor).numpy()
            label = to_tensor(resized_label_tensor).numpy()

            image = image.permute(1, 2, 0)
            label = label.permute(1, 2, 0)

        sample = {'image': image, 'label': label}

        # transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

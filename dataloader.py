import os
import torch
import tqdm
import numpy as np
import pickle as pkl
import cv2
import mxnet as mx
from torch.utils.data import Dataset
import random

class UMD(Dataset):

    def __init__(self, root=r"D:\faces_umd", transforms=None):
        self.root = root
        self.transforms = transforms

        self.len_indexes = 367888
        self.num_labels = 8277

        self.recorder = mx.recordio.MXIndexedRecordIO(
            os.path.join(self.root, "train.idx"),
            os.path.join(self.root, "train.rec"), 'r')

        self.indexes_for_labels = {}
        index_path = os.path.join(self.root, "indexes.pkl")

        if not os.path.exists(index_path):
            for idx in tqdm.trange(1, self.len_indexes):
                img_mxnet = self.recorder.read_idx(idx)
                header, _ = mx.recordio.unpack(img_mxnet)
                if int(header.label) in self.indexes_for_labels:
                    self.indexes_for_labels[int(header.label)].append(idx)
                else:
                    self.indexes_for_labels[int(header.label)] = [idx]
            with open(index_path, "wb") as f:
                pkl.dump(self.indexes_for_labels, f)
        else:
            with open(index_path, "rb") as f:
                self.indexes_for_labels = pkl.load(f)

    def __getitem__(self, idx):
        img_mxnet = self.recorder.read_idx(idx+1)
        header, img = mx.recordio.unpack_img(img_mxnet)
        label = int(header.label)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return self.len_indexes


class SiameseNetworkDataset(Dataset):

    def __init__(self, data: UMD):
        self.data = data
        self.labels = list(self.data.indexes_for_labels.keys())

    def __getitem__(self, index):
        random_person1_idx = random.choice(self.labels)

        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            random_person2_idx = random_person1_idx
        else:
            random_person2_idx = random.choice(self.labels)

        img1, label1 = self.data[random.choice(self.data.indexes_for_labels[random_person1_idx])]
        img2, label2 = self.data[random.choice(self.data.indexes_for_labels[random_person2_idx])]

        return img1, img2, torch.from_numpy(np.array([1 - should_get_same_class], dtype=np.float32))

    def __len__(self):

        return len(self.data)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from torchvision import transforms

    IMAGE_SHAPE = (64, 64)

    transforms_list = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.CenterCrop((90, 90))
        # transforms.Resize(IMAGE_SHAPE),

    ])

    data = SiameseNetworkDataset(UMD(transforms=transforms_list))
    for i in range(len(data)):
        img1, img2, diff = data[i]
        print(diff)

        fig = plt.figure()
        ax_1 = fig.add_subplot(1, 2, 1)
        ax_2 = fig.add_subplot(1, 2, 2)

        ax_1.imshow(np.transpose(img1.numpy() * 255, (1, 2, 0)).astype(np.uint8))
        ax_2.imshow(np.transpose(img2.numpy() * 255, (1, 2, 0)).astype(np.uint8))
        plt.show()


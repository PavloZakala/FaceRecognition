import numpy as np
import torch
np.random.seed(7100)
from model import SiameseNetwork
from loss import ContrastiveLoss
from torch import optim
from torch.utils.data import DataLoader
from dataloader import SiameseNetworkDataset, UMD
if __name__ == '__main__':

    from torchvision import transforms

    IMAGE_SHAPE = (128, 128)

    transforms_list = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize(IMAGE_SHAPE),
    ])

    dataset = SiameseNetworkDataset(UMD(transforms=transforms_list))
    train_dataloader = DataLoader(dataset, batch_size=8)

    net = SiameseNetwork()
    if torch.cuda.is_available():
        net = net.cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, 10):
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            output1, output2 = net(img0, img1)

            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()

            optimizer.step()
            if i % 100 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, float(loss_contrastive)))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(float(loss_contrastive))
    print(counter, loss_history)



import numpy as np
import torch

np.random.seed(7100)
from model import siameseResNet
from torch import optim
from torch.utils.data import DataLoader
from dataloader import SiameseNetworkDataset, UMD
from torchvision import transforms
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch import nn

from torch.autograd import Variable

if __name__ == '__main__':

    IMAGE_SHAPE = (128, 128)
    BATCH_SIZE = 8
    EPOCH_SIZE = 1

    transforms_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.CenterCrop((90, 90)),
        transforms.Resize(IMAGE_SHAPE),
    ])

    dataset = SiameseNetworkDataset(UMD(transforms=transforms_list))
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    net = siameseResNet()
    if torch.cuda.is_available():
        net = net.cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, EPOCH_SIZE):
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):

            img0 = Variable(img0)
            img1 = Variable(img1)
            label = Variable(label)
            if torch.cuda.is_available():
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            euclidean_distance = torch.unsqueeze(torch.sigmoid(F.pairwise_distance(output1, output2)), dim=1)
            loss_contrastive = criterion(euclidean_distance, label)
            loss_contrastive.backward()
            iteration_number += 1

            optimizer.step()
            if (i + 1) % 1 == 0:
                metric = accuracy_score(F.pairwise_distance(output1, output2).detach().cpu().numpy() > 0.5,
                                        label.detach().cpu().numpy())
                print("Epoch number {} Iter num {}\n Current loss {}\n Binary_accuracy {}\n".format(epoch,
                                                                                                    iteration_number * BATCH_SIZE,
                                                                                                    float(
                                                                                                        loss_contrastive),
                                                                                                    metric))
                counter.append(iteration_number)
                loss_history.append(float(loss_contrastive))

            if (i + 1) % 1000 == 0:
                torch.save(net.state_dict(), 'best-model-parameters.pt')
    print(counter, loss_history)

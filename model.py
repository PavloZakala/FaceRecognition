from torch import nn
import torchvision.models as models

class siameseCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN = nn.Sequential(nn.Conv2d(3, 64, 10),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, 2),

                                 nn.Conv2d(64, 128, 7),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, 2),

                                 nn.Conv2d(128, 128, 4),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, 2),

                                 nn.Conv2d(128, 256, 4),
                                 nn.ReLU(inplace=True))

        self.FC = nn.Sequential(nn.Linear(8 * 8 * 256, 128),
                                nn.Sigmoid(),

                                nn.Linear(128, 1))

    def forward_img(self, x):
        x = self.CNN(x)
        x = x.view(x.size()[0], -1)
        x = self.FC(x)

        return x

    def forward(self, x1, x2):
        featvect1 = self.forward_img(x1)
        featvect2 = self.forward_img(x2)
        return featvect1, featvect2

class siameseResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.CNN = models.resnet18(num_classes=256)

    def forward_img(self, x):
        x = self.CNN(x)
        return x

    def forward(self, x1, x2):
        featvect1 = self.forward_img(x1)
        featvect2 = self.forward_img(x2)
        return featvect1, featvect2

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            # nn.ReflectionPad2d(1),
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=.5),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.5),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.5),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.5),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 12 * 12, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
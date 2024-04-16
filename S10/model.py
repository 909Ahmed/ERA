import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.prep = self.make_conv(3, 32, resnet=True)
        
        self.layer1 = self.make_conv(32, 64)
        self.res1 = self.make_res(64, 64)
        
        self.layer2 = self.make_conv(64, 128)
        self.res2 = self.make_res(128, 128)
        
        self.layer3 = self.make_conv(128, 256)
        self.res3 = self.make_res(256, 256)

        self.maxpool = nn.MaxPool2d(kernel_size=4)
        
        self.fc =  self.make_FC()
        
        
    def make_res(self, inchannels, outchannels, kernel=3, padding=1, stride=1):
        
        layers = []
        
        layers.append(self.make_conv(inchannels, outchannels, kernel=3, padding=1, stride=1, resnet=True))
        layers.append(self.make_conv(outchannels, outchannels, kernel=3, padding=1, stride=1, resnet=True))
        
        return nn.Sequential(*layers)
        
    def make_conv(self, inchannels, outchannels, kernel=3, padding=1, stride=1, resnet=False):
        
        layers = []
        
        layers.append(nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=kernel, padding=padding, stride=stride))
        if not resnet:
            layers.append(nn.MaxPool2d(kernel_size=2))
        layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.GELU())
        
        return nn.Sequential(*layers)
        
    def make_FC(self):
        
        layers = []
        
        layers.append(nn.Linear(256, 256))
        layers.append(nn.GELU())
        layers.append(nn.Linear(256, 10))
        layers.append(nn.LogSoftmax(dim=1))
        
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.prep(x)
                
        x = self.layer1(x)
        r = self.res1(x)
        x = x + r
        
        x = self.layer2(x)
        r = self.res2(x)
        x = x + r
        
        x = self.layer3(x)
        r = self.res3(x)
        x = x + r
        
        x = self.maxpool(x)
        
        x = nn.Flatten()(x)
        
        x = self.fc(x)

        return x
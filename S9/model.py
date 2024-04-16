import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.block1 = nn.Sequential(
        
            self.make_conv(inchannels=3, outchannels=32),           
            self.make_conv(inchannels=32, outchannels=32),                  
            self.make_conv(inchannels=32, outchannels=32),              
            self.make_conv(inchannels=32, outchannels=32),          
        
        )

        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.block2 = nn.Sequential(
        
            self.make_conv(inchannels=32, outchannels=64, padding=2, dilation=2, groups=32, dropout=0.2),  
            self.make_conv(inchannels=64, outchannels=64, padding=2, dilation=2, groups=64 ,dropout=0.2),  
            self.make_conv(inchannels=64, outchannels=64, padding=2, dilation=2, groups=64 ,dropout=0.2),  
            self.make_conv(inchannels=64, outchannels=64, padding=2, dilation=2, groups=64 ,dropout=0.2)
        
        )

        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.block3 = nn.Sequential(
        
            self.make_conv(inchannels=64, outchannels=128, padding=2, dilation=2, groups=64, dropout=0.3),   
            self.make_conv(inchannels=128, outchannels=128, padding=2, dilation=2, groups=128, dropout=0.3), 
            self.make_conv(inchannels=128, outchannels=128, padding=2, dilation=2, groups=128, dropout=0.3), 
            self.make_conv(inchannels=128, outchannels=128, padding=2, dilation=2, groups=128, dropout=0.3)

        )
        
        self.max_pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.block4 = nn.Sequential(
        
            self.make_conv(inchannels=128, outchannels=256, padding=2, dilation=2, groups=128, dropout=0.3),                          
            self.make_conv(inchannels=256, outchannels=256, padding=2, dilation=2, groups=256, dropout=0.3),                          

        )
        

        self.gap = nn.AvgPool2d(kernel_size=2)
        
        self.last = nn.Sequential(

             nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(2, 2), padding=0)

        )

    def make_conv(self, inchannels, outchannels, kernel=3, padding=1, dropout=0.0, stride=1, dilation=1, groups=1):
        
        layers = []
        
        if groups != 1:
            layers.append(nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=kernel, padding=padding, stride=stride, dilation=dilation, groups=groups))
            layers.append(nn.Conv2d(inchannels, outchannels, kernel_size=1))
        else:
            layers.append(nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=kernel, padding=padding, stride=stride, dilation=dilation))
        
        layers.append(nn.GELU())
        layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.Dropout(dropout))
        
        return nn.Sequential(*layers)
        

    def forward(self, x):

        x = self.block1(x)
        x = self.max_pool1(x)
        x = self.block2(x)
        x = self.max_pool2(x)
        x = self.block3(x)
        x = self.max_pool3(x)
        x = self.block4(x)
        
        x = self.gap(x)

        x = self.last(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
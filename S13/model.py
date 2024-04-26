import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import Accuracy
import torch.nn.functional as F

BATCH_SIZE = 256

class ResBlocks(LightningModule):
    
    def __init__(self, inchannels, outchannels, stride):
        super(ResBlocks, self).__init__()
        
        self.conv1 = self.make_conv(inchannels, outchannels, stride=stride)
        self.conv2 = self.make_conv(outchannels, outchannels)
    
        if stride != 1 or inchannels != outchannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride)
            )
    
    def make_conv(self, inchannels, outchannels, kernel=3, padding=1, stride=1):
        
        layers = []
        
        layers.append(nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=kernel, padding=padding, stride=stride))
        layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(x)
        out = self.conv2(out)
        
        return out + shortcut
    
class ResNet18(LightningModule):
    def __init__(self, lr=0.05):
        super(ResNet18, self).__init__()
        
        self.save_hyperparameters()
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc =  self.make_FC()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.in_layers = [64, 64, 128, 256]
        self.out_layers = [64, 128, 256, 512]
        self.strides = [1, 2, 2, 2]
        self.num = [2, 2, 2, 2]
        
        self.convin = nn.Sequential(
                nn.Conv2d(3, 64, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()   
        )
        self.res_layers = nn.ModuleList([self.make_res(self.in_layers[i], self.out_layers[i], self.num[i], self.strides[i]) for i in range(len(self.in_layers))])
    
    
    def make_res(self, inchannels, outchannels, num, stride):
        
        strides = [stride] + [1] * (num-1)
        layers = []
        
        for stride in strides:
            layers.append(ResBlocks(inchannels=inchannels, outchannels=outchannels, stride=stride))
            inchannels = outchannels
        
        return nn.Sequential(*layers)
        
    
    def make_FC(self):
        
        layers = []
        
        layers.append(nn.Linear(512, 256))
        layers.append(nn.GELU())
        layers.append(nn.Linear(256, 10))
        layers.append(nn.LogSoftmax(dim=1))
        
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.convin(x)
        
        for layer in self.res_layers:
            x = layer(x)
        
        x = self.avgpool(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
           self.parameters(),
           lr=self.hparams.lr,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer, 
                max_lr=1.26*1e-2,
                steps_per_epoch=steps_per_epoch, 
                epochs=20,
                pct_start=0.2,
                div_factor=10, 
                three_phase=False,
                final_div_factor=10,
                anneal_strategy='linear'
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
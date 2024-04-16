import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import Dataset
import albumentations as A
import numpy as np

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

class Transforms(Dataset):
    def __init__(self, images, Train=True):
        
        self.images = images
        self.transforms = A.Compose([
                                
                                A.Sequential([
                                    
                                    A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                                    A.RandomCrop (32, 32, always_apply=True, p=1.0)
                                    
                                ], p=0.5),
            
                                A.HorizontalFlip(p=0.5),
                                A.CoarseDropout(max_holes=2, min_holes=1, max_height=8, max_width=8, fill_value=0, p=0.5),
                                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.20, rotate_limit=15, p=0.2),

                        ])

        self.Train = Train
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, i):
        
        image, label = self.images[i]
                
        image = np.array(image) / 255
            
        if self.Train:
            image = self.transforms(image=image)['image']
            
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.tensor(image, dtype=torch.float), label
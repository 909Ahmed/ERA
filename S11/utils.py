from torch.utils.data import Dataset
import albumentations as A
from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch.nn.functional as F

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
    


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

def show_images(images):
    num_images = len(images)
    
    fig, axes = plt.subplots(num_images // 3 + 1, 3, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < num_images - 1:
            ax.imshow(images[i])
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.show()

def cam(rgb_img, _tensor, model):

    target_layers = [model.all_layers[2][1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=_tensor, targets=None)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    model_outputs = cam.outputs
    
    return model_outputs, visualization

def misclassified(model, device, test_loader):
    model.eval()
    test_acc = []

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            y_pred = model(data)

            pred = y_pred.argmax(dim=1, keepdim=True)

            idxs_mask = ((pred == target.view_as(pred)) == False).view(-1)

            for index, mask in enumerate(idxs_mask):
                if mask == False:
                    test_acc.append(data[index])
    
    imgs = []
    rgb_imgs = np.array([test_acc[i].cpu().numpy() for i in range(10)])    
    rgb_imgs = np.transpose(rgb_imgs, axes=[0, 2, 3, 1])
    for i in range(10):
        
        rbg_img = rgb_imgs[i]
        input_tensor = preprocess_image(rbg_img)
        img, cam_out = cam(rbg_img, input_tensor)
        imgs.append(cam_out)
        
    show_images(imgs)
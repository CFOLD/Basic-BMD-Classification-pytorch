import torch
import os
import numpy as np

from torchvision import transforms
from PIL import Image

class BmdDataset(torch.utils.data.Dataset): 
    def __init__(self,dataset_dir,transform=None,aug=True):
        self.img_list = []
        self.transform = transform
        self.aug = aug
        
        with open((dataset_dir / 'label.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                name, _class = line.split(',')
                self.img_list.append([os.path.join(str(dataset_dir),'images',name),int(_class)])

    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx][0]).convert('RGB')
        
        if self.aug == True:
            case = np.random.randint(1,9)
            if case == 2:
                img = img.transpose(Image.ROTATE_90)
            elif case == 3:
                img = img.transpose(Image.ROTATE_180)
            elif case == 4:
                img = img.transpose(Image.ROTATE_270)
            elif case == 5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif case == 6:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif case == 7:
                img = img.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
            elif case == 8:
                img = img.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        
            if np.random.rand() < 0.5:
                w, h = img.size
                size = w if w <= h else h
                img = transforms.RandomCrop(np.random.randint(int(size*0.6),int(size*0.9) ) )(img)
                
            if np.random.rand() < 0.1:
                img = transforms.TrivialAugmentWide()(img)

        if self.transform is not None:
            img = self.transform(img)
        label = self.img_list[idx][1]
        return (img, label)
import torch
import torchvision.transforms as transforms
import os
from PIL import Image

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, imgFolder,transform=transforms.ToTensor()):
        super(CelebADataset, self).__init__()
        self.imgFolder = imgFolder
        self.limg = os.listdir(self.imgFolder)
        self.transform = transform

    def __len__(self):
        return len(self.limg)

    def __getitem__(self, i):
        path = "/local/besnier/img_align_celeba/"+str(self.limg[i])
        imgx = self.transform(Image.open(path).crop((0,0,128,128)))
        if(imgx.size(0) == 1):
            imgx = imgx.expand(3, imgx.size(1), imgx.size(2))
        return imgx


def dataload(size):
    return CelebADataset("/local/besnier/img_align_celeba",
                         transforms.Compose([transforms.Resize(size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             ]))

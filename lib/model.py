import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import os
import PIL

class MnistModel(nn.Module):
    def __init__(self, n_filters1=64,
            n_filters2=64,
            n_fc=256,
            dropout=False):

        super(MnistModel, self).__init__()

        self.n_filters1 = n_filters1
        self.n_filters2 = n_filters2
        self.n_fc = n_fc
        self.dropout = dropout

        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, self.n_filters1, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(self.n_filters1, self.n_filters2, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(self.n_filters2*7*7, self.n_fc)
        self.fc2 = nn.Linear(self.n_fc, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.n_filters2*7*7)   # reshape Variable
        x = F.relu(self.fc1(x))
        if self.dropout: x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class NotMnist(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filelist = glob.glob(os.path.join(self.root_dir, '**', '*.png'))
        new_filelist = []
        for file in self.filelist:
            try:
                io.imread(file)
                new_filelist.append(file)
            except:
                pass
                
        self.filelist = new_filelist
        
    def __len__(self):
        return len(self.filelist)
        
    def __getitem__(self, idx):
        image = io.imread(self.filelist[idx])
        image = PIL.Image.fromarray(image)
        if self.transform:
            return self.transform(image)
        return image
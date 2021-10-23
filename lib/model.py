import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import os
import PIL

class MnistModel(nn.Module):
    def __init__(self, n_filters1=64,
            n_filters2=64,
            n_fc=256,
            dropout=False, output_logits=False):

        super(MnistModel, self).__init__()

        self.n_filters1 = n_filters1
        self.n_filters2 = n_filters2
        self.n_fc = n_fc
        self.dropout = dropout
        self.output_logits = output_logits

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
        
        if self.output_logits:
            return x
        
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

class LeNet(nn.Module):
    def __init__(self, output_logits=False):
        super(LeNet, self).__init__()
        self.output_logits = output_logits
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        if self.output_logits:
            return x
        
        return F.log_softmax(x, dim=1)

class LeNetMnist(nn.Module):
    def __init__(self, output_logits=False):
        super(LeNetMnist, self).__init__()
        self.output_logits = output_logits
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.output_logits:
            return x
        
        return F.log_softmax(x, dim=1)

# this class is taken from: https://github.com/fregu856/evaluating_bdl/blob/master/toyClassification/SGLD-64/model.py
class ToyNet(nn.Module):
    def __init__(self, model_id, project_dir):
        super(ToyNet, self).__init__()

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        input_dim = 2
        hidden_dim = 10
        num_classes = 2

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # (x has shape (batch_size, input_dim))

        out = F.relu(self.fc1(x)) # (shape: (batch_size, hidden_dim))
        out = F.relu(self.fc2(out)) # (shape: (batch_size, hidden_dim))
        out = self.fc3(out) # (shape: batch_size, num_classes))

        return out

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PDLNet(nn.Module):
    # [ B * N * (3+z) ] -> # [ B * N * 3 ] 
    def __init__(self, size_z, num_point):
        super(PDLNet, self).__init__()
        size_kernel = 1
        size_pad = 0

        self.size_z = size_z
        self.num_point = num_point
        self.conv1 = torch.nn.Conv1d(3 + self.size_z, 128, size_kernel, padding=size_pad)
        self.conv2 = torch.nn.Conv1d(128, 32, size_kernel, padding=size_pad)
        self.conv3 = torch.nn.Conv1d(32, 3, size_kernel, padding=size_pad)
        
        self.conv4 = torch.nn.Conv1d(3 + self.size_z, 128, size_kernel, padding=size_pad)
        self.conv5 = torch.nn.Conv1d(128, 32, size_kernel, padding=size_pad)
        self.conv6 = torch.nn.Conv1d(32, 3, size_kernel, padding=size_pad)
        
        self.ln0 = nn.LayerNorm((self.size_z , num_point))
        self.ln1 = nn.LayerNorm((128, num_point))
        self.ln2 = nn.LayerNorm((32, num_point))
        self.ln3 = nn.LayerNorm((3, num_point))
        self.ln4 = nn.LayerNorm((128 , num_point))
        self.ln5 = nn.LayerNorm((32, num_point))
        self.ln6 = nn.LayerNorm((3, num_point))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)

    def forward(self, x_z, x, z):
        z = self.ln0(z)
        x = torch.cat([z, x], 1)
        x = self.dropout(F.relu(self.ln1(self.conv1(x))))
        x = self.dropout(F.relu(self.ln2(self.conv2(x))))
        x = self.dropout(F.relu(self.ln3(self.conv3(x))))
        
        x = torch.cat([z, x], 1)
        x = self.dropout(F.relu(self.ln4(self.conv4(x))))
        x = self.dropout(F.relu(self.ln5(self.conv5(x))))
        x1 = self.dropout((self.ln6(self.conv6(x))))
        return x1




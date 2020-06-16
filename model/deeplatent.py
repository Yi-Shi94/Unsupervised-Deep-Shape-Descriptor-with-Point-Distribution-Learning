from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import sys
 
sys.path.append('../')

from loss.chamfer import ChamferDistance
from model.networks import PDLNet

class DeepLatent(nn.Module):
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, latent_length, n_samples = 1024, chamfer_weight=0):
        super(deeplatent, self).__init__()
        self.latent_length = latent_length
        self.pdl_net = PDLNet(latent_length,n_samples)
        self.chamfer_dist = ChamferDistance()
        self.L2_dist = nn.MSELoss()
        self.chamfer_weight = chamfer_weight
    
    def forward(self, pc, pc_gt, latent):
        self.pc = deepcopy(pc)
        self.pc_gt = deepcopy(pc_gt)
        pc_with_lat = torch.cat([self.pc,latent], 1)
        self.pc_est = self.pdl_net(pc_with_lat, self.pc, latent)
        loss = self.compute_loss()
        return loss

    def compute_loss(self):
        loss_chamfer = self.chamfer_dist(self.pc_gt,self.pc_est)
        loss_L2 = self.L2_dist(self.pc_gt,self.pc_est)
        loss = self.chamfer_weight * loss_chamfer + (1 - self.chamfer_weight)*loss_L2
        return loss,loss_chamfer,loss_L2    
    


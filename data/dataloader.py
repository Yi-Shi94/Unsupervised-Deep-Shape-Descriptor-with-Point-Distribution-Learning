import os
import glob
import random
import numpy as np
import open3d as o3d
import tqdm
import torch
import math
from copy import deepcopy

#from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
#import utils

def open_mesh(filename):
    pc = o3d.read_point_cloud(filename)
    return pc

def get_mean_dis(pc):
    center = np.sum(pc)
    mean_dis = np.mean(pc-center)
    print(mean_dis)
    return mean_dis

def generate_random_drift(pc,sigma=0.02):
    # pc [N*3] 
    # sigma [1],for controlling the random distance
    # pc_gen [N*3]
    pc_gen = torch.zeros(0)
    for pt in pc:  
        #pt [1*3] x y z
        pt = torch.FloatTensor(pt)
        pt_new = torch.normal(pt, sigma, out=None).unsqueeze(-1).transpose(0,1)
        pc_gen = torch.cat([pc_gen, pt_new], 0)
    return pc_gen

'''
def generate_random_drift(pc,sigma=0.02):
    cov = [[sigma,0,0],[0,sigma,0],[0,0,sigma]]
    pc_gen = np.array([np.random.multivariate_normal(mean, cov, 1) for mean in pc]).squeeze()
    pc_gen = torch.FloatTensor(pc_gen)
    return pc_gen
'''

class ShapeNet(Dataset):
    def __init__(self,root,device,mode='train',sigma =0.04):
        self.sigma = sigma
        self.device = device
        self.mode = mode
        
        if mode=='both':
            data_train = np.load(os.path.join(root,'train','pntcloud_full.npy'))
            data_test = np.load(os.path.join(root,'test','pntcloud_full.npy'))
            self.data = np.concatenate((data_train,data_test),axis=0)
            label_train = np.load(os.path.join(root,'train','label_full.npy'))
            label_test = np.load(os.path.join(root,'test','label_full.npy'))
            self.label = np.concatenate((label_train,label_test),axis=0)
        else:
            self.data = np.load(os.path.join(root,'train','pntcloud_7.npy'))
            self.label = np.load(os.path.join(root,'train','label_7.npy'))
        
        self.train_num = self.label.shape[0]
        self.indices = range(self.data.shape[0])
        
    def __getitem__(self,index,is_online=True):
        pc_gt = self.data[index]
        sig = self.sigma
        pc = generate_random_drift(pc_gt,sigma=sig).tolist()
        pc = torch.FloatTensor(pc).transpose(0,1) 
        pc_gt = torch.FloatTensor(pc_gt).transpose(0,1)
        return pc, pc_gt, self.indices[index]

    def __len__(self):
        return len(self.data)
    
    
###################################################################################


class ModelNet40(Dataset):
    def __init__(self,root,device,mode='train',sigma =0.08):
        self.sigma = sigma
        self.device = device
        self.mode = mode

        if mode=='both':
            data_train = np.load(os.path.join(root,'train','pntcloud.npy'))
            data_test = np.load(os.path.join(root,'test','pntcloud.npy'))
            self.data = np.concatenate((data_train,data_test),axis=0)
            label_train = np.load(os.path.join(root,'train','label.npy'))
            label_test = np.load(os.path.join(root,'test','label.npy'))
            self.label = np.concatenate((label_train,label_test),axis=0)
    
        else:
            self.data = np.load(os.path.join(root,mode,'pntcloud.npy'))
            self.label = np.load(os.path.join(root,mode,'label.npy'))
            
        self.indices = range(self.data.shape[0])
        
    def __getitem__(self,index,is_online=True):
        pc_gt = self.data[index]
        sig = self.sigma
        pc = generate_random_drift(pc_gt,sigma=sig).tolist()
        pc = torch.FloatTensor(pc).transpose(0,1) 
        pc_gt = torch.FloatTensor(pc_gt).transpose(0,1)
        return pc, pc_gt, self.indices[index]

    def __len__(self):
        return len(self.data)

###################################################################################

class ModelNet40_multi(Dataset):
    def __init__(self,name,root,device,sigma =0.08, block_num = None, start = None, block_size = None):
        assert block_num is not None and block_size is not None
        print(block_num,block_size)
        self.block_size = int(block_size)
        self.block_num = int(block_num)
        self.save_dir = 'results/%s/md40_subset%d/'%(name,block_num))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.train_num = 12308
        self.sigma = sigma
        self.device = device
        
        data_train = np.load(os.path.join(root,'train','pntcloud.npy'))
        data_test = np.load(os.path.join(root,'test','pntcloud.npy'))
       
        self.data = np.concatenate((data_train,data_test),axis=0)[start:start+self.block_size,:]
            
        label_train = np.load(os.path.join(root,'train','label.npy'))
        label_test = np.load(os.path.join(root,'test','label.npy'))
        
        self.label = np.concatenate((label_train,label_test),axis=0)[start:start+self.block_size]
        self.indices = range(self.block_size)
        
    def __getitem__(self,index,is_online=True):
        pc_gt = self.data[index]
        sig = self.sigma
        pc = generate_random_drift(pc_gt,sigma=sig).tolist()
        pc = torch.FloatTensor(pc).transpose(0,1) 
        pc_gt = torch.FloatTensor(pc_gt).transpose(0,1)
        return pc, pc_gt, self.indices[index]

    def __len__(self):
        return len(self.data)
    

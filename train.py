import argparse
import os
import numpy as np
import tqdm

from matplotlib import cm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

from data.dataloader import *
from model.deeplatent import *
from model.networks import *
from utils.utils import *

parser = argparse.ArgumentParser(description='3D auto decoder')
parser.add_argument('-r','--root', type=str, default='datasets/ShapeNet', help='data_dir')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--debug', default=True, type=lambda x: (str(x).lower() == 'true'),help='load part of the dataset to run quickly')
parser.add_argument('-y','--latent_size', type=int, default=128, help='length_latent')
parser.add_argument('--weight_file', default='', help='path to weights to load')
parser.add_argument('--name', type=str, default='default', help='name of experiment (continue training if existing)')
parser.add_argument('-s','--seed',type=str,default=42,help="seed string")
parser.add_argument('--log_interval',type=str, default=1,help="log_interval")
parser.add_argument('--sample_num',type=int, default=2048,help="num_point")
parser.add_argument('--resume',type=bool, default=False,help="if load model")
parser.add_argument('--save_dir',type=str, default='results/',help='directory for saving weights')
parser.add_argument('--sigma',type=float, default=0.08,help='sigma')
parser.add_argument('--mode',type=str, default='train',help='training mode')
opt = parser.parse_args()
print(opt)

torch.manual_seed(opt.seed)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = opt.root

dataset = ShapeNet(root, device, opt.mode, opt.sigma)
loader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=True)

checkpoint_dir = opt.save_dir
if not os.path.isdir(checkpoint_dir):
   os.mkdir('results')
    
latent_size = opt.latent_size
num_total_instance = len(dataset)
num_batch =  np.ceil(num_total_instance/opt.batch_size)

model = DeepLatent(latent_length = latent_size, n_samples = opt.sample_num, chamfer_weight = 0)

latent_vecs = []
for i in range(len(dataset)):
    vec = (torch.ones(latent_size).normal_(0, 0.9).to(device))
    vec = torch.nn.Parameter(vec)
    latent_vecs.append(vec)

optimizer = optim.Adam([
                {
                    "params":model.parameters(), "lr":opt.lr,
                },
                {
                    "params": latent_vecs, "lr":opt.lr*0.5,
                }
            ]
            )

if opt.resume:
    model, latent_vecs, optimizer = load_checkpoint(os.path.join(checkpoint_dir,'model_best'),model,optimizer) 

model.to(device)

min_loss = float('inf')
for epoch in range(opt.epochs):
    training_loss= 0.0
    model.train()
    for index,(shape_batch,shape_gt_batch,latent_indices) in enumerate(loader):
        
        lats_inds =  latent_indices.cpu().detach().numpy()
        latent_inputs = torch.ones((lats_inds.shape[0],opt.latent_size), dtype=torch.float32, requires_grad=True).to(device) 
        i = 0
        for i_lat in lats_inds:
            latent_inputs[i] *= latent_vecs[i_lat]
            i += 1
        latent_repeat = latent_inputs.unsqueeze(-1).expand(-1,-1,shape_batch.size()[-1])
        shape_batch = shape_batch.to(device)
        shape_gt_batch = shape_gt_batch.to(device)
        loss,chamfer,l2 = model(shape_batch,shape_gt_batch,latent_repeat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        
        print("Epoch:[%d|%d], Batch:[%d|%d]  loss: %f , chamfer: %f, l2: %f"%(epoch,opt.epochs,index,num_batch,loss.item()/opt.batch_size,chamfer.item()/opt.batch_size,l2.item()/opt.batch_size))
        
    training_loss_epoch = training_loss/(len(loader)*opt.batch_size)

    if training_loss_epoch < min_loss:
        min_loss = training_loss_epoch
        print('New best performance! saving')
        save_name = os.path.join(checkpoint_dir,'model_best')
        save_checkpoint(save_name,model,latent_vecs,optimizer)

    if (epoch+1) % opt.log_interval == 0:
        save_name = os.path.join(checkpoint_dir,'model_routine')
        save_checkpoint(save_name,model,latent_vecs,optimizer)








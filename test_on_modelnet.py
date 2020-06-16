import argparse
import os
import numpy as np
import tqdm

from matplotlib import cm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from data.dataloader import *
from model.deeplatent import *
from model.networks import *
from utils.utils import *

parser = argparse.ArgumentParser(description='3D auto decoder')
parser.add_argument('-r','--root', type=str, default='datasets/ModelNet40', help='data_root')
parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
#manual stop
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('-y','--latent_size', type=int, default=128, help='length_latent')
parser.add_argument('--weight_file', default='', help='path to weights to load')
parser.add_argument('--name', type=str, default='default', help='name of experiment')
parser.add_argument('-s','--seed',type=str,default=42,help="seed string")
parser.add_argument('--log_interval',type=str, default=1,help="log_interval")
parser.add_argument('--sample_num',type=int, default=2048,help="num_point")
parser.add_argument('--mode',type=str, default='test',help="loading mode")
parser.add_argument('--save_dir',type=str, default=None,help="save directory")
parser.add_argument('--load_dir',type=str, default=None,help="load model directory")
parser.add_argument('--sigma',type=float, default=0.08,help="sigma")
parser.add_argument('--gpuid',type=str,default=0,help="gpu_id")
opt = parser.parse_args()
print(opt)

torch.manual_seed(opt.seed)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= opt.gpuid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = opt.root

dataset = ModelNet40(root,device,mode=opt.mode,sigma=opt.sigma)
loader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=True,num_workers=8)

if opt.load_dir is not None:
    checkpoint_dir = opt.load_dir
else:
    checkpoint_dir = 'results/'

if opt.save_dir is not None:
    save_dir = opt.save_dir
else:
    save_dir = 'results/'

latent_size = opt.latent_size
model = DeepLatent(latent_length = latent_size, n_samples = opt.sample_num)

latent_vecs = []
for i in range(len(dataset)):
    vec = (torch.ones(latent_size).normal_(0, 0.8).to(device))
    vec.requires_grad = True
    latent_vecs.append(vec)

optimizer = optim.Adam([
                {
                     "params": latent_vecs, "lr":0.01,
                }
            ]
            )

model, _ , _  = load_checkpoint(os.path.join(checkpoint_dir,'model_best'),model,optimizer) 
model.to(device)

min_loss = float('inf')
for epoch in range(opt.epochs):
    training_loss= 0.0
    model.train()
    for index,(shape_batch,shape_gt_batch,latent_indices) in enumerate(loader):
        latent_inputs = torch.zeros(0).to(device)
        
        for i_lat in latent_indices.cpu().detach().numpy():
            latent = latent_vecs[i_lat] 
            latent_inputs = torch.cat([latent_inputs, latent.unsqueeze(0)], 0)
            
        latent_repeat = latent_inputs.unsqueeze(-1).repeat(1,1,shape_batch.size()[-1])
        shape_batch = shape_batch.to(device)
        shape_gt_batch = shape_gt_batch.to(device)
        loss,chamfer,l2 = model(shape_batch,shape_gt_batch,latent_repeat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
       
        print("Epoch:[%d|%d], Batch:%d  loss: %f , chamfer: %f, l2: %f"%(epoch,opt.epochs,index,loss.item()/opt.batch_size,chamfer.item()/opt.batch_size,l2.item()/opt.batch_size))
        
    training_loss_epoch = training_loss/(len(loader)*opt.batch_size)

    if training_loss_epoch < min_loss:
        min_loss = training_loss_epoch
        print('New best performance! saving')
        save_name = os.path.join(save_dir,'model_best_test')
        save_checkpoint(save_name,model,latent_vecs,optimizer)

    if (epoch+1) % opt.log_interval == 0:
        save_name = os.path.join(save_dir,'model_routine_test')
        save_checkpoint(save_name,model,latent_vecs,optimizer)

save_name = os.path.join(checkpoint_dir,save_dir,'model_test')
save_checkpoint(save_name,model,latent_vecs,optimizer)

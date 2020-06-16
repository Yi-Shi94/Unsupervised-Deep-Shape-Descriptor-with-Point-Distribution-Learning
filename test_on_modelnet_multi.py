import argparse
import os
import numpy as np
import tqdm
import time
from matplotlib import cm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.multiprocessing import Pool, Process, set_start_method

try:
     set_start_method('spawn')
except RuntimeError:
    pass

from data.dataloader import *
from model.deeplatent import *
from model.networks import *
from utils.utils import *

def train(model,dataset,device,latent_size,n_samples,if_continue):
    save_dir = dataset.save_dir
    
    batch_size = 1
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    
    latent_vecs = []
    if if_continue is True:
        print("continuing from chekpt")
        save_name = os.path.join(save_dir,'model_best_test')
        _, z_lst, _ = load_checkpoint(save_name, None, None)
        
        for i in range(len(z_lst)):
            #print(z_lst[i])
            vec = (torch.ones(latent_size).normal_(0, 0.9).to(device))
            vec.data = z_lst[i].data
            vec.requires_grad = True
            latent_vecs.append(vec)
        
    else:
        for i in range(len(dataset)):
            vec = (torch.ones(latent_size).normal_(0, 0.9).to(device))
            vec.requires_grad = True
            latent_vecs.append(vec)
        
    optimizer = optim.Adam([
                {
                "params": latent_vecs, "lr":0.05,
                }
                ])
    
    min_loss = float('inf')
    model.to(device)
    for epoch in range(120):
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
       
        training_loss_epoch = training_loss/(len(loader)*batch_size)

        if training_loss_epoch < min_loss:
            min_loss = training_loss_epoch
            save_name = os.path.join(save_dir,'model_best_test')
            save_checkpoint(save_name,model,latent_vecs,optimizer)
        
        print('process: %d'%dataset.block_num,' epoch: %d'%epoch, ' loss: %d'%training_loss_epoch)
        save_name = os.path.join(save_dir,'model_routine_test')
        save_checkpoint(save_name,model,latent_vecs,optimizer)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D auto decoder for tracking')
    parser.add_argument('-r','--root', type=str, default='datasets/ModelNet40', help='data_dir')
    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    #manual stop
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-y','--latent_size', type=int, default=128, help='length_latent')
    parser.add_argument('--weight_file', default='', help='path to weights to load')
    parser.add_argument('-s','--seed',type=str,default=42,help="seed string")
    parser.add_argument('--log_interval',type=str, default=1,help="log_interval")
    parser.add_argument('--sample_num',type=int, default=2048,help="num_point")
    parser.add_argument('--resume',type=str,help="if load model")
    parser.add_argument('--load_dir',type=str, default='results/md40_sigma008_filtered',help="load model directory")
    parser.add_argument('--sigma',type=float, default=0.08,help="sigma")
    parser.add_argument('--gpuid',type=str,default='0',help="gpu_id")
    opt = parser.parse_args()

    torch.manual_seed(opt.seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= opt.gpuid
    torch.backends.cudnn.benchmark = False
    root = opt.root
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = DeepLatent(latent_length = opt.latent_size, n_samples = opt.sample_num)
    
    model, _ , _  = load_checkpoint(os.path.join(opt.load_dir, 'model_best'),model,None) 
    
    name = os.path.split(opt.load_dir)[-1]
    
    if opt.resume!= '1':
        if_continue = False
    else:
        if_continue = True 
        
    num_instance = 12308
    block_size = 1000
    num_processes= int(np.ceil(num_instance/block_size))
   
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        gpuid = str(rank%2+2)
        
        if rank==12:
            os.environ["CUDA_VISIBLE_DEVICES"]= '3'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
        if rank == num_processes-1:
            cur_block_size = num_instance%block_size
        else:
            cur_block_size = block_size
        
        dataset = ModelNet40_multi(name,root,device,sigma=opt.sigma,block_num=rank,
                                   start = rank*block_size,block_size=cur_block_size)
        
        p = Process(target=train, args=(model,dataset,device,opt.latent_size,opt.sample_num,if_continue))
        
        p.start()
        print('process ID: '+str(rank+1)+ '/'+str(num_processes)+' started!')
        processes.append(p)
        time.sleep(4)
    for p in processes:
        p.join()
    
    
    
    
    
    
    

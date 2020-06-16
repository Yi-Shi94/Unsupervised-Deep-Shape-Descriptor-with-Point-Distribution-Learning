import torch 
import torch.optim as optim

def save_checkpoint(save_name, model, z, optimizer):
    state = {
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()
             }
    z_ts = torch.stack(z)
    torch.save(z_ts, save_name+'_latents.pt')
    torch.save(state, save_name+'.pth')
    print('model saved to {}'.format(save_name))

def load_checkpoint(save_name, model, optimizer):
    
    z_ts = torch.load(save_name+'_latents.pt')
    z_lst = []
    for i in range(z_ts.size()[0]):
        z_lst.append(z_ts[i,:])
    if model is None:
        pass
    else:
        model_CKPT = torch.load(save_name+'.pth')
        model.load_state_dict(model_CKPT['state_dict'])
        #model.cuda()
        #optimizer = optim.Adam(model.parameters())
        print('loading checkpoint!')
        #optimizer.load_state_dict(model_CKPT['optimizer'])
    
    return model, z_lst, optimizer
    

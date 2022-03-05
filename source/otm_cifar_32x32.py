##########################################################
## Standard Imports
##########################################################
import matplotlib.pyplot as plt
import numpy as np
import os, sys, random
from tqdm import tqdm
import argparse
import json
import cv2
import utils
import warnings
warnings.filterwarnings('ignore')


##########################################################
## DL Imports
##########################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from torch import autograd
from torchvision.utils import save_image
from torch.autograd import Variable
import inception

##########################################################
from src.plotters import plot_noise_interp_unequal, plot_inv_noise_interp_unequal



##########################################################
## Hyper-parameters
##########################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

SEED = 9999
torch.manual_seed(SEED)
path = '../input/CIFAR10/'
output_path = './output/CIFAR10/'  
pretrain_path = './pretrained/CIFAR10/'
inception_path = './Eval/utils/output/CIFAR10/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


if not os.path.exists(output_path):
    print('Creating output directory...', output_path)
    os.makedirs(output_path)
    
if not os.path.exists(pretrain_path):
    print('Creating pretrain weights directory...', pretrain_path)
    os.makedirs(pretrain_path)

if not os.path.exists(inception_path):
    print('Creating inception directory...', inception_path)
    os.makedirs(inception_path)
 
##########################################################
## Parse Arguments
##########################################################
parser = argparse.ArgumentParser()
parser.add_argument('--train', dest= 'train', type=int, default=0)
parser.add_argument('--init_iter', dest= 'init_iter', type=int, default=-1)
parser.add_argument('--restore', dest= 'restore', type=int, default=0)
parser.add_argument('--inference', dest= 'inference', type=int, default=0)
parser.add_argument('--base_cfg_path', default='configs/base_config.json', type=str, help='path to base config file')
parser.add_argument('--cfg_path', default='configs/resnet_OTM_GO.json', type=str, help='path to config file')
args = parser.parse_args()

train_model = args.train
init_iter = args.init_iter # initial iteration
restore_model = args.restore # Restart training from init_epoch (iteration) checkpoint
inference = args.inference # Run inference engine

##########################################################
with open(args.base_cfg_path, "r") as fp:
        configs = json.load(fp)

# Overriding base configs
if args.cfg_path != '':
    with open(args.cfg_path, "r") as fp:
        exp_configs = json.load(fp)
    for k in exp_configs.keys():
        configs[k] = exp_configs[k]

configs['gen_iters'] = 1
configs['go_lamb'] = 10
configs['lrD'] = 1e-3

print('configs:\n ', configs)
configs = utils.ConfigMapper(configs)


##########################################################
T = configs.nepochs # Total epochs
latent_dim = configs.nz # Input dimension
size = configs.imageSize # Size of each image, [size,size]
channels = configs.nc # Number of channels, [channels,size,size]
num_workers = 10 # Used in data loader
devices = [0]


## Training parameters
M = configs.batchSize # Batch size
C = configs.gen_iters # Number of G updates

lam_go = configs.go_lamb # Coefficient of gradient penalty term

parallel = True # Data parallelization, if multiple gpus are available
save_model = True 

##########################################################
## Prepare Data
##########################################################
transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5 ), (0.5, 0.5, 0.5)),
        ])

train_data = datasets.CIFAR10(root=path, train=True,
                                   download=True, transform=transform)
test_data = datasets.CIFAR10(root=path, train=False,
                                  download=True, transform=transform)

print('Train data: ', len(train_data), 'Test data: ', len(test_data) )

train_loader = torch.utils.data.DataLoader(train_data, batch_size=M, num_workers=num_workers, shuffle=True, drop_last = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=M, num_workers=num_workers, shuffle=False, drop_last = True)

train_loader_iterator = iter(train_loader)
test_loader_iterator = iter(test_loader)

##########################################################
# Embeddings
Q = lambda x: F.interpolate(x.reshape(-1, 3, 8, 8), size, mode='bicubic').detach() 
INV_TRANSFORM = lambda x: 0.5*x + 0.5

##########################################################
## Main Modules
##########################################################
from resnet import Generator 
from resnet import Discriminator


print('='*64)
print('G Architecture: \n')
G = Generator(configs).to(device)

ip = torch.randn(2,latent_dim).to(device)
op = G(ip)

print('ip: ', ip.shape)
print('op: ', op.shape)

print('='*64)

# sys.exit()

##########################################################
print('='*64)
print('Psi Architecture: \n')
psi = Discriminator(configs).to(device)
# summary(psi,(channels,size,size))

ip = torch.randn(2,channels,size,size).to(device)
op = psi(ip)

print('ip: ', ip.shape)
print('op: ', op.shape)

print('='*64)

# sys.exit()

##########################################################
def GradientOptimality(psi, G, Q, x):
    """ Gradient Optimality cost for potential"""
    G_x = G(x).cuda()
    G_x.requires_grad_(True)
    psi_G_x = psi(G_x)

    gradients = autograd.grad(
        outputs=psi_G_x, inputs=G_x,
        grad_outputs=torch.ones(psi_G_x.size()).to(G_x),
        create_graph=True, retain_graph=True
    )[0]
    return (gradients.mean(dim=0) - Q(x).mean(dim=0)).norm('fro')


##########################################################
def Loss(psi, G, Q, x, y):
    G_x = G(x)
    dot = torch.mean(Q(x)*G_x, dim=(1,2,3)).unsqueeze(dim=1)
    loss = ( dot - psi(G_x) + psi(y)).mean()
    # print(loss.item())
    
    # sys.exit()
    return loss

def SaveModel(model, name='OTM', path=pretrain_path):
    torch.save(model.state_dict(), path+name+'.ckpt')
    print('Model saved at '+path+name+'.ckpt')

def LoadModel(model, name='OTM', path=pretrain_path):
    model.load_state_dict(torch.load(path+name+'.ckpt'))
    print('Model loaded from '+path+name+'.ckpt')
    return model


def Test(epoch,G,test_loader_iterator):
    ## Compute Transport plan for evaluation
    try:
        y, _ = next(test_loader_iterator)
    except StopIteration:
        test_loader_iterator = iter(test_loader)
        y, _ = next(test_loader_iterator)

    y = y.to(device)

    x = np.random.normal(0,1, (M, latent_dim))
    x = torch.FloatTensor(x).to(device)
    
    Gx = G(x)

    Gx = Gx.mul(0.5).add(0.5)
    y = y.mul(0.5).add(0.5)

    # for visualization
    Gx = Gx[:64]
    y = y[:64]
    
    save_image(Gx.view(Gx.shape[0], channels, size, size), output_path+'fake_sample_iter_'+str(epoch)+'.png', nrow=8, normalize=True)
    # save_image(y.view(y.shape[0], channels, size, size), output_path+'real_sample_iter_'+str(epoch)+'.png', nrow=8, normalize=True)

    return test_loader_iterator

def Eval(epoch,G,train_loader_iterator):
    ## Compute Transport plan for evaluation
    try:
        y, _ = next(train_loader_iterator)
    except StopIteration:
        train_loader_iterator = iter(train_loader)
        y, _ = next(train_loader_iterator)

    y = y.to(device)
    
    x = np.random.normal(0,1, (M,latent_dim))
    x = torch.FloatTensor(x).to(device)
    
    Gx = G(x)

    Gx = Gx.mul(0.5).add(0.5)
    y = y.mul(0.5).add(0.5)

    # for visualization
    Gx = Gx[:64]
    y = y[:64]
    
    save_image(Gx.view(Gx.shape[0], channels, size, size), output_path+'fake_sample_iter_'+str(epoch)+'.png', nrow=8, normalize=True)
    save_image(y.view(y.shape[0], channels, size, size), output_path+'real_sample_iter_'+str(epoch)+'.png', nrow=8, normalize=True)

    return train_loader_iterator


##########################################################
## Training Unit
##########################################################

if train_model:
    print('Start training...')
    # initialize model parameters
    G = Generator(configs)
    psi = Discriminator(configs)


    if parallel:
        G = nn.DataParallel(G, devices)
        psi = nn.DataParallel(psi, devices)

    if cuda:
        G.to(device)
        psi.to(device)

    print('Model initlized...')

    ##########################################################
    ## Restote Unit
    ##########################################################
    if restore_model:
        G = LoadModel(G, 'otm_G_iter_'+str(init_epoch))
        psi = LoadModel(psi, 'otm_psi_iter_'+str(init_epoch))

    n = 20
    X_fixed = np.random.normal(0,1, (n, latent_dim))
    try:
        Y_fixed, _ = next(train_loader_iterator)
    except StopIteration:
        train_loader_iterator = iter(train_loader)
        Y_fixed, _ = next(train_loader_iterator)
    
    X_fixed = torch.FloatTensor(X_fixed).to(device)
    Y_fixed = Y_fixed[:n].to(device)

    # Define optimizers
    G_opt = optim.Adam(G.parameters(), betas=(configs.beta1G, configs.beta2G), lr = configs.lrG)
    psi_opt = optim.Adam(psi.parameters(), betas=(configs.beta1D, configs.beta2D), lr = configs.lrD)


    # Define placeholders
    psi_l = []
    G_l = []
    Go_l = []

   # Start training
    print('Running training engine...')
    init_epoch_start = init_iter*M//len(train_data)
    
    for epoch in tqdm(range(init_epoch_start+1,T)):
        G_l_ = []

        for i, (y_batch, _) in enumerate(train_loader):
            t = epoch*len(train_data)//M + i # counts number of batches
            ##########################################################
            ## Inner maximization loop
            ##########################################################
            x_batch = np.random.normal(0,1, (M, latent_dim))
            x_batch = torch.FloatTensor(x_batch).to(device)
            y_batch = y_batch.to(device)
            
            # print('Data: ', x_batch.shape, y_batch.shape)
            
            # sys.exit()
            ## Fix psi and update G to compute convex conjugate/ supremum 
            for param in psi.parameters():
                param.requires_grad = False
            for param in G.parameters():
                param.requires_grad = True

            G_loss = -Loss(psi, G, Q, x_batch, y_batch) 
            G_opt.zero_grad(); G_loss.backward(retain_graph=True); G_opt.step()
            G_l_.append(G_loss.item())

            ##########################################################
            ## Outer minimization loop: update once for every C updates
            ##########################################################       
            if i%C==0:
                G_l.append(np.asarray(G_l_).mean())   
                G_l_ = [] 
                ## Fix G and update psi to compute infimum.
                x_batch = np.random.normal(0,1, (M, latent_dim))
                x_batch = torch.FloatTensor(x_batch).to(device)

                try:
                    y_batch, _ = next(train_loader_iterator)
                except StopIteration:
                    train_loader_iterator = iter(train_loader)
                    y_batch, _ = next(train_loader_iterator)
                
                y_batch = y_batch.to(device)
                
                for param in psi.parameters():
                    param.requires_grad = True
                for param in G.parameters():
                    param.requires_grad = False

                go_loss = GradientOptimality(psi, G, Q, x_batch)
                psi_loss = Loss(psi, G, Q, x_batch, y_batch) + lam_go * go_loss

                psi_opt.zero_grad(); psi_loss.backward(retain_graph=True); psi_opt.step()

                psi_l.append(psi_loss.item())
                Go_l.append(go_loss.item())
                            


            if t%configs.log_every==0:
                print("Epoch: ", epoch,"|",T, "\t psi loss: ", np.round(psi_l[-1],3),"\t GO loss: ",np.round(Go_l[-1],4), "\t G loss: ", np.round(G_l[-1],3)) 
                
                fig, axes = plot_noise_interp_unequal(G, Q, X_fixed, Y_fixed, INV_TRANSFORM, show=False, n=n)
                fig.savefig(output_path+'OTM_samples.pdf', bbox_inches='tight')
                plt.close(fig)

                fig, axes = plot_inv_noise_interp_unequal(G, psi, Q, X_fixed, Y_fixed, INV_TRANSFORM, show=False, n=n)
                fig.savefig(output_path+'OTM_inv_samples.pdf', bbox_inches='tight')
                plt.close(fig)

                if t<=configs.test_every:
                    train_loader_iterator = Eval(t,G,train_loader_iterator)
                
            if t%configs.save_every==0 and save_model:
                SaveModel(psi, 'otm_psi_iter_'+str(t))
                SaveModel(G, 'otm_G_iter_'+str(t))
            
            if t%configs.test_every==0 and t>configs.test_every:
                test_loader_iterator = Test(t,G,test_loader_iterator)

            ##########################################################
            ## Generate transported samples for inception and FID
            ##########################################################    
            if t%configs.test_inception_every==0:
                images = []
                for _ in tqdm(range(configs.num_inception_imgs // configs.batchSize + 1)):
                    x = np.random.normal(0,1, (configs.batchSize, latent_dim))
                    x = torch.FloatTensor(x).cuda()
                    Gx = G(x)
                    Gx = Gx.mul(0.5).add(0.5)
                    Gx = Gx.cpu().detach().numpy()
                    images.append(Gx)
                images = np.vstack(images)
                images = images[:configs.num_inception_imgs] # Using 50000 sample to compute IS.
                images = images.transpose(0,2,3,1)

                np.save(inception_path+'fake_samples_iter_'+str(t)+'.npy', images)
                print('Generated samples saved successfully for evaluation!!!')
            

    ##########################################################
    ## Save training error
    ##########################################################
    plt.rcParams.update({'font.size': 30})
    
    plt.figure(figsize=(20,10))
    plt.plot(psi_l,'r', label='Sup')
    plt.plot(G_l,'g', label='Inf')
    plt.legend()
    
    plt.savefig(output_path+'otm_training_full.pdf', bbox_inches='tight')


    # save after clipping error for visualization purpose
    psi_l = np.asarray(psi_l)
    G_l = np.asarray(G_l)
    psi_l[psi_l>1e+4] = 1e+4
    G_l[G_l>1e+4] = 1e+4

    psi_l[psi_l<-1e+4] = -1e+4
    G_l[G_l<-1e+4] = -1e+4

    plt.figure(figsize=(20,10))
    plt.plot(psi_l,'r', label='Sup')
    plt.plot(G_l,'g', label='Inf')
    plt.legend()

    plt.savefig(output_path+'otm_training_clip.pdf', bbox_inches='tight')

    ##########################################################
    ## Generate transported samples for evaluation
    ##########################################################
    ## Compute Transport plan
    n = 256
    try:
        y, _ = next(test_loader_iterator)
    except StopIteration:
        test_loader_iterator = iter(test_loader)
        y, _ = next(test_loader_iterator)

    y = y.to(device)
    x = np.random.normal(0,1, (n, latent_dim))
    x = torch.FloatTensor(x).to(device)
           
    Gx = G(x)
    
    Gx = Gx.mul(0.5).add(0.5)
    save_image(Gx.view(n, channels, size, size), output_path+'fake_sample_extnd_iter_'+str(epoch)+'.pdf', nrow=16, normalize=True)
    print('Extended fake sample saved successfully!!!')

    # for visualization
    y = y[:64]
    y = y.mul(0.5).add(0.5)
    save_image(y.view(64, channels, size, size), output_path+'real_sample_extnd_iter_'+str(epoch)+'.pdf', nrow=8, normalize=True)
    print('Extended fake sample saved successfully!!!')


##########################################################
## Inference Engine
##########################################################
def RunInference(iteration,test_loader_iterator=test_loader_iterator):
    G = Generator(configs)
    if parallel:
        G = nn.DataParallel(G, devices)
    G.to(device)
    G = LoadModel(G, 'otm_G_iter_'+str(iteration))
    G.eval() 

    ## Compute Transport plan for evaluation
    n = 400
    try:
        y, _ = next(test_loader_iterator)
    except StopIteration:
        test_loader_iterator = iter(test_loader)
        y, _ = next(test_loader_iterator)

    y = y.to(device)
    x = np.random.normal(0,1, (n, latent_dim))
    x = torch.FloatTensor(x).to(device)
           
    Gx = G(x)
    
    Gx = Gx.mul(0.5).add(0.5)
    save_image(Gx.view(n, channels, size, size), output_path+'fake_sample_extnd_iter_'+str(iteration)+'.pdf', nrow=20, normalize=True)
    print('Extended fake sample saved successfully!!!')

    # for visualization
    y = y[:64]
    y = y.mul(0.5).add(0.5)
    save_image(y.view(y.shape[0], channels, size, size), output_path+'real_sample_extnd_iter_'+str(iteration)+'.pdf', nrow=8, normalize=True)
    print('Extended fake sample saved successfully!!!')

    ##########################################################
    ## Generate transported samples for inception and FID
    ##########################################################    
    if not os.path.isfile(inception_path+'fake_samples_iter_'+str(iteration)+'.npy'):
        images = []
        for _ in tqdm(range(configs.num_inception_imgs // configs.batchSize + 1)):
            x = np.random.normal(0,1, (configs.batchSize, latent_dim))
            x = torch.FloatTensor(x).cuda()
            Gx = G(x)
            Gx = Gx.mul(0.5).add(0.5)
            Gx = Gx.cpu().detach().numpy()
            images.append(Gx)
        images = np.vstack(images)
        images = images[:configs.num_inception_imgs] # Using 50000 sample to compute IS.
        images = images.transpose(0,2,3,1)

        np.save(inception_path+'fake_samples_iter_'+str(iteration)+'.npy', images)
        print('Generated samples saved successfully for evaluation!!!')

if inference:
    print('Running inference engine...')
    RunInference(iteration=init_iter)

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

##########################################################
from src import distributions
from src.tools_wo_crop import get_statistics_of_dataloader, get_generated_inception_stats
from src.fid_score import calculate_frechet_distance
from src.plotters import plot_noise_interp_unequal, plot_inv_noise_interp_unequal

import gc

##########################################################
## Hyper-parameters
##########################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEED = 9999
torch.manual_seed(SEED)
path = '../input/'
output_path = './output/MNIST/'
pretrain_path = './pretrained/MNIST/'
inception_path = './Eval/utils/output/MNIST/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

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
parser.add_argument('--total_epoch', dest= 'total_epoch', type=int, default=101)
parser.add_argument('--restore', dest= 'restore', type=int, default=0)
parser.add_argument('--inference', dest= 'inference', type=int, default=0)
args = parser.parse_args()

train_model = args.train
init_iter = args.init_iter # initial iteration
T = args.total_epoch # total epoch
restore_model = args.restore # Restart training from init_epoch checkpoint
inference = args.inference # Run inference engine

##########################################################
latent_dim = 3*8*8 # Input dimension
size = 32 # Size of each image, [size,size]
channels = 3 # Number of channels, [channels,size,size]

num_workers = 10 # Used in data loader
devices = [0]


## Training parameters
BATCH_SIZE = 64 # Batch size
K_G = 2
K_psi = 1
lam_go = 10

lr_G = 2e-4
lr_psi = 2e-4

beta1D = 0.0
beta1G = 0.0
beta2D = 0.9
beta2G = 0.9


parallel = True # Data parallelization, if multiple gpus are available
save_model = True # saves weights of G and psi if true
save_every = 1000 # save weights of G and psi
log_every = 100 # print on console
test_every = 1000 # save transport samples
test_inception_every = 1000 # compute FID stats
# test_inception_every = 1

test_inception = True # compute FID stats if true
num_inception_imgs = 50000 # number of images used to compute FID

##########################################################
## Prepare Data
##########################################################
transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize((0.5, 0.5, 0.5 ), (0.5, 0.5, 0.5)),
        ])

train_data = datasets.MNIST(root=path, train=True,
                                   download=True, transform=transform)
print('Train data: ', len(train_data))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, drop_last = True)

Y_sampler = iter(train_loader)
X_sampler = distributions.StandartNormalSampler(latent_dim, requires_grad=True)

##########################################################
## Compute inception stats of real data 
##########################################################
MNIST_INC_STATS_PATH = './Eval/utils/output/MNIST/mnist_inception_stats.json' # or None to recompute stats from scratch

if not os.path.isfile(MNIST_INC_STATS_PATH):    
    dataset_float = datasets.MNIST(root=path, train=True, download=True,
     transform=transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1))
        ]))

    loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, drop_last = True)

    mu, sigma = get_statistics_of_dataloader(loader, cuda = True)
    
    with open(MNIST_INC_STATS_PATH, 'w') as f:
        json.dump({"mu": mu.tolist(), "sigma": sigma.tolist()}, f)
        
    del dataset_float, loader
    torch.cuda.empty_cache()
    gc.collect()
    
else:
    with open(MNIST_INC_STATS_PATH, 'r') as f:
        stats = json.load(f)
    mu = np.array(stats['mu'])
    sigma = np.array(stats['sigma'])
    
    del stats
    gc.collect()

##########################################################
## Main Modules
##########################################################
from blocks import ResidualBlock, UpsampleConv
from normalization import InstanceNorm2dPlus


class TransportMap(torch.nn.Module):
    def __init__(self, latent_dim=latent_dim, out_channels=3, features=256):
        super().__init__()        
        self.begin_conv = nn.Sequential(
            nn.Conv2d(in_channels=latent_dim, out_channels=features, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=features, out_channels=features*2, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(features*2, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.up1 = nn.Sequential(
            UpsampleConv(features*2,features*4),
            nn.BatchNorm2d(features*4, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.up2 = nn.Sequential(
            UpsampleConv(features*4,features*4),
            nn.BatchNorm2d(features*4, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.up3 = nn.Sequential(
            UpsampleConv(features*4,features*2),
            nn.BatchNorm2d(features*2, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.end_conv = nn.Conv2d(in_channels=features*2, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            
        self.output = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0),x.size(1),1,1)
        x = self.begin_conv(x)
        x = self.trans_conv(x)

        up1 = self.up1(x)
        up2 = self.up2(up1)
        up3 = self.up3(up2)

        op = self.end_conv(up3) 
        op = self.output(op)

        return op


print('='*64)
print('G Architecture: \n')
G = nn.Sequential(
    TransportMap(features=256),
    )
G = G.to(device)

summary(G,(latent_dim,))
print('='*64)

# sys.exit()

##########################################################
""" ResNet_D from NCSN """
class ResNet_D(torch.nn.Module):
    def __init__(self, in_channels = channels, out_channels=1, features=256):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.norm = InstanceNorm2dPlus

        self.begin_conv = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3,stride=1,padding=1)

        self.down1 = nn.Sequential(
            ResidualBlock(features, features*2, resample='down', act=self.act, normalization=self.norm)
            )
        self.down2 = nn.Sequential(
            ResidualBlock(features*2, features*2, resample='down', act=self.act, normalization=self.norm)
            )
        self.down3 = nn.Sequential(
            ResidualBlock(features*2, features, resample='down', act=self.act, normalization=self.norm)
            )
        

        self.end_conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=4, stride=1, padding=0)

    
    def forward(self, x):
        x = self.begin_conv(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        op = self.end_conv(x)
        return op


print('='*64)
print('Psi Architecture: \n')
psi = nn.Sequential(
    ResNet_D(features=128),
    )
psi = psi.to(device)
summary(psi,(channels,size,size))
print('='*64)

# sys.exit()

##########################################################
# Embeddings
Q = lambda x: F.interpolate(x.reshape(-1, 3, 8, 8), size, mode='bicubic').detach() 
INV_TRANSFORM = lambda x: 0.5*x + 0.5

##########################################################
def Loss(psi, G, Q, x, y):
    G_x = G(x)
    dot = torch.mean(Q(x)*G_x, dim=(1,2,3)).unsqueeze(dim=1)
    loss = ( dot - psi(G_x) + psi(y)).mean()
    # print(loss.item())
    
    # sys.exit()
    return loss

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
def SaveModel(model, name='OTM', path=pretrain_path):
    torch.save(model.state_dict(), path+name+'.ckpt')
    print('Model saved at '+path+name+'.ckpt')

def LoadModel(model, name='OTM', path=pretrain_path):
    model.load_state_dict(torch.load(path+name+'.ckpt'))
    print('Model loaded from '+path+name+'.ckpt')
    return model

##########################################################
def Test(iteration, G, Y_sampler, save_real=True):
    x = X_sampler.sample(BATCH_SIZE)

    try:
        y, _ = next(Y_sampler)
    except StopIteration:
        Y_sampler = iter(train_loader)
        y, _ = next(Y_sampler)

    x = x.to(device)
    y = y.to(device)

    G_x = G(x)
    
    G_x = INV_TRANSFORM(G_x)
    y = INV_TRANSFORM(y)

    # for visualization
    G_x = G_x[:64]
    y = y[:64]
    
    save_image(G_x.view(G_x.shape[0], channels, size, size), output_path+'fake_sample_iter_'+str(iteration)+'.png', nrow=8, normalize=True)
    if save_real:
        save_image(y.view(y.shape[0], channels, size, size), output_path+'real_sample_iter_'+str(iteration)+'.png', nrow=8, normalize=True)

 
##########################################################
## Training Unit
##########################################################
if train_model:
    print('Start training...')
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
        G = LoadModel(G, 'otm_g_it_'+str(init_iter))
        psi = LoadModel(psi, 'otm_psi_it_'+str(init_iter))

    ##########################################################
    # Define placeholders 
    ##########################################################
    n = 20
             
    X_fixed = X_sampler.sample(n)
    Y_fixed,_ = next(Y_sampler)
    Y_fixed = Y_fixed[:n].to(device)

    G_opt = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(beta1G, beta2G))
    psi_opt = torch.optim.Adam(psi.parameters(), lr=lr_psi, betas=(beta1D, beta2D))

    FID_history = []
    psi_l = []
    G_l = []

   # Start training
    print('Run training engine...')
    init_epoch_start = init_iter*BATCH_SIZE//len(train_data)
    
    for epoch in tqdm(range(init_epoch_start+1,T)):
        
        for i in range(len(train_loader)):
            # counts number of batches
            t = epoch*len(train_loader) + i 

            ##########################################################
            ## Outer minimization loop
            ##########################################################       
            ## Fix G and update psi to compute infimum.
            for param in psi.parameters():
                param.requires_grad = True
            for param in G.parameters():
                param.requires_grad = False

            psi_l_ = []

            for k_psi in range(K_psi):
                ## Sample data
                X = X_sampler.sample(BATCH_SIZE)

                try:
                    Y,_ =  next(Y_sampler)
                except StopIteration:
                    Y_sampler = iter(train_loader)
                    Y,_ =  next(Y_sampler)
                
                X = X.to(device)
                Y = Y.to(device)
                
                ## Compute losses
                psi_loss = Loss(psi, G, Q, X, Y)
                psi_l_.append(psi_loss.item())
                go_loss = GradientOptimality(psi, G, Q, X)
                psi_loss = psi_loss + lam_go * go_loss
                psi_opt.zero_grad(); psi_loss.backward(retain_graph=True); psi_opt.step()

            psi_l.append(np.asarray(psi_l_).mean())    
            

            ##########################################################
            ## Inner maximization loop
            ##########################################################
            ## Fix Psi and update G to compute convex conjugate/ supremum 
            for param in psi.parameters():
                param.requires_grad = False
            for param in G.parameters():
                param.requires_grad = True

            G_l_ = []

            for k_G in range(K_G):
                ## Sample data
                X = X_sampler.sample(BATCH_SIZE)

                try:
                    Y,_ =  next(Y_sampler)
                except StopIteration:
                    Y_sampler = iter(train_loader)
                    Y,_ =  next(Y_sampler)
                
                X = X.to(device)
                Y = Y.to(device)
            
                ## Compute losses
                G_loss = -Loss(psi, G, Q, X, Y)
                G_l_.append(G_loss.item())                
                G_opt.zero_grad(); G_loss.backward(retain_graph=True); G_opt.step()
            
            G_l.append(np.asarray(G_l_).mean())       

        
            if t%log_every==0:
                print("Epoch: ", epoch,"|",T, "\t psi loss: ", np.round(psi_l[-1],3), "\t GO loss: ", np.round(go_loss.item(),3), "\t G loss: ", np.round(G_l[-1],2)) 
                
                fig, axes = plot_noise_interp_unequal(G, Q, X_fixed, Y_fixed, INV_TRANSFORM, show=False)
                fig.savefig(output_path+'OTM_samples.pdf', bbox_inches='tight')
                plt.close(fig)

                fig, axes = plot_inv_noise_interp_unequal(G, psi, Q, X_fixed, Y_fixed, INV_TRANSFORM, show=False)
                fig.savefig(output_path+'OTM_inv_samples.pdf', bbox_inches='tight')
                plt.close(fig)

                if t<=test_every:
                    Test(t, G, Y_sampler)
                
            if t%save_every==0 and save_model:
                SaveModel(psi, 'otm_psi_it_'+str(t))
                SaveModel(G, 'otm_g_it_'+str(t))
            
            if t%test_every==0 and t>test_every:
                Test(t, G, Y_sampler, save_real=False)

            ##########################################################
            ## Generate transported samples for inception and FID
            ##########################################################    
            if t%test_inception_every==0:
                if test_inception:
                    print("It is high time to compute FID score.")
                    if FID_history and test_inception:
                        print('Current FID:', FID_history[-1])
                        print('Best FID: ', np.array(FID_history).min())
        
                    mu_gen, sigma_gen = get_generated_inception_stats(
                        lambda X: G(X).clamp(-1, 1), X_sampler,
                        INV_TRANSFORM, size = num_inception_imgs, batch_size=BATCH_SIZE
                    )
                    FID_history.append(calculate_frechet_distance(mu, sigma, mu_gen, sigma_gen))

                else:
                    images = []
                    for b in tqdm(range((num_inception_imgs//BATCH_SIZE)+1)):
                        X = X_sampler.sample(BATCH_SIZE)
                        G_X = INV_TRANSFORM(G(X))
                        G_X = G_X.cpu().detach().numpy()

                        images.append(G_X)
                    images = np.vstack(images)
                    images = images[:num_inception_imgs] # Using 50000 sample to compute IS.
                    images = images.transpose(0,2,3,1)

                    np.save(inception_path+'fake_samples_iter_'+str(t)+'.npy', images)
                    print('Generated samples saved successfully for evaluation!!!')
                       

    ##########################################################
    ## Save training error
    ##########################################################
    plt.rcParams.update({'font.size': 25})
    
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


    # save FID score
    fig, ax = plt.subplots(1,1,figsize=(10,5))

    ax.set_xlabel('iteration (x100)', fontsize=25)
    ax.set_ylabel('FID Score', fontsize=25)

    ax.plot(FID_history, color='green', marker='o',
             linestyle='dashed', linewidth=2, markersize=6)
    ax.grid()
    # ax.text(60,100,'Best FID: '+str(np.array(FID_history).min().round(2)))

    fig.tight_layout()
    plt.savefig(output_path+'otm_best_fid_score.pdf', bbox_inches='tight')


##########################################################
## Inference Engine
##########################################################
def RunInference(iteration, test_inception=False, Y_sampler = Y_sampler):
    G = nn.Sequential(
        TransportMap(latent_dim, size, nfilter=128, nfilter_max=512, res_ratio=0.1),
        )

    if parallel:
        G = nn.DataParallel(G, devices)
    G.to(device)
    G = LoadModel(G, 'otm_g_it_'+str(epoch))
    G.eval() 

    ## Compute Transport plan for evaluation
    n = 100

    X = X_sampler.sample(n)
    G_X = INV_TRANSFORM(G(X))
    G_X = G_X.cpu().detach().numpy()

    save_image(G_x.view(G_x.shape[0], channels, size, size), output_path+'fake_sample_extnd_iter_'+str(iteraton)+'.png', nrow=10, normalize=True)
    print('Extended fake sample saved successfully!!!')

    ##########################################################
    ## Generate transported samples for inception and FID
    ##########################################################    
    if test_inception:
        print("It is high time to compute FID score.")
        mu_gen, sigma_gen = get_generated_inception_stats(
            lambda X: G(X).clamp(-1, 1), X_sampler,
            INV_TRANSFORM, size = num_inception_imgs, batch_size=BATCH_SIZE
        )
        current_FID = calculate_frechet_distance(mu, sigma, mu_gen, sigma_gen)
        print('Current FID: ', current_FID)

    else:
        images = []
        for b in tqdm(range((num_inception_imgs//BATCH_SIZE)+1)):
            X = X_sampler.sample(BATCH_SIZE)
            G_X = INV_TRANSFORM(G(X))
            G_X = G_X.cpu().detach().numpy()

            images.append(G_X)
        images = np.vstack(images)
        images = images[:num_inception_imgs] # Using 50000 sample to compute IS.
        images = images.transpose(0,2,3,1)

        if not os.path.isfile(inception_path+'fake_samples_iter_'+str(iteration)+'.npy'):
            np.save(inception_path+'fake_samples_iter_'+str(iteration)+'.npy', images)
            print('Generated samples saved successfully for evaluation!!!')
       
if inference:
    print('Running inference engine...')
    RunInference(iteration=init_iter)

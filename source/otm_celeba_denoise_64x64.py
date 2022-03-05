##########################################################
## Standard Imports
##########################################################
import matplotlib.pyplot as plt
import numpy as np
import os, sys, random
from tqdm import tqdm
import argparse
import cv2



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
from src.plotters import plot_noise_interp_unequal, plot_inv_noise_interp_unequal



##########################################################
## Hyper-parameters
##########################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

SEED = 9999
torch.manual_seed(SEED)
path = '../../../Data/CelebA/archive/img_align_celeba/img_align_celeba/'
output_path = './output/CelebA_denoise_64x64/'
pretrain_path = './pretrained/CelebA_denoise_64x64/'
inception_path = './Eval/utils/output/CelebA_denoise_64x64/'

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
parser.add_argument('--total_epoch', dest= 'total_epoch', type=int, default=301)
parser.add_argument('--restore', dest= 'restore', type=int, default=0)
parser.add_argument('--inference', dest= 'inference', type=int, default=0)

args = parser.parse_args()

train_model = args.train
T = args.total_epoch # Total epochs
init_iter = args.init_iter # initial iter
restore_model = args.restore # Restart training from init_iter checkpoint
inference = args.inference # Run inference engine

##########################################################
size = 64 # Size of each image, [size,size]
channels = 3 # Number of channels, [channels,size,size]

num_workers = 10 # Used in data loader
devices = [0]


## Training parameters
BATCH_SIZE = 64 # Batch size
K_G = 5
K_psi = 1
lam_go = 0

lr_G = 1e-4
lr_psi = 1e-4

beta1D = 0.0
beta1G = 0.0
beta2D = 0.9
beta2G = 0.9


parallel = True # Data parallelization, if multiple gpus are available
save_model = True # saves weights of G and psi if true
save_every = 5000 # save weights of G and psi
log_every = 100 # print on console
test_every = 1000 # save transport samples
test_inception_every = 5000 # compute FID stats
# test_inception_every = 1

num_inception_imgs = 50000 # number of images used to compute FID

sigma = 0.3 # noise standard deviation

##########################################################
## Prepare Data
##########################################################
transform = transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5 ), (0.5, 0.5, 0.5)),
        ])

train_dataA = datasets.ImageFolder(path+'trainA', transform=transform)
train_dataB = datasets.ImageFolder(path+'trainB', transform=transform)

test_data = datasets.ImageFolder(path+'test', transform=transform)
print('Train dataA: ', len(train_dataA), 'Train dataB: ', len(train_dataB), 'Test data: ', len(test_data) )


train_loaderA = torch.utils.data.DataLoader(train_dataA, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, drop_last = True)
train_loaderB = torch.utils.data.DataLoader(train_dataB, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=True, drop_last = True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=False, drop_last = True)

train_loader_iteratorA = iter(train_loaderA)
train_loader_iteratorB = iter(train_loaderB)

test_loader_iterator = iter(test_loader)

##########################################################
## Main Modules
##########################################################
def spectral_norm(layer, n_iters=1):
    return torch.nn.utils.spectral_norm(layer, n_power_iterations=n_iters)

def conv3x3(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "3x3 convolution with padding"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv

class TransportMap(torch.nn.Module):
    def __init__(self, out_channels=channels, features=256):
        super().__init__()
        self.act = nn.ReLU()

        self.ip = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        
        ##########################################################
        self.down1 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down2 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down3 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down4 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        ##########################################################
        
        self.up1 = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            conv3x3(features, features),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.ReLU()
            ])
        self.up2 = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            conv3x3(features, features),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.ReLU()
            ])
        self.up3 = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            conv3x3(features, features),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.ReLU()
            ])
        self.up4 = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            conv3x3(features, features),
            nn.BatchNorm2d(features, affine=True,  track_running_stats=False),
            nn.ReLU()
            ])

        self.op = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()           
            )

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x):
        x = self.ip(x)

        x1 = self._compute_cond_module(self.down1, x)
        x2 = self._compute_cond_module(self.down2, x1)
        x3 = self._compute_cond_module(self.down3, x2)
        x4 = self._compute_cond_module(self.down4, x3)


        y3 = self._compute_cond_module(self.up1, x4)
        y3 = y3 + x3

        y2 = self._compute_cond_module(self.up2, y3)
        y2 = y2 + x2

        y1 = self._compute_cond_module(self.up3, y2)
        y1 = y1 + x1

        y = self._compute_cond_module(self.up4, y1)
        y = y + x

        op = self.op(y)
        return op

print('='*64)
print('Ki Architecture: \n')
G = TransportMap().to(device)
summary(G,(channels,size,size))
print('='*64)

# sys.exit()

##########################################################
class Psi(torch.nn.Module):
    def __init__(self, in_channels = channels, out_channels=1, features=256):
        super().__init__()
        
        ##########################################################
        self.down1 = nn.ModuleList([
            conv3x3(in_channels, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down2 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down3 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])
        self.down4 = nn.ModuleList([
            conv3x3(features, features),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            ])

        self.op = nn.Linear(in_features=features*4*4, out_features=1)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x


    def forward(self, x):
        x = self._compute_cond_module(self.down1, x)
        x = self._compute_cond_module(self.down2, x)
        x = self._compute_cond_module(self.down3, x)
        x = self._compute_cond_module(self.down4, x)

        x = x.view(x.shape[0],-1)
        op = self.op(x)
        return op

print('='*64)
print('Psi Architecture: \n')
psi = Psi().to(device)
summary(psi,(channels,size,size))
print('='*64)

# sys.exit()

###########################################################
# Embeddings
Q = lambda x: x.detach() 
INV_TRANSFORM = lambda x: 0.5*x + 0.5

##########################################################
def Loss(psi, G, Q, x, y):
    G_x = G(x)
    dot = torch.mean(Q(x)*G_x, dim=(1,2,3)).unsqueeze(dim=1)
    loss = (dot - psi(G_x) + psi(y)).mean()
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
def Degrade(y):
    noise = sigma * torch.randn(y.shape[0],y.shape[1],y.shape[2],y.shape[3])
    noise = noise.to(device)
    x = y + noise
    return x

###########################################################
def SaveModel(model, name='OTM', path=pretrain_path):
    torch.save(model.state_dict(), path+name+'.ckpt')
    print('Model saved at '+path+name+'.ckpt')

def LoadModel(model, name='OTM', path=pretrain_path):
    model.load_state_dict(torch.load(path+name+'.ckpt'))
    print('Model loaded from '+path+name+'.ckpt')
    return model

##########################################################
def Test(iteration, G, test_loader_iterator):
    ## Compute Transport plan for evaluation
    try:
        y, _ = next(test_loader_iterator)
    except StopIteration:
        test_loader_iterator = iter(test_loader)
        y, _ = next(test_loader_iterator)

    y = y.to(device)
    x = y.clone().detach()
    x = Degrade(x)
    Gx = G(x)

    x = INV_TRANSFORM(x)[:64]
    Gx = INV_TRANSFORM(Gx)[:64]
    y = INV_TRANSFORM(y)[:64]

    save_image(Gx.view(Gx.shape[0], channels, size, size), output_path+'fake_sample_iteration_'+str(iteration)+'.png', nrow=8, normalize=True)
    save_image(x.view(x.shape[0], channels, size, size), output_path+'inpaint_sample_iteration_'+str(iteration)+'.png', nrow=8, normalize=True)
    save_image(y.view(y.shape[0], channels, size, size), output_path+'real_sample_iteration_'+str(iteration)+'.png', nrow=8, normalize=True)

    return test_loader_iterator


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
             
    Y_fixed,_ = next(train_loader_iteratorB)
    Y_fixed = Y_fixed[:n].to(device)
    X_fixed = Degrade(Y_fixed.clone().detach())

    G_opt = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(beta1G, beta2G))
    psi_opt = torch.optim.Adam(psi.parameters(), lr=lr_psi, betas=(beta1D, beta2D))

    FID_history = []
    psi_l = []
    G_l = []
    go_l = []

   # Start training
    print('Running training engine...')
    init_epoch_start = init_iter*BATCH_SIZE//len(train_dataB)
    
    for epoch in tqdm(range(init_epoch_start+1,T)):
        psi_l_ = []
        go_l_ = []

        for i in range(len(train_loaderA)):
            # counts number of batches
            t = epoch*len(train_loaderA) + i
            
            ##########################################################
            ## Outer minimization loop
            ##########################################################       
            ## Fix G and update psi to compute infimum.
            for param in psi.parameters():
                param.requires_grad = True
            for param in G.parameters():
                param.requires_grad = False

            psi_l_ = []
            go_l

            for k_psi in range(K_psi):
                ## unpaired data
                try:
                    X, _ = next(train_loader_iteratorA)
                except StopIteration:
                    train_loader_iteratorA = iter(train_loaderA)
                    X, _ = next(train_loader_iteratorA)
                
                try:
                    Y, _ = next(train_loader_iteratorB)
                except StopIteration:
                    train_loader_iteratorB = iter(train_loaderB)
                    Y, _ = next(train_loader_iteratorB)

                X = Degrade(X.to(device))                
                Y = Y.to(device)   
                
                ## Compute losses
                psi_loss = Loss(psi, G, Q, X, Y)
                psi_l_.append(psi_loss.item())
                go_loss = GradientOptimality(psi, G, Q, X)
                go_l_.append(go_loss.item())
                psi_loss = psi_loss + lam_go * go_loss
                psi_opt.zero_grad(); psi_loss.backward(retain_graph=True); psi_opt.step()

            psi_l.append(np.asarray(psi_l_).mean())    
            go_l.append(np.asarray(go_l_).mean())    
            
            ##########################################################
            ## Inner maximization loop
            ##########################################################
            ## Fix Psi and update G to compute supremum.
            for param in psi.parameters():
                param.requires_grad = False
            for param in G.parameters():
                param.requires_grad = True

            G_l_ = []

            for k_G in range(K_G):
                ## unpaired data
                try:
                    X, _ = next(train_loader_iteratorA)
                except StopIteration:
                    train_loader_iteratorA = iter(train_loaderA)
                    X, _ = next(train_loader_iteratorA)
                
                try:
                    Y, _ = next(train_loader_iteratorB)
                except StopIteration:
                    train_loader_iteratorB = iter(train_loaderB)
                    Y, _ = next(train_loader_iteratorB)

                X = Degrade(X.to(device))
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
                    train_loader_iteratorB = Test(t, G, train_loader_iteratorB)
                
            if t%save_every==0 and save_model:
                SaveModel(psi, 'otm_psi_it_'+str(t))
                SaveModel(G, 'otm_g_it_'+str(t))
            
            if t%test_every==0 and t>test_every:
                test_loader_iterator = Test(t, G, test_loader_iterator)

            ##########################################################
            ## Generate transported samples for inception and FID
            ##########################################################    
            if t%test_inception_every==0:
                images = []
                for _ in tqdm(range((num_inception_imgs//BATCH_SIZE)+1)):
                    ## unpaired data
                    try:
                        X, _ = next(test_loader_iterator)
                    except StopIteration:
                        test_loader_iterator = iter(test_loader)
                        X, _ = next(test_loader_iterator)
                      
                    X = Degrade(X.to(device))
                    G_X = INV_TRANSFORM(G(X))
                    G_X = G_X.cpu().detach().numpy()

                    images.append(G_X)
                images = np.vstack(images)
                images = images[:num_inception_imgs] # Using 50000 sample to compute IS/FID.
                images = images.transpose(0,2,3,1)

                np.save(inception_path+'fake_samples_iter_'+str(t)+'.npy', images)
                print('Generated samples saved successfully for evaluation!!!')
                

    ##########################################################
    ## Save training error
    ##########################################################
    plt.rcParams.update({'font.size': 25})
    
    plt.figure(figsize=(20,10))
    plt.plot(psi_l,'r', label='Inf')
    plt.plot(G_l,'g', label='Sup')
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
    plt.plot(psi_l,'r', label='Inf')
    plt.plot(G_l,'g', label='Sup')
    plt.legend()

    plt.savefig(output_path+'otm_training_clip.pdf', bbox_inches='tight')

    ##########################################################
    ## Generate transported samples for testing
    ##########################################################
    ## Compute Transport plan
    n = BATCH_SIZE
    try:
        Y, _ = next(test_loader_iterator)
    except StopIteration:
        test_loader_iterator = iter(test_loader)
        Y, _ = next(test_loader_iterator)
 
    X = Degrade(Y.to(device))
    Y = Y.to(device)

    G_X = G(X)

    X = INV_TRANSFORM(X)[:64]
    G_X = INV_TRANSFORM(G_X)[:64]
    Y = INV_TRANSFORM(Y)[:64]

    save_image(G_X.view(n, channels, size, size), output_path+'fake_sample_extnd_epoch_'+str(epoch)+'.pdf', nrow=8, normalize=True)
    print('Extended fake sample saved successfully!!!')

    save_image(Y.view(64, channels, size, size), output_path+'inpaint_sample_extnd_epoch_'+str(epoch)+'.pdf', nrow=8, normalize=True)
    print('Extended inpainting sample saved successfully!!!')

    save_image(X.view(64, channels, size, size), output_path+'real_sample_extnd_epoch_'+str(epoch)+'.pdf', nrow=8, normalize=True)
    print('Extended real sample saved successfully!!!')


##########################################################
## Inference Engine
##########################################################
def RunInference(iteration, test_loader_iterator=test_loader_iterator, test_loader = test_loader):
    G = TransportMap().to(device)
    if parallel:
        G = nn.DataParallel(G, devices)
    G = LoadModel(G, 'otm_g_it_'+str(iteration))
    G.eval() 

    images = []
    for _ in tqdm(range((num_inception_imgs//BATCH_SIZE)+1)):
        ## unpaired data
        try:
            X, _ = next(test_loader_iterator)
        except StopIteration:
            test_loader_iterator = iter(test_loader)
            X, _ = next(test_loader_iterator)
          
        X = Degrade(X.to(device))
        G_X = INV_TRANSFORM(G(X))
        G_X = G_X.cpu().detach().numpy()

        images.append(G_X)
    images = np.vstack(images)
    images = images[:num_inception_imgs] # Using 50000 sample to compute IS/FID.
    images = images.transpose(0,2,3,1)

    if not os.path.isfile(inception_path+'fake_samples_iter_'+str(iteration)+'.npy'):
        np.save(inception_path+'fake_samples_iter_'+str(iteration)+'.npy', images)
        print('Generated samples saved successfully for evaluation!!!')
       
if inference:
    print('Running inference engine...')
    RunInference(iteration=init_iter)
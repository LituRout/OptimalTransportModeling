import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd


def plot_rgb_cloud(cloud, ax):
    colors = np.clip(cloud, 0, 1)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=colors)
    ax.set_xlabel('Red'); ax.set_ylabel('Green'); ax.set_zlabel('Blue');

def plot_combined_generated(Dec, Z, X, D, inv_transform=None, show=True):
    Dec.train(False)
    
    if inv_transform is None:
        inv_transform = lambda x: x
    
    fig, axes = plt.subplots(3, len(Z), figsize=(2 * len(Z), 6))
    
    Dec_Z = inv_transform(Dec(Z).permute(0, 2, 3, 1).cpu().detach().numpy()).clip(0,1)
    axes.flatten()[0].set_ylabel('$Dec(x)$', fontsize=25)
    for i in range(len(Z)):
        axes.flatten()[i].imshow(Dec_Z[i], cmap='gray')
    
    D.train(False)
    D_Z = D(Z.requires_grad_(True))
    D.train(True)

    Dec_D_Z = inv_transform(D_Z.permute(0, 2, 3, 1).cpu().detach().numpy()).clip(0,1)
    axes[1, 0].set_ylabel(r'$\hat{G}(x)$', fontsize=25)
    for i in range(len(Z)):
        axes[1, i].imshow(Dec_D_Z[i], cmap='gray')
    
    Real_X = inv_transform(X.permute(0, 2, 3, 1).cpu().detach().numpy()).clip(0,1)
    axes[2, 0].set_ylabel(r'$Y\sim \nu$', fontsize=25)
    for i in range(len(Z)):
        axes[2, i].imshow(Real_X[i], cmap='gray')
     
    fig.tight_layout(h_pad=0.01, w_pad=0.01)

    if show:
        plt.show()
        
    return fig, axes

def plot_latent_pca(Z, E, D, pca, n_pairs=3, show=True):
    assert n_pairs > 1
    D.train(False);
    pca_Z = pca.transform(Z.cpu().detach().numpy().reshape(len(Z), -1))
    pca_E = pca.transform(E.cpu().detach().numpy().reshape(len(E), -1))
    D_Z = D(Z).cpu().detach().numpy().reshape(len(Z), -1)
    pca_D_Z = pca.transform(D_Z)
    
    fig, axes = plt.subplots(n_pairs, 3, figsize=(12, 4 * n_pairs), sharex=True, sharey=True)
    
    for n in range(n_pairs):
        axes[n, 0].set_ylabel(f'Component {2*n+1}')
        axes[n, 0].set_xlabel(f'Component {2*n}')
        axes[n, 0].set_title(f'Initial Z', fontsize=25)
        axes[n, 1].set_title('Transported Z', fontsize=25)
        axes[n, 2].set_title('Latent Space', fontsize=25)

        axes[n, 0].scatter(pca_Z[:, 2*n], pca_Z[:, 2*n+1], color='b', alpha=0.5)
        axes[n, 1].scatter(pca_D_Z[:, 2*n], pca_D_Z[:, 2*n+1], color='r', alpha=0.5)
        axes[n, 2].scatter(pca_E[:, 2*n], pca_E[:, 2*n+1], color='g', alpha=0.5)
        
    fig.tight_layout()
    D.train(True)
    
    if show:
        plt.show()
    return fig, axes

def plot_noise_interp_unequal(G, Q,  X, Y, inv_transform=None, show=True, n = 12, test=False):

    if inv_transform is None:
        inv_transform = lambda x: x
    
    if test:
        fig, axes = plt.subplots(3, n, figsize=(6 * n, 6))
    else:
        fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    
    Q_X = Q(X)
    Q_X_interp = inv_transform(Q_X.permute(0, 2, 3, 1).cpu().detach().numpy()).clip(0,1)
    
    axes.flatten()[0].set_ylabel('$Q(X)$', fontsize=25)
    for i in range(n):
        axes.flatten()[i].imshow(Q_X_interp[i], cmap='gray')
    
    G.train(False)
    G_X = G(X)
    G.train(True)

    G_X_push = inv_transform(G_X.permute(0, 2, 3, 1).cpu().detach().numpy()).clip(0,1)
    axes[1, 0].set_ylabel(r'$\hat{G}(X)$', fontsize=25)
    for i in range(n):
        axes[1, i].imshow(G_X_push[i], cmap='gray')
    
    Real_Y = inv_transform(Y.permute(0, 2, 3, 1).cpu().detach().numpy()).clip(0,1)
    axes[2, 0].set_ylabel(r'$Y \sim \nu$', fontsize=25)
    for i in range(n):
        axes[2, i].imshow(Real_Y[i], cmap='gray')
     
    for i, ax in enumerate(axes.flatten()):
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    
    fig.tight_layout(h_pad=0.01, w_pad=0.01)

    if show:
        plt.show()
        
    return fig, axes

def plot_inv_noise_interp_unequal(G, psi, Q,  X, Y, inv_transform=None, show=True, n = 12):
    
    if inv_transform is None:
        inv_transform = lambda x: x
    
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
        
    Q_X = Q(X)
    Q_X_interp = inv_transform(Q_X.permute(0, 2, 3, 1).cpu().detach().numpy()).clip(0,1)
    axes.flatten()[0].set_ylabel('$Q(X)$', fontsize=25)
    for i in range(n):
        axes.flatten()[i].imshow(Q_X_interp[i], cmap='gray')


    ## Pushforward using conjugate transform
    psi.train(False)
    G.train(False)

    G_X = G(X)
        
    batch_size = Y.shape[0]
    eta = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    eta = torch.bernoulli(eta)
    eta = eta.expand(batch_size, Y.size(1), Y.size(2), Y.size(3))
    eta = eta.cuda()

    interpolated = eta * Y + (1 - eta) * G_X

    interpolated.requires_grad_(True)
    psi_interpolated = psi(interpolated)

    gradients = autograd.grad(
        outputs=psi_interpolated, inputs=interpolated,
        grad_outputs=torch.ones(psi_interpolated.size()).to(interpolated),
        create_graph=True, retain_graph=True
    )[0]

    psi.train(True)
    G.train(True)

    grad_psi_push = inv_transform(gradients.permute(0, 2, 3, 1).cpu().detach().numpy()).clip(0,1)
    axes[1, 0].set_ylabel(r'$\nabla{\hat{\psi}}(Y)$', fontsize=25)
    for i in range(n):
        axes[1, i].imshow(grad_psi_push[i], cmap='gray')

    Real_Y = inv_transform(Y.permute(0, 2, 3, 1).cpu().detach().numpy()).clip(0,1)
    axes[2, 0].set_ylabel(r'$Y \sim \nu$', fontsize=25)
    for i in range(n):
        axes[2, i].imshow(Real_Y[i], cmap='gray')

    for i, ax in enumerate(axes.flatten()):
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])


    fig.tight_layout(h_pad=0.01, w_pad=0.01)

    if show:
        plt.show()
        
    return fig, axes

def plot_low_dim_equal(G, Q, X, Y, show=True):
    
    G.train(False)
    G_X = G(X).cpu().detach().numpy()
    G.train(True)
    
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    
    plt.rcParams.update({'font.size': 30})
    fig, axes = plt.subplots(1,1,figsize=(10,10))

    axes.plot(X[:,0], X[:,1],'og',label=r'$X \sim \mu$', color='forestgreen')
    axes.plot(G_X[:,0],G_X[:,1],'d',label=r'$\hat{G}(X)$', color='blue')
    axes.plot(Y[:,0], Y[:,1],'s', label=r'$Y \sim \nu$', color='peru')

    for i in range(X.shape[0]):
        plt.arrow(X[i,0], X[i,1], G_X[i,0] - X[i,0], G_X[i,1] - X[i,1],  alpha=0.2, head_width=0.03, head_length = 0.05, fc='y', ec = 'y', color='y')

    axes.grid()

    # plt.legend(loc='upper left')
    
    fig.tight_layout()
        
    if show:
        plt.show()
    
    return fig, axes  

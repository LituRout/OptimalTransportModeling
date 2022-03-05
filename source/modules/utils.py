import torch
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F


class ConfigMapper(object):
    def __init__(self, args):
        for key in args:
            self.__dict__[key] = args[key]


def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off


def weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1) and (classname.find('Cond') == -1) and (classname.find('Spectral') == -1):
        try:
            # Normal conv layer
            m.weight.data.normal_(0.0, 0.02)
        except:
            # Conv layer with spectral norm
            m.weight_u.data.normal_(0.0, 0.02)
            m.weight_v.data.normal_(0.0, 0.02)
            m.weight_bar.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1 and classname.find('cond') == -1 and classname.find('Cond') == -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1) and (classname.find('Cond') == -1) and (classname.find('Spectral') == -1):
        try:
            # Normal conv layer
            nn.init.xavier_uniform_(m.weight)
        except:
            # Conv layer with spectral norm
            nn.init.xavier_uniform_(m.weight_u)
            nn.init.xavier_uniform_(m.weight_v)
            nn.init.xavier_uniform_(m.weight_bar)

    elif classname.find('BatchNorm') != -1 and classname.find('cond') == -1 and classname.find('Cond') == -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Sampling functions
def sample_normal(batch_size, emb_size, device='cuda:0'):
    return torch.randn(batch_size, emb_size, device=device)


def sample_cats(batch_size, num_classes, device='cuda:0'):
    labels = torch.randint(0, num_classes, (batch_size, )).to(device)
    return labels


def form_onehot(labels, num_classes, device='cuda:0'):
    batch_size = labels.size(0)
    y = torch.FloatTensor(batch_size, num_classes).fill_(0).to(device)

    for i in range(batch_size):
        y[i][labels[i]] = 1
    return y


class ema(object):
  def __init__(self, source, decay=0.99, start_itr=0):
    self.source = source # Our base network
    self.decay = decay
    # Optional parameter indicating what iteration to start the decay at
    self.start_itr = start_itr
    # Initialize target's params to be source's
    self.source_dict = self.source.state_dict()
    self.target_dict = copy.deepcopy(self.source_dict)
    print('Initializing EMA parameters to be source parameters...')
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.source_dict[key].data)

  def update(self, itr=None):
    # If an iteration counter is provided and itr is less than the start itr,
    # peg the ema weights to the underlying weights.
    if itr and itr < self.start_itr:
      decay = 0.0
    else:
      decay = self.decay
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                     + self.source_dict[key].data * (1 - decay))


class LinearLR(_LRScheduler):
    def __init__(self, optimizer, num_steps, last_epoch=-1):
        self.num_steps = num_steps
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - (float(self.last_epoch) / self.num_steps))
                for base_lr in self.base_lrs]


# Transformation function for adding noise
class UniformNoising(object):
    """
    Add uniform noise to input images
    """
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        noise = torch.FloatTensor(tensor.shape).uniform_(self.min, self.max)
        tensor = tensor + noise
        return tensor

    def __repr__(self):
        return self.__class__.__name__
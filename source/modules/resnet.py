import torch
import torch.nn as nn
import modules.utils
from modules.layers import *
from modules.base import *


class ResBlockGenerator(nn.Module):
    def __init__(self, config, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        conv_layer = cond_conv_layers[config.G_conv]
        norm_layer = cond_norm_layers[config.G_normalization]
        activation_layer = cond_activation_layers[config.G_activation]

        self.model = nn.Sequential(
            norm_layer(in_channels, config.num_classes),
            activation_layer(True),
            CondUpsample(scale_factor=2),
            conv_layer(in_channels, out_channels, 3, 1, padding=1),
            norm_layer(out_channels, config.num_classes),
            activation_layer(True),
            conv_layer(out_channels, out_channels, 3, 1, padding=1)
        )
        utils.weights_init_xavier(self.model)
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = CondUpsample(scale_factor=2)

    def forward(self, x):
        lab = x[1]
        model_out, _ = self.model(x)
        bypass_out, _ = self.model(x)
        return (model_out + bypass_out, lab)


class ResBlockDiscriminator(nn.Module):
    def __init__(self, config, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        conv_layer = conv_layers[config.D_conv]
        activation_layer = activation_layers[config.D_activation]

        if stride == 1:
            self.model = nn.Sequential(
                activation_layer(True),
                conv_layer(in_channels, out_channels, 3, 1, padding=1),
                activation_layer(True),
                conv_layer(out_channels, out_channels, 3, 1, padding=1)
            )
        else:
            self.model = nn.Sequential(
                activation_layer(True),
                conv_layer(in_channels, out_channels, 3, 1, padding=1),
                activation_layer(True),
                conv_layer(out_channels, out_channels, 3, 1, padding=1),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        utils.weights_init_xavier(self.model)

        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Sequential(
                conv_layer(in_channels,out_channels, 1, 1, padding=0),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            utils.weights_init_xavier(self.bypass)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):
    def __init__(self, config, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        conv_layer = conv_layers[config.D_conv]
        activation_layer = activation_layers[config.D_activation]

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            conv_layer(in_channels, out_channels, 3, 1, padding=1),
            activation_layer(True),
            conv_layer(out_channels, out_channels, 3, 1, padding=1),
            nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            conv_layer(in_channels, out_channels, 1, 1, padding=0),
        )
        utils.weights_init_xavier(self.model)
        utils.weights_init_xavier(self.bypass)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.z_dim = config.nz
        self.ngpu = config.ngpu
        ngf = self.ngf = config.ngf
        self.config = config

        if config.conditional and (config.conditioning == 'concat' or config.conditioning == 'acgan'):
            inp_dim = self.z_dim + config.num_classes
        else:
            inp_dim = self.z_dim

        conv_layer = cond_conv_layers[config.G_conv]
        norm_layer = cond_norm_layers['BN']
        activation_layer = cond_activation_layers[config.G_activation]
        lin_layer = cond_linear_layers[config.G_linear]

        self.init_size = int(config.imageSize / (2 ** 3))
        self.dense = lin_layer(inp_dim, self.init_size * self.init_size * ngf)

        self.network = nn.Sequential(
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            ResBlockGenerator(config, ngf, ngf, stride=2),
            norm_layer(ngf, config.num_classes),
            activation_layer(True),
            conv_layer(ngf, config.nc, 3, stride=1, padding=1),
            CondTanh()
        )
        utils.weights_init_xavier(self.network)

    def forward(self, input_noise, label=None):
        input_noise = input_noise.view(input_noise.shape[0],-1)

        if self.config.conditional and (self.config.conditioning == 'concat' or self.config.conditioning == 'acgan'):
            assert label is not None
            label_onehot = utils.form_onehot(label, self.config.num_classes, device=input_noise.device)
            input_noise = torch.cat((input_noise, label_onehot), dim=1)
        layer1_out, _ = self.dense((input_noise, label))
        layer1_out = layer1_out.view(-1, self.ngf, self.init_size, self.init_size)
        output = self.network((layer1_out, label))
        out, _ = output
        return out


class Discriminator(BaseDiscriminator):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)

        self.config = config
        ndf = config.ndf
        
        linear_layer = linear_layers[config.D_linear]

        out_dim = config.projection_dim
        if not config.conditional:
            out_dim = 1

        self.feat_net = nn.Sequential(
            FirstResBlockDiscriminator(config, config.nc, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf, stride=2),
            ResBlockDiscriminator(config, ndf, ndf),
            ResBlockDiscriminator(config, ndf, ndf),
            nn.ReLU(),
        )

        self.final_layer = nn.Sequential(
            linear_layer(ndf, out_dim)
        )
        self.final_layer.apply(utils.weights_init_xavier)

    def forward(self, input, label=None):
        feat = self.feat_net(input)
        feat = torch.sum(feat, (2, 3))
        feat = feat.view(feat.size(0), -1)
        disc_logits = self.final_layer(feat)

        if self.config.conditional:
            disc_logits = self.project(disc_logits, label)

        return disc_logits



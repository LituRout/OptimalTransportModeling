import torch
import torch.nn as nn
from modules.layers import *
import modules.utils


class BaseDiscriminator(nn.Module):
    def __init__(self, config):
        super(BaseDiscriminator, self).__init__()

        self.config = config
        linear_layer = linear_layers[config.D_linear]

        if self.config.conditioning == 'projection':
            self.projection = linear_layer(config.num_classes, config.projection_dim)
        elif self.config.conditioning == 'concat':
            self.projection = linear_layer(config.num_classes + config.projection_dim, 1)
        elif self.config.conditioning == 'acgan':
            self.classifier = linear_layer(config.projection_dim, config.num_classes)
            self.discriminator = linear_layer(config.projection_dim, 1)

    def project(self, input, label):
        label = utils.form_onehot(label, self.config.num_classes, device=input.device)
        if self.config.conditioning == 'projection':
            projection = self.projection(label)
            dot_product = projection * input
            out = torch.sum(dot_product, dim=1)
        elif self.config.conditioning == 'concat':
            inp_cat = torch.cat((input, label), dim=1)
            out = self.projection(inp_cat)
        elif self.config.conditioning == 'acgan':
            disc_logits = self.discriminator(input)
            class_logits = self.classifier(input)
            out = (disc_logits, class_logits)
        return out

import torch
import torch.nn as nn

from model import layers


class FDT(nn.Module):

    def __init__(self, moudle_type, order, Kt, Ks, blocks, batch_size, n_feature, T, n_vertex, gated_act_func, graph_conv_type, chebconv_matrix, drop_rate, device):
        super(FDT, self).__init__()
        modules = []
        Ko = T - 2 * (Kt - 1)
        self.Ko = Ko
        self.order = order
        self.device = device
        self.n_feature = n_feature
        self.moudle_type = moudle_type # full, no-near, and no-future 
        
        self.encode = layers.Encode(device, moudle_type, Kt, Ks, order, n_feature, T, n_vertex, blocks[0][0], blocks[1], gated_act_func, graph_conv_type, chebconv_matrix, drop_rate)
        self.decode = layers.Decode(moudle_type, Ko, blocks[1][-1], blocks[2], blocks[3][0], n_vertex, gated_act_func, drop_rate)


    def forward(self, x):
        x_stbs = self.encode(x)

        x_out = self.decode(x_stbs)
        return x_out

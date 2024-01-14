"""
Author: Eckert ZHANG
Date: 2021-12-02 17:25:58
LastEditTime: 2021-12-03 19:59:59
LastEditors: Eckert ZHANG
Description: 
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class AttentionNet(nn.Module):
    """
    Simple attention net for UV feature selection
    """
    def __init__(
        self,
        D_in=512,
        D_hidden=256,
        re_enc=True,
        # param_drop=0.1,
    ):

        super().__init__()

        self.D_in = D_in
        self.D_hidden = D_hidden
        self.re_enc = re_enc
        # self.param_drop = param_drop

        # self.embedding = nn.Embedding(self.D_in, self.D_hidden)
        if self.re_enc:
            lin = nn.Linear(self.D_in, self.D_hidden)
            nn.init.constant_(lin.bias, 0.0)
            nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")
            self.enc = nn.ModuleList([lin, nn.ReLU(inplace=True)])
            d_attn_in = self.D_hidden * 2
        else:
            d_attn_in = self.D_in * 2
        self.attn = nn.Linear(d_attn_in, self.D_hidden)  #bias=False?
        self.v = nn.Linear(self.D_hidden, 1, bias=False)
        # self.dropout = nn.Dropout(self.param_drop)

        # initialize
        nn.init.constant_(self.attn.bias, 0.0)
        nn.init.kaiming_normal_(self.attn.weight, a=0, mode="fan_in")
        nn.init.kaiming_normal_(self.v.weight, a=0, mode="fan_in")

    def forward(self, Q, Ks):
        B, L, N = Ks.shape

        # in: Q = [B, L, 1], Ks = [B, L, N], e.g., L=512, N=3*3
        # out: Q = [B, N, Lh], Ks = [B, N, Lh]
        if self.re_enc:
            # encoding the input features
            Q = Q.transpose(1, 2).reshape(-1, L)
            Ks = Ks.transpose(1, 2).reshape(-1, L)
            for layer in self.enc:
                Q = layer(Q)
                Ks = layer(Ks)
            Q = Q.reshape(B, 1, -1).repeat(1, N, 1)
            Ks = Ks.reshape(B, N, -1)
        else:
            Q = Q.transpose(1, 2).repeat(1, N, 1)
            Ks = Ks.transpose(1, 2)

        energy = torch.tanh(self.attn(torch.cat([Q, Ks], dim=2)))
        attention = self.v(energy).squeeze(2)
        weights = F.softmax(attention, dim=1)

        return weights

    # @classmethod
    # def from_conf(cls, conf, d_in, **kwargs):
    #     # PyHocon construction
    #     return cls(
    #         d_in,
    #         conf.get_list("dims"),
    #         skip_in=conf.get_list("skip_in"),
    #         beta=conf.get_float("beta", 0.0),
    #         dim_excludes_skip=conf.get_bool("dim_excludes_skip", False),
    #         combine_layer=conf.get_int("combine_layer", 1000),
    #         combine_type=conf.get_string("combine_type",
    #                                      "average"),  # average | max
    #         **kwargs)

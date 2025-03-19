#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:17:13 2023

@author: yuanbeiming
"""

import torch

import torch.nn as nn

import torch.nn.functional as F
from Blocks_clip import *

import resnet18 as mm

from Valen import wq as backbone




class Cross_Transformer_layer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([
                
                Cross_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                
                # Cross_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                # PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                
                
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, q_1, kv):
        
        # print(q.shape)
        for c_attn, ff, attn, ff_1 in self.layers:

            kv = attn(kv, name = 'yuanbeiming', name_didi = 'chendiancheng') + kv
            kv = ff_1(kv) + kv
            
            q_1 = c_attn(q_1, kv) + q_1
            q_1 = ff(q_1) + q_1

        return q_1


class markov_mixed_gussan(nn.Module):
    def __init__(self, words, dim, depth_block, depth,  heads,
                              dim_head, mlp_dim, dropout=0.1):
        super().__init__()

        assert depth >= 2, 'depth error'

        self.low_dim = dim

        self.encoder = nn.ModuleList([])

        self.num_cls = [1] + (depth - 1) *[1]
        
        self.init_mu = nn.Parameter(torch.randn(1,1,self.low_dim))
        
        self.std_scle = torch.tanh

        for i in range(depth):

            self.encoder.append(graph_transformer(words=words + self.num_cls[i], dim=self.low_dim, depth=depth_block, heads=heads,
                              dim_head=dim_head, mlp_dim=mlp_dim, num_cls = 2, dropout=dropout),)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * self.std_scle(std)



    def forward(self, x):

        b, _, d = x.shape

        sample = torch.randn(b, 1, d).to(x.device) + self.init_mu
        for layer in self.encoder:

            mu_logvar = layer( torch.cat([x, sample], dim = 1))[:,:2]
            # print(mu_logvar.shape)
            sample = self.reparameterize(*mu_logvar.chunk(2, dim = 1))

        return mu_logvar



class glove_mixed_gussan(nn.Module):
    def __init__(self, num_patterns, num_gussan, words, dim, depth,  heads,
                              dim_head, mlp_dim, dropout=0.1):
        super().__init__()

        assert depth >= 2, 'depth error'

        self.low_dim = dim

        self.encoder = nn.ModuleList([])

        self.num_cls = [1] + (depth - 1) *[1]
        
        self.init_mu = nn.Parameter(torch.randn(1,1,self.low_dim))
        
        self.std_scle = torch.tanh
        
        # self.distribute = nn.Parameter(torch.randn(1, num_gussan, self.low_dim))

        

        self.encoder= nn.Sequential(graph_transformer(words=num_patterns + words + 1, dim=self.low_dim, depth=depth, heads=heads,
                          dim_head=dim_head, mlp_dim=mlp_dim, num_cls = num_gussan, dropout=dropout, PositionalEncoding = False),
                                    take_cls(num_gussan + 1, keepdim = True))
        
        self.to_mu_logvar = nn.Linear(self.low_dim, self.low_dim*2)
        
        self.to_prob = nn.Linear(self.low_dim, num_gussan)
        
        self.num_gussan = num_gussan

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * self.std_scle(std)



    def forward(self,  x):

        b, _, d = x.shape

        sample = torch.randn(b, 1, d).to(x.device) + self.init_mu
        
        x = torch.cat([sample,  x], dim = 1)
        
        mu_logvar_prob = self.encoder(x)
        
        
        mu_logvar, prob = mu_logvar_prob.split([self.num_gussan, 1], dim = 1)
        
        
        mu_logvar = self.to_mu_logvar(mu_logvar)#b, 5, 2d
        
        prob = self.to_prob(prob)  #b, 5, 1
        
        prob = F.gumbel_softmax(prob, hard=True, dim = -1).permute(0,2,1).contiguous()
        
        # print(prob)
        
        mu_logvar = (mu_logvar*prob).sum(dim = 1)

        return mu_logvar.reshape(b, 2, d)



class wq(nn.Module):
    def __init__(self, *args):
        super(wq,self).__init__()

        self.name = 'Clip_pgm_plus_NA'

        size = 512
        patch = 64
        
        self.temperature = 0.01

        

        resnet18 = mm.ResNet18()
        
        num_depth = 6
        
        num_head = 8
        
        
        _dropout = .1

        self.discrim = backbone()

        self.num_rule = self.discrim.num_rule

        self.low_dim = self.discrim.low_dim


        self.discrim.load_state_dict(torch.load('./model/model_Clip_pgm_plus_NA_best.pt', map_location = 'cpu'))

        self.generate_times = 5



        self.generate = nn.Sequential(Rearrange('b n s d -> (b s) n d'),
                        glove_mixed_gussan( num_patterns = self.discrim.num_rule, num_gussan = 9, words = 6, dim = self.low_dim, depth = 3*3,  heads = 4,
                          dim_head = int(self.low_dim/4), mlp_dim = self.low_dim, dropout=0.1),
                        
                        Reparameterize_times(self.generate_times),
                        
  
                        
                        )






    def forward(self, x):

        
        b, n, h, w = x.shape

        x = x.view(b*n, 1, h, w)

        x = self.discrim.resnet18(X).reshape(b*n, self.low_dim, -1).permute(0, 2, 1).contiguous()

        hmap = x.shape[1]

        x = self.discrim.ff(x).reshape(b, n, hmap, self.low_dim)

        #x_statement = x.permute(0,2,1,3).reshape(b*hmap, n, self.low_dim)

        with torch.no_grad():

            x = self.graph_clip(x).reshape(b, -1, hmap, self.num_rule, self.low_dim).permute(0, 3, 2, 1, 4).contiguous()

            patterns = x[:,:,:,0]


        generate_feature = self.generate(torch.cat([x[:,-6:].detach(), patterns], dim = 1)).reshape(b, hmap, self.generate_times, self.low_dim).permute(0,2,1,3).contiguous()

        x = self.graph_clip(torch.cat([generate_feature, x], dim = 1)).reshape(b, -1, hmap, self.num_rule, self.low_dim).permute(0, 2, 3, 1, 4).contiguous()

        qkv = x.reshape(b*hmap*self.num_rule, -1, self.low_dim)

        out = self.forward_cross_attention(qkv)

        out = out.reshape(b, hmap, self.num_rule, -1)

        return out.mean(1,2)




    def loss_function(self,x):

        idx = torch.ones(x.shape[0]).to(x.device)
        
        loss = F.cross_entropy(x/1,idx*(7+self.generate_times))
                
        return loss


    def loss_function_G(self, out):


        
        x_generate, x = out.split([self.generate_times, 8], dim = 1)
        

        
        
        x = torch.cat([x_generate[:,:,None], x[:,None,:-1].expand(-1,self.generate_times, -1)]).reshape(-1, 8)




        
        idx = torch.ones(x.shape[0]).to(x.device)
        
       
        loss_2 = F.cross_entropy(x/1,idx)
        
        return loss_2
    
    
    
    def loss_function_D(self, x):
        
        # idx = None

        
        idx = torch.ones(x.shape[0]).to(x.device)
        
       
        loss_2 = F.cross_entropy(x/1,idx*(7+self.generate_times))
        
        return loss_2


    
    def stop_generate_grad(self):
        
        for _, q in self.generate.named_parameters():
            
            q.requires_grad = False
            
            
    def stop_discrim_grad(self):
        
        for _, q in self.discrim.named_parameters():
            
            q.requires_grad = False
            
    def open_all_grad(self):
        
        for _, q in self.named_parameters():
            
            q.requires_grad = True



    @torch.no_grad()
    def choose_accuracy(self, x):




        right = x[:,[-1]]
        
        left = x[:,[-2]]
        
        
        
        return (right > left).float().sum()
        
        
        
                    
                    
            
            
            
            
            
            


        
        
        
        
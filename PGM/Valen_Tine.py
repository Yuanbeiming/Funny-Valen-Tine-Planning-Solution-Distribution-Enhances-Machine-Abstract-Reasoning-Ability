#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:26:31 2021

@author: yuanbeiming
"""
import torch
import torch.nn as nn
import numpy as np


import torch.nn.functional as F

from Blocks_clip import *


from einops.layers.torch import Rearrange
from einops import rearrange
from Valen import raven_clip as backbone


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
big = False
dropout = False
temperature = 1e-6


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
    
class PGM_positional_embedding(nn.Module):
    def __init__(self, num_patterns, dim):
        super().__init__()



        

        self.embedding = nn.Parameter(torch.randn(1,num_patterns + 8,dim))
        
        self.num_patterns = num_patterns



    def forward(self, x):

        b, _, d = x.shape

        embedding = self.embedding
        
        embedding = embedding + embedding[:,[0, 3, 6, 1 ,4, 7, 2, 5] + [i for i in range(8, 8 + self.num_patterns)]]#
        
        #0 1 2
        #3 4 5
        #6 7

        return x + embedding




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
    
class Reparameterize_times(nn.Module):
    def __init__(self, times):
        super().__init__()

        self.times = times
        
        # self.std = 

    def forward(self, mu_logvar):
        
        mu, logvar = mu_logvar.chunk(2, dim = 1)

        b, n, d = mu.shape
        
        assert n == 1

        std = torch.exp(0.5 * logvar)
        eps = torch.randn(b, self.times, d).to(mu.device)
       

        return mu + eps * std

class raven_clip(nn.Module):
    def __init__(self, *args):
        super(raven_clip,self).__init__()

        self.name = 'against_doctor_exII_plus_patterns'

        size = 80
        patch = 20
        
        
        self.temperature = 1
        self.discrim = backbone()
        
        self.discrim.opposite = False
        

        self.discrim.load_state_dict(torch.load('./model_Clip_pgm_perfect_cross_cov_doctor_exII_1200000_neutral_now.pt', map_location = 'cpu'))

        #params = torch.load('./model_lico_net_regression_ex_single_view_1200000_neutral_now.pt', map_location = 'cpu')

        

        #self.discrim.load_state_dict(torch.load('./model_lico_net_regression_ex_single_view_1200000_neutral_now.pt', map_location = 'cpu'))

        
        for k, q in self.discrim.named_parameters():

        
            print(q)

            #q.data = params[k].data
        
        
        self.discrim = self.add_sn(self.discrim)
        # print(self.discrim)
        
        
        self.generate_times = 5
        self.low_dim = self.discrim.low_dim
    
        self.generate = nn.Sequential(Rearrange('b n s d -> (b s) n d'),
                                      PGM_positional_embedding(num_patterns = self.discrim.num_rule, dim = self.low_dim),
                                glove_mixed_gussan( num_patterns = self.discrim.num_rule, num_gussan = 9, words = 8, dim = self.low_dim, depth = 3*3,  heads = 4,
                                  dim_head = int(self.low_dim/4), mlp_dim = self.low_dim, dropout=0.1),
                                
                                Reparameterize_times(self.generate_times),
                                
                                Rearrange('(b s) n d -> b n s d', s = self.discrim.w)
                                
                                )
        
        
        
       
       
       
    
    
    def forward(self, x):
 
        
        b, n, h, w = x.shape

        x = x.view(b*n, 1, h, w)
 
        x = self.discrim.vit(x)
        
        x = rearrange(x, '(b n) s d -> b n s d', b= b, n = n)
        
        # print(x.shape)
        
        with torch.no_grad():
            
            patterns = self.discrim.forward_feature(x)[:-2]
            
            patterns = list(map(lambda t: t[:,:,0].permute(0,2,1,3), patterns))
            
        # print(patterns[0].shape)
        
        
        
        generate_feature = self.generate(torch.cat([x[:,:8].detach(), *patterns], dim = 1))
        
        x_generate_feature = torch.cat([ x, generate_feature], dim = 1)
        
        
        z, out = self.discrim.forward_feature(x_generate_feature)[-2:]
        
        # print(out.shape)
        #y = self.txt_clip(self.txt_data.to(x.device))

        return z, out[:,:-self.generate_times], out[:,-self.generate_times:]
        
    #"""
    
    def add_sn(self, m):
        
        
        
        
        for name, layer in m.named_children():
  
            m.add_module(name, self.add_sn(layer))# if m_sn is not None else  m.add_module(name, layer)

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                
                return nn.utils.spectral_norm(m)
            else:
                return m

    def my_cov(self, z):
        n = z.shape[0]
        d = z.shape[1]
        # z = z.reshape(-1, self.laten_code)
        z_ = z.mean(dim = 0,keepdim = True)
        C_z = (torch.matmul((z - z_)[:,:,None], (z - z_)[:,None,:]).sum(dim = 0))/(n - 1)
        c_z = ((C_z**2)*(1 - torch.eye(d).to(z.device))).sum()/d
        return c_z
    
        
    def loss_function_ce(self, x, idx):

        
        
        
        loss = F.cross_entropy(x/1,idx)

        return loss, (x[:,:8].argmax(dim = -1) == idx).float().sum()

    
    
    def loss_function_G(self, *out, **kwargs):


        
        z, x, x_generate = out
        
        idx = kwargs['idx']
        
        idx_one_hot = F.one_hot(idx, 8)
        
        anti_idx_one_hot = 1-idx_one_hot
        
        
        
        
        
        
        b = x.shape[0]
        loss_1 = self.my_cov(z)
        
        # x = x[:,None, :].expand(-1, self.generate_times, -1)
        
        # x_generate = x_generate[:,:,None]
        
        # x = torch.cat([x_generate, x], dim = -1).reshape(b*self.generate_times, -1)
        
        # loss_2, _ = self.loss_function_ce(x, torch.zeros(x.shape[0], device = x.device, dtype = torch.long))
        # print(x*anti_idx_one_hot)
        x = (x*anti_idx_one_hot)[:,None, :].expand(-1, self.generate_times, -1)
        
        # print(x_generate)
        
        
        
        x_generate = x_generate[:,:,None]*idx_one_hot[:,None,:]
        assert x.shape == x_generate.shape
        
        x = (x+x_generate).reshape(b*self.generate_times, -1)
    
        # print(x)
        loss_2, _ = self.loss_function_ce(x, idx.repeat_interleave(self.generate_times))
        
        
        return loss_1 + 100*loss_2
    
    
    
    def loss_function_D(self, *out, target_shape, target_line, idx):
        
        # idx = None
        z, x, x_generate = out
        
        x = torch.cat([x,x_generate], dim = 1)

        
        """
        loss_1, right_shape = self.loss_function_sl(x_shape, y, target = target_shape)
        
        loss_2, right_line = self.loss_function_sl(x_line, y, target = target_line)

        """
        
        loss_3, right = self.loss_function_ce(x, idx)

        loss_4 = self.my_cov(z)

        """

        if self.training:
            
            self.beta.step()

        """


        return 100*loss_3 + 1*loss_4, torch.zeros(1).to(x.device).sum(),  torch.zeros(1).to(x.device).sum(), right
    
    def stop_generate_grad(self):
        
        for _, q in self.generate.named_parameters():
            
            q.requires_grad = False
            
            
    def stop_discrim_grad(self):
        
        for _, q in self.discrim.named_parameters():
            
            q.requires_grad = False
            
    def open_all_grad(self):
        
        for _, q in self.named_parameters():
            
            q.requires_grad = True
        

        
def transpose(x):
    return x.transpose(-2, -1).contiguous()
    
def mul_dot(a, b):
    
    assert a.dim() == b.dim() and a.dim() == 3 and b.shape[1] == 7776  and a.shape[1] == 1, 'error'
    
    # a@transpose(b)
    
    # print(a.shape,b.shape, (a@transpose(b)).shape)
    return (a@transpose(b)).squeeze(-1)
    
    
def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]          
        
        
def reasoning(*args):
	return raven_clip()

 
if __name__ == '__main__':
    x = torch.randn(10,16,80,80).cuda()
    y = torch.randint(1,(10,7776,16)).long().cuda()
    target = torch.randint(7776,(10,)).long().cuda()
    label = torch.randint(8,(10,)).long().cuda()
    
    model = raven_clip().cuda()
    # params = torch.load('./model_Clip_raven_120000_distribute_nine_best.pt', map_location = 'cpu')
    # # model_dict =  model.state_dict()
    
    # # state_dict = {k:v for k,v in params.items() if k in model_dict.keys()}
    # for k,q in model.named_parameters():
    #     if k[:7] != 'tajador':
    #         print(k)
    #         q.data = params[k].data
            
    model.cuda()
    # 
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)       
    # model.load_state_dict(torch.load('./model_Clip_raven_120000_distribute_nine_best.pt', map_location = 'cpu'))
    out = model(x)
    
    l, right_shape, right_line, right = model.loss_function_D(*out, target_shape = target, target_line = target, idx = label)
    
    
    l_ = model.loss_function_G(*out, idx = label)
    
    
    # accuracy = model.choose_accuracy(*out, idx = label)
    

    
    


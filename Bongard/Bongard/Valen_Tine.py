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

class get_choise(nn.Module):
    def __init__(self):
        super(get_choise, self).__init__()
       

    def forward(self, x):
        
        b, s, n, m, d = x.shape
        
        assert m == 6

        
        return torch.stack([
                x[:,[0,1,2,3,4,5,6]],
                x[:,[0,1,2,3,4,5,7]],
                x[:,[0,1,2,3,4,5,8]],
                x[:,[0,1,2,3,4,5,9]],
                x[:,[0,1,2,3,4,5,10]],
                x[:,[0,1,2,3,4,5,11]],
                x[:,[0,1,2,3,4,5,12]],
                x[:,[0,1,2,3,4,5,13]]
                ], dim = 1)


class shuffle_sample(nn.Module):
    def __init__(self):
        super(shuffle_sample, self).__init__()
        
    def forward(self, x):
        #"""
        if self.training:
        
            s = x.shape[2]
            
            assert s == 7
            
            index = torch.randperm(s)
            
            
            return x[:,:,index]
        
        else:
        #"""
            return x


class wq(nn.Module):
    def __init__(self, *args):
        super(wq,self).__init__()

        self.name = 'Clip_pgm_plus_NA'

        size = 512
        patch = 64
        self.low_dim = 512
        self.temperature = 0.01

        self.num_rule = 7

        resnet18 = mm.ResNet18()
        
        num_depth = 6
        
        num_head = 8
        
        
        _dropout = .1

        self.ff = Clstransformer(words = 16, dim = self.low_dim, depth = num_depth, heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, num_cls = self.num_rule, dropout = 0.1)

        self.graph_clip = nn.Sequential( 
                                nn.Sequential(get_choise(), shuffle_sample(),),
                                Rearrange('b s n m d -> (b s m) n d', s = 8,  n = 7),
                                graph_transformer(words = 7, dim = self.low_dim, depth = num_depth, heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, num_cls = self.num_rule, dropout = 0.1),
                                take_cls(self.num_rule, keepdim = True),
                                
                                )

        self.tajador = nn.Sequential(Rearrange('b c d -> (b c) d', c = 1),
                            Bottleneck_judge(self.low_dim, self.low_dim),
                            )



        
        def forward(self, x):
     
            
            b, n, h, w = x.shape

            x = x.view(b*n, 1, h, w)

            x = resnet18(X).reshape(b*n, self.low_dim, -1).permute(0, 2, 1).contiguous()

            x = self.ff(x).reshape(b, n, -1, self.low_dim)

            x = self.graph_clip(x).reshape(-1, self.low_dim)

            x = self.tajador(x).reshape(b,8,-1,self.num_rule)
     

            return x.mean(-1,-2)
        

        
        
        def loss_function(self,x):

            idx = torch.zeros(x.shape[0]).to(x.device)
            
            loss = F.cross_entropy(x/1,idx)
                    
            return loss
        
        @torch.no_grad()
        def choose_accuracy(self, x):
        
        
            right = x[:,[0]]
            
            left = x[:,[-1]]
            
            
            
            return (right > left).float().sum()
        
        
        
                    
                    
            
            
            
            
            
            


        
        
        
        
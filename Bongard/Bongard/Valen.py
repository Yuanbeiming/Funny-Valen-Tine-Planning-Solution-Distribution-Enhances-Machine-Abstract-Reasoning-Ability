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
        
        b, n, s, d = x.shape
        
        assert n == 14

        index = [8, 9, 10, 11, 12, 13]
 
        for i in range(8):
            index += [i, 9, 10, 11, 12, 13]
            index += [8, i, 10, 11, 12, 13]
            index += [8, 9, i, 11, 12, 13]
            index += [8, 9, 10, i, 12, 13]
            index += [8, 9, 10, 11, i, 13]
            index += [8, 9, 10, 11, 12, i]





        return x[:, index, ].reshape(b,6,-1,s, d) 
    


class shuffle_sample(nn.Module):
    def __init__(self):
        super(shuffle_sample, self).__init__()
        
    def forward(self, x):
        #"""
        if self.training:
        
            s = x.shape[1]
            
            assert s == 6
            
            index = torch.randperm(s)
            
            
            return x[:,index]
        
        else:
        #"""
            return x


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

        self.shuffle = shuffle_sample()

        self.ff = Clstransformer(words = 16, dim = self.low_dim, depth = num_depth, heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, num_cls = self.num_rule, dropout = 0.1)

        self.graph_clip = nn.Sequential( 
                                nn.Sequential(get_choise(), shuffle_sample(),),
                                Rearrange('b s n m d -> (b n m) s d', s = 6),
                                graph_transformer(words = 6, dim = self.low_dim, depth = num_depth, heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, num_cls = self.num_rule, dropout = 0.1),
                                take_cls(self.num_rule, keepdim = True),
                                
                                )

        self.cross_graph_clip = Cross_Transformer(words = 6, dim = self.low_dim, depth = num_depth, heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, dropout = 0.1)

        self.tajador = nn.Sequential(Rearrange('b c d -> (b c) d', c = 1),
                            Bottleneck_judge(self.low_dim, self.low_dim),
                            )

    def forward_cross_attention(self,qkv):

        # print(q_1.shape)

       num_tokens =qkv.shape[1]

       q, kv = qkv.split([1,num_tokens-1], dim = 1)
       
       q = q[:,:,None].expand(-1, -1,  8, -1).permute(0,2,1,3).contiguous().reshape(-1, 1, self.low_dim)
       
       kv = kv.reshape(-1, 6, 8, self,low_dim).permute(0,2,1,3).contiguous().reshape(-1, 6, self.low_dim)
       
       kv = self.shuffle(kv)
       
       out = self.cross_graph_clip(q, kv)
       
       
       
       return self.tajador(out)




    def forward(self, x):

        
        b, n, h, w = x.shape

        x = x.view(b*n, 1, h, w)

        x = resnet18(X).reshape(b*n, self.low_dim, -1).permute(0, 2, 1).contiguous()

        hmap = x.shape[1]

        x = self.ff(x).reshape(b, n, hmap, self.low_dim)

        x = self.graph_clip(x).reshape(b, -1, hmap, self.num_rule, self.low_dim).permute(0, 2, 3, 1, 4).contiguous()

        qkv = x.reshape(b*hmap*self.num_rule, -1, self.low_dim)

        out = self.forward_cross_attention(qkv)

        out = out.reshape(b, hmap, self.num_rule, 8)


        return x.mean(1,2)




    def loss_function(self,x):

        idx = torch.ones(x.shape[0]).to(x.device)
        
        loss = F.cross_entropy(x/1,idx*7)
                
        return loss

    @torch.no_grad()
    def choose_accuracy(self, x):


        right = x[:,[-1]]
        
        left = x[:,[0]]
        
        
        
        return (right > left).float().sum()
        
        
        
                    
                    
            
            
            
            
            
            


        
        
        
        
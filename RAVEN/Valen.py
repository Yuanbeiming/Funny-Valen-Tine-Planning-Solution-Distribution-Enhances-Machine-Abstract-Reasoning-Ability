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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
big = False
dropout = False
temperature = 1e-6

class Beta(nn.Module):
    def __init__(self, alpha = 10, step_size = 0.01):
        super(Beta, self).__init__()

        self.register_buffer("beta",torch.ones(1))
        
        self.alpha = alpha
        
        self.step_size = step_size
        
        self.register_buffer("action",torch.zeros(1))
        
        step_beta = self.alpha/step_size
        
        self.step_beta = math.pi/step_beta
        
    def forward(self):
        
        
        return self.beta.item()
        


    def step(self):
        
        self.beta += ((torch.sin(self.action) >= 0 ).long()*2 - 1)*self.step_size
        # print((torch.sin(self.action) >= 0 ).long())
        self.action += self.step_beta
        """
        if self.beta.item() < self.alpha:
            self.beta += self.step_size
        """

class Recat(nn.Module):
    def __init__(self):
        super(Recat, self).__init__()


    def forward(self, x):
        b ,n ,s ,d = x.shape
        
    

        return x[:, [0,1,2,3,4,5,6,7,8,
                    6,7,9,
                    6,7,10,
                    6,7,11,
                    6,7,12,
                    6,7,13,
                    6,7,14,
                    6,7,15,
                  
                    ]].reshape(b, 10, 3, s, d)
    
class Recombine(nn.Module):
    def __init__(self):
        super(Recombine, self).__init__()
       

    def forward(self, x):
        b ,s ,m ,d = x.shape
        

        
        return x[:,:, [0,1,2,
                       0,1,3,
                       0,1,4,
                       0,1,5,
                       0,1,6,
                       0,1,7,
                       0,1,8,
                       0,1,9,
                    ]].reshape(b, s, 8, 3, d)
    
    
    
    
class print_layer(nn.Module):
    def __init__(self):
        super(print_layer, self).__init__()
       

    def forward(self, x):
        
        print(x.shape)

        
        return x


class shuffle_sample(nn.Module):
    def __init__(self):
        super(shuffle_sample, self).__init__()
        
    def forward(self, x):
        #"""
        if self.training:
        
            s = x.shape[-2]
            
            assert s == 2
            
            index = torch.randperm(s)
            
            
            return x[:,:,:,:,index]
        
        else:
        #"""
            return x
    
    
class shuffle_sample_(nn.Module):
    def __init__(self):
        super(shuffle_sample_, self).__init__()
        
    def forward(self, x):
        #"""
        if self.training:
        
            s = x.shape[-2]
            
            assert s == 2
            
            index = torch.randperm(s)
            
            
            return x[:,index]
        
        else:
        #"""
            return x


class get_choise(nn.Module):
    def __init__(self):
        super(get_choise, self).__init__()
       

    def forward(self, x):
        
        b, s, n, m, d = x.shape

        
        return torch.stack([
                x[:,:,:,[0,1]],
                x[:,:,:,[0,2]],
              
                x[:,:,:,[1,2]],

                ], dim = 3)

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
    


class Cross_Transformer(nn.Module):
    def __init__(self, words, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Cross_Transformer,self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, words, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Cross_Transformer_layer(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q_1, kv):
        b,n,d = kv.shape
        kv = kv + self.pos_embedding[:, :]
        # dropout
        kv = self.dropout(kv)

        x = self.transformer(q_1, kv)

        return x



class raven_clip(nn.Module):
    def __init__(self, *args):
        super(raven_clip,self).__init__()

        self.name = 'Clip_raven_perfect_cross_cov_doctor__II'

        size = 80
        patch = 20
        
        if big:
            num_head = 8
            num_depth = 6
            self.low_dim = 256
        else:
            num_head = 4
            num_depth = 3
            self.low_dim = 128
            
        if dropout:
            _dropout = 0.1
        else:
            _dropout = 0
        txt_data = []
        for c in range(8,14):#color
            for n in range(8,14):#number
                for p in range(8,14):#position
                    for s in range(8,14):#size
                        for t in range(8,14):#type
                            txt_data.append(np.array([ 3,  c,  4, n,  5, p,  6, s, 7, t]))
                        
        



        
        txt_data = np.array(txt_data)[:-1]
        
        assert txt_data.shape[0] == 7775
        
        txt_size = 7775
        
        
        self.dict_ = { 'EOF':0,
                      'shape':1,
                      'line':2, 
                      'color':3, 
                      'number':4, 
                      'position':5, 
                      'size':6, 
                      'type':7, 
                      'progression':8, 
                      'XOR':9, 
                      'OR':10, 
                      'AND':11, 
                      'consistent_union':12,
                      ' NA':13}
        
        self.txt_data = torch.from_numpy(txt_data[None,:,:]).long()
        
        
        assert self.txt_data.shape[1] == 7775
        
        self.txt_data.requires_grad = False
    

        self.w = int(size/patch)*int(size/patch)


        self.temperature = temperature

        self.beta = Beta(alpha = 20, step_size = 0.005)
        
        self.num_rule = 2
            
        self.vit = nn.Sequential(ViT(image_size = size, 
                                   patch_size = patch,  
                                   dim = self.low_dim,
                                   depth = num_depth, 
                                   heads = num_head, 
                                   mlp_dim = self.low_dim,
                                   channels = 1, 
                                   dim_head = int(self.low_dim/num_head), 
                                   dropout = _dropout, 
                                   emb_dropout = _dropout),
                                 Rearrange('(b n) s d -> b n s d', n = 16),
                                 Recat(),
                                 Rearrange('b m n s d -> b s m (n d)', s = self.w, n = 3, m = 10),
                                 

                                 )#b*16, 16, dim
        
        
        
        self.g_function = nn.Sequential(
            Rearrange('b s m d -> (b s m) d'),

            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), 3*self.low_dim),#10,10
            
            Bottleneck_judge(3*self.low_dim, int(self.low_dim*0.75), self.low_dim),#10,10
            
            Rearrange('(b s m) d -> b s m d', s = self.w, m = 10),
            
            # Recombine(),
            

            
        )
        


        self.graph_clip = nn.Sequential( 
                                        nn.Sequential(get_choise(), shuffle_sample(),),
                                        Rearrange('b s n c m d -> (b s n c) m d', m = 2, n = 8, c = 3),
                                        graph_transformer(words = 2, dim = self.low_dim, depth = num_depth, heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, num_cls = self.num_rule, dropout = 0.1),
                                        take_cls(self.num_rule, keepdim = True),
                                        Rearrange('(b s n c) m d -> b s n c m d', s = self.w, d = self.low_dim, m = self.num_rule, n = 8, c = 3),
                                        )

        
        self.recombine = Recombine()
        self.shuffle = shuffle_sample_()
        self.rearrange = Rearrange('b s n c d -> (b s n) c d', n = 8, s = self.w)
                                      
        
        self.cross_graph_clip = Cross_Transformer(words = 2, dim = self.low_dim, depth = num_depth, heads = num_head, dim_head = int(self.low_dim/num_head), mlp_dim = self.low_dim, dropout = 0.1)
        
        self.to_out = nn.Sequential( 
                              
                                        Rearrange('(b s n) c d -> b s n c d', s = self.w, n = 8, c = 1),
                                        
                                        Rearrange('b s n c d -> b n s c d', s = self.w, n = 8, c = 1),

                                        )


        
        self.tajador = nn.Sequential(Rearrange('b m s n d -> (b m s n) d'),
                            Bottleneck_judge(self.low_dim, self.low_dim),
                            
                            Rearrange('(b m s n) d -> b m (s n d)', s = self.w, m = 8, n = 1),
                            Mean(dim = -1))

        self.num_forward = 0
    
    def forward_cross_attention(self,qkv):
        
       # print(q_1.shape)
       
       q, kv = qkv.split([1,2], dim = 3)
       
       q = self.rearrange(q)
       
       kv = self.rearrange(kv)
       
       kv = self.shuffle(kv)
       
       out = self.cross_graph_clip(q, kv)
       
       
       
       return self.to_out(out)
       
       
       
    
    
    def forward(self, x):
 
        
        b, n, h, w = x.shape
        

        

        x = x.view(b*n, 1, h, w)
 
        x = self.vit(x)
        
        x = self.g_function(x)

        qkv = self.graph_clip(self.recombine(x))
        
        qkv = qkv.chunk(self.num_rule, dim = -2)
        
        # q_2 = q[:,int(self.w*0.5):].mean(dim = 1, keepdim = True)
        
        # out_1 = self.forward_cross_attention(qkv_1.squeeze(-2))
        
        # out_2 = self.forward_cross_attention(qkv_2.squeeze(-2))
        
        out =  map(lambda t: self.tajador(self.forward_cross_attention(t.squeeze(-2))), qkv)
        
        
        # print(out.shape)
        #y = self.txt_clip(self.txt_data.to(x.device))

        return *list(map(lambda t: t[:,:,:,0].mean(dim = (1,2)), qkv)), x.reshape(-1, self.low_dim), sum(list(out))
        
    #"""

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

        return loss, (x.argmax(dim = -1) == idx).float().sum()

    
    
    def loss_function_sl(self, *out, target):


        
        graph = out[0].mean(dim = 1)
        
        # print(graph.shape)


        txt = out[1].mean(dim  = 0 ,keepdim = True)
        
        
        target_mask = target.ne(7775)

        
        graph = graph[target_mask]
        
        target = target[target_mask]
        
        
        if target.shape[0] == 0:
            
            return 0, torch.zeros(1).to(graph.device)
        
        else:


        
            r = F.cosine_similarity(graph,txt, dim = -1)
            
            
            
    
    
            loss_1 = F.cross_entropy(r / self.temperature, target)
            
    
    
    
            return loss_1, (r.argmax(dim = -1) == target).float().sum()
    
    
    
    def loss_function(self, *out, target_shape, target_line, idx):
        
        # idx = None
        x_shape, x_line, z, x = out

        
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


        return 200*loss_3 + 1*loss_4, torch.zeros(1).to(x.device).sum(),  torch.zeros(1).to(x.device).sum(), right
    
        
        

        
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
    
    l, right_shape, right_line, right = model.loss_function(*out, target_shape = target, target_line = target, idx = label)
    
    
    # accuracy = model.choose_accuracy(*out, idx = label)
    

    
    

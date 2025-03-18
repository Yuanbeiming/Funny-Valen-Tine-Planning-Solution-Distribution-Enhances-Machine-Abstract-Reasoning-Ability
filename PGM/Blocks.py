import torch
from torch import einsum
import torch.nn as nn
from einops import rearrange, repeat
import math
import numpy as np
import torch.nn.functional as F
from einops.layers.torch import Rearrange

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


def get_attn_pad_mask(seq_q, seq_k, mask_value = 0):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(mask_value).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

def get_rpm_attn_pad_mask(seq_q, seq_k, mask_value = 13):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(mask_value).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    pad_attn_mask += torch.cat((pad_attn_mask[:,:,1:], pad_attn_mask[:,:,0:1]), dim = 2)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(conv3x3(in_channel, out_channel, stride),nn.BatchNorm2d(out_channel))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.downsample(x) + self.bn2(self.conv2(out)))

        return out


class ResBlock1x1(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample = None):
        super(ResBlock1x1, self).__init__()
        self.conv1 = conv1x1(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(conv3x3(in_channel, out_channel, stride),nn.BatchNorm2d(out_channel))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.downsample(x) + self.bn2(self.conv2(out)))

        return out



class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)






class take_cls(nn.Module):
    def __init__(self, keepdim = False):
        super(take_cls, self).__init__()
        self.keepdim = keepdim

    def forward(self, x):
        if self.keepdim == False:
            return x[:,0]
        
        else:
            return x[:,0:1]






class Mean(nn.Module):
    def __init__(self, dim, keepdim = False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim = self.keepdim)

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1, downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        


        if self.downsampling:
            residual = self.downsample(x.contiguous())
            


        out += residual
        out = self.relu(out)
        return out
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
       
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        # hidden_dim = mlp_dim
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    


# Attention
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        #dim_head: dim of signle head
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
 
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
 
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
 
 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
 
    def forward(self, x, **kwargs):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    
class Routing(nn.Module):
    def __init__(self, iterations, words, dim, heads = 8, dim_head = 32):
        #dim_head: dim of signle head
        super().__init__()

        self.iterations = iterations
        
        self.words = words
        
        inner_dim = dim_head *  heads
        project_out = inner_dim == dim
 
 
        self.heads = heads
        
        self.dim_head = dim_head
        
        
        self.W = nn.Parameter(torch.randn(1,
                                                self.words*self.heads,   #6x6x32  
                                                words,       #16
                                                inner_dim,       #256
                                                dim_head) )     #32
        
        self.bias = nn.Parameter(0.01 * torch.randn(1,
                                                self.words*self.heads,   #6x6x32  
                                                words,       #16
                                                inner_dim) )     #32
        
        self.inner = nn.Linear(dim, inner_dim) if project_out else nn.Identity()
        
        self.outer = nn.Linear(inner_dim, dim) if project_out else nn.Identity()
 
 
 
    def routing(self, u_hat):
        b = torch.zeros_like( u_hat )#同尺寸可以内积相乘 torch.Size([256, 1152, 10, 16])
        u_hat_routing = u_hat.detach()  #前两次禁止回传
        for i in range(self.iterations):#3次迭代
            c = F.softmax(b, dim=2)   #在第三维度上进行softmax,10类连接强度，初始值1/10 torch.Size([256, 1152, 10, 16]
            if i==(self.iterations-1):#最后一次迭代保存梯度
                s = (c*u_hat).sum(1, keepdim=True)  #tor, input_size = 201ch.Size([256, 1, 10, 16]
            else:
                s = (c*u_hat_routing).sum(1, keepdim=True)   #torch.Size([256, 1, 10, 16]
            v = self.squash(s)  #torch.Size([256, 1, 10, 16]，投票结果
            if i < self.iterations - 1: #并未最后一次迭代                             v                      u_hat_routing                u_hat_routing与v的内积
                b = (b + (u_hat_routing*v).sum(3, keepdim=True))#(torch.Size([256, 1, 10, 16]*torch.Size([256, 1152, 10, 16])→ torch.Size([256, 1152, 10, 1]))+torch.Size([256, 1152, 10, 16])
        return v  #torch.Size([256, 1, 10, 16])  
            
    def squash(self, s):
        s_norm = s.norm(dim=-1, keepdim=True)
        v = s_norm / (1. + s_norm**2) * s
        return v    
    def forward(self, x, **kwargs):
        b, n, d, h = *x.shape, self.heads
        
        x = self.inner(x)
        

        x = rearrange(x, 'b n (h d) -> b (n h) () d ()', h = self.heads, d = self.dim_head)
        
        
        x = self.W@x #b, n*h, n, d, 1
        
        x = x.squeeze(-1) + self.bias
        
        x = self.routing(x)
        
        
        x = x.squeeze(1)
        
        return self.outer(x)  #self.routing = Routing(iterations = iterations, words = words, dim = dim, heads = heads, dim_head = dim_head, residual = residual)
    
class Routing_Transformer_layer(nn.Module):
    def __init__(self, iterations, words, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout),
                PreNorm(dim, Routing(iterations = iterations, words = words, dim = dim, heads = heads, dim_head = dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
                )
            ]))
    def forward(self, x):
        for attn, ff_0, routing, ff_1 in self.layers:
            x = attn(x, name = 'yuanbeiming', name_didi = 'chendiancheng') + x
            x = ff_0(x) + x
            x = routing(x) + x
            x = ff_1(x) + x
        return x
    
class Routing_Transformer(nn.Module):
    def __init__(self, iterations, words, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, words, dim))
        self.transformer = Routing_Transformer_layer(iterations, words, dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b,n,d = x.shape
        x += self.pos_embedding[:, :]
        # dropout
        x = self.dropout(x)

        x = self.transformer(x)

        return x
    
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_head):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = dim_head ** -0.5

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        super(MultiHeadAttention, self).__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.fc = nn.Linear(heads * dim_head, dim, bias=False)
    def forward(self, x, attn_mask, **kwargs):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        b, n, _, h = *x.shape, self.heads
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(q, k, v, attn_mask)
        context = rearrange(context, 'b h n d -> b n (h d)') # context: [batch_size, len_q, n_heads * d_v]
       
        return self.fc(context), attn
    
    
class Mask_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, dropout = 0.):
        #dim_head: dim of signle head
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
 
 
        self.heads = heads
        self.scale = dim_head ** -0.5
 
 
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
 
 
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
 
 
    def forward(self, x, attn_mask, **kwargs):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
        
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        dots.masked_fill_(attn_mask, -1e9)
        
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)    
    
    
    

    
#%%   
# 基于PreNorm、Attention和FFN搭建Transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x, name = 'yuanbeiming', name_didi = 'chendiancheng') + x
            x = ff(x) + x
        return x
    
class Musk_PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.norm(x)

            
class Mask_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Mask_Transformer).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):#L
            self.layers.append(nn.ModuleList([
                Musk_PreNorm(dim),
                 Mask_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                 Musk_PreNorm(dim),
                 FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, attn_mask):
        for attn, ff in self.layers:
            for p_1, attn, p_2, ff in self.layers:
            
	            x = p_1(x)
	            x = attn(x, attn_mask) + x
	            
	            x = p_2(x)
	            x = ff(x) + x
        return x

class Clstransformer(nn.Module):
    def __init__(self, words, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Clstransformer,self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, words, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b,n,d = x.shape
        x += self.pos_embedding[:, :]
        # dropout
        x = self.dropout(x)

        x = self.transformer(x)

        return x
    
class graph_transformer(nn.Module):
    def __init__(self, words, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(graph_transformer,self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, words + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b,n,d = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.pos_embedding[:, :]
        # dropout
        x = self.dropout(x)

        x = self.transformer(x)

        return x
    

class txt_transformer(nn.Module):
    def __init__(self, words,  dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(txt_transformer,self).__init__()
        assert dim%2 == 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pos_embedding = torch.zeros(words + 1, dim)
        
        
        position = torch.arange(0, words + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        self.pos_embedding[:, 0::2] = torch.sin(position * div_term)
        self.pos_embedding[:, 1::2] = torch.cos(position * div_term)

        self.pos_embedding = self.pos_embedding.unsqueeze(0)
        
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        b,n,d = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        
        x += self.pos_embedding[:, :].to(x.device)
        # dropout
        x = self.dropout(x)

        x = self.transformer(x)

        return x    


class txt_mask_transformer(nn.Module):
    def __init__(self, dict_len, words,  dim, depth, heads, dim_head, mlp_dim, dropout = 0., is_pgm = False):
        super(txt_transformer,self).__init__()
        assert dim%2 == 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pos_embedding = torch.zeros(words + 1, dim)
        self.Enbedding = nn.Embedding(dict_len, dim)
        
        self.is_pgm = is_pgm
        
        position = torch.arange(0, words + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        self.pos_embedding[:, 0::2] = torch.sin(position * div_term)
        self.pos_embedding[:, 1::2] = torch.cos(position * div_term)

        self.pos_embedding = self.pos_embedding.unsqueeze(0)
        
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Mask_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq):#b,n
        x = self.Enbedding(seq)
        b,n,d = x.shape
        
        seq = torch.cat((torch.ones(b,1), seq), dim = 1)
        
        if self.is_pgm:
            attn_mask = get_rpm_attn_pad_mask(seq,seq,13)
        else:
            attn_mask = get_rpm_attn_pad_mask(seq,seq,0)
        
        
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        
            
            
        x = torch.cat((cls_tokens, x), dim=1)
        
        
        x += self.pos_embedding[:, :].to(x.device)
        # dropout
        x = self.dropout(x)

        x = self.transformer(x, attn_mask)

        return x    

    
def pair(t):
    return t if isinstance(t, tuple) else (t, t)       
        
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size,  dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        #dim: lengh of token ，depth： depth of tranformer, dim_head: dim of signle head, mlp_dim: dim of mlp of transfomer
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) or  isinstance(image_size, list) else pair(image_size)
        patch_height, patch_width = pair(patch_size)
     
 
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # patch数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch维度
        patch_dim = channels * patch_height * patch_width
        
        # 定义块嵌入
        self.name = 'ViT'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        # 定义位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # 定义类别向量

        self.dropout = nn.Dropout(emb_dropout)
 
 
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
 
 
        self.to_latent = nn.Identity()
        # 定义MLP

    # ViT前向流程
    def forward(self, img):
        # 块嵌入
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 追加位置编码
        # print(x)
        x += self.pos_embedding[:, :n]
        # dropout
        x = self.dropout(x)
        # 输入到transformer
        x = self.transformer(x)
        # x_ = x.mean(dim = 1, keepdim = True) if self.pool == 'mean' else x[:,1:(n + 1)]
        # MLP
        return x        
    
class Bottleneck_judge(nn.Module):
    def __init__(self, in_places, hidden_places, out_places = 1,  dropout = 0.1, last_dropout = 0.5):
        super(Bottleneck_judge,self).__init__()



        self.bottleneck = nn.Sequential(
            nn.Linear(in_places, hidden_places),
            nn.GELU(),
            nn.BatchNorm1d(hidden_places),
            nn.Linear(hidden_places, hidden_places),
            nn.GELU(),
            nn.BatchNorm1d(hidden_places),
            nn.Linear(hidden_places, out_places)
        )

        if in_places != out_places:
            self.downsample = nn.Sequential(
                nn.Linear(in_places, out_places)
            )
            
        else:
            self.downsample = nn.Identity()
            

    def forward(self, x):

        out = self.bottleneck(x)
        
        residual = self.downsample(x)
        
        out += residual
        
        return out
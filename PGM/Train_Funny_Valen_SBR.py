# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:46:20 2021

@author: yuanbeiming
"""



import torch
import torch.nn as nn
from tqdm import tqdm



#import Valen_SBR as model_vit
#import Funny_Valen as model_vit
import Valen as model_vit

import numpy as np
from torchvision import transforms



import torch.distributed as dist
dist.init_process_group(backend='nccl', init_method='env://')

from torch.utils.data import Dataset, DataLoader
import os

import torch.backends.cudnn as cudnn
import random

t = transforms.Resize((80,80))

import make_pgm_data as make_data


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

# same_seed(1024)


batch_size = 100


weight_decay = 0


train_set = make_data.Raven_Data(train = True,val = False)
len_train_set = len(train_set)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)

num_train = len_train_set
train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, shuffle = False, num_workers=4, prefetch_factor=4,pin_memory=True)


if dist.get_rank() == 0:
    print('number before shuffle:', num_train)
    
    print(num_train, len(train_loader))

#train_set.shuffle_set()

num_train = len(train_set)

if dist.get_rank() == 0:    
    print('number after shuffle:', num_train)
    
    
    print(num_train, len(train_loader))

val_set = make_data.Raven_Data(train = False,val = False)

num_val = len(val_set)

val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

val_loader = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, shuffle = False, num_workers=4, prefetch_factor=4,pin_memory=True)


if dist.get_rank() == 0:
    print('train:', num_train, 'test:', num_val)

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

dist.barrier() #等待每块GPU都运行到这个地方之后再继续往下走
world_size = torch.distributed.get_world_size()
if dist.get_rank() == 0:
  print('全局进程数: ',world_size)
# dist.init_process_group(backend="nccl")
device = torch.device("cuda", local_rank)


init_seeds(2048+local_rank)

model = model_vit.raven_clip()

if model_vit.big == True:
	name =  'big_' + model.name +  '_' +str(len_train_set) + '_' + make_data.aaa
	text_name =  'big_' + model.name  + make_data.aaa
else:
	name =  model.name +  '_' +str(len_train_set) + '_' + make_data.aaa
	text_name =  model.name +  make_data.aaa
print(name)

#model.load_state_dict(torch.load('./model_'+name+'_now.pt', map_location = 'cpu'))



model.load_state_dict(torch.load('/home/user/ybm/pgm/model_lico_net_regression_ex_single_view_1200000_neutral_best_.pt', map_location = 'cpu'))
# /home/yuanbeiming/python_work/vit_for_raven/vit_for_raven_92.pt
#%%

def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value
 
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
 
        return value



# /home/yuanbeiming/python_work/vit_for_raven/vit_for_raven_92.pt

#params = torch.load('./model_'+name+'_now.pt', map_location = 'cpu')

"""
params = torch.load('/home/user/ybm/pgm/model_Clip_pgm_perfect_cross_cov_doctor_exII_1200000_neutral_best_98_55.pt', map_location = 'cpu')

#params = torch.load('./model_'+name+'_now.pt', map_location = 'cpu')
for k, q in model.named_parameters():

    #if k[:3] == 'vit' or k[:len('g_function')] == 'g_function' or k[:len('graph_clip')] == 'graph_clip': 
    if k in params.keys(): 
	    q.data = params[k].data
	
	    print(k)
"""

#%%
if torch.cuda.device_count() > 1:


  model = model.to(device)
  # model = nn.DataParallel(model)
  
  # model = ...
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)#)
  #model= torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

  print( torch.cuda.device_count())


#%%

optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay = 0)#
#optimiser.load_state_dict(torch.load('./optimiser_'+name+'_now.pt'))

optimiser.load_state_dict(torch.load('/home/user/ybm/pgm/optimiser_lico_net_regression_ex_single_view_1200000_neutral_best_.pt'))

for param_group in optimiser.param_groups:
    param_group["lr"] = param_group["lr"]*0.5
    param_group["weight_decay"] = weight_decay
""""""
print('lr:', optimiser.state_dict()['param_groups'][0]['lr'])
print('w_d:', optimiser.state_dict()['param_groups'][0]['weight_decay'])

lr_decay = torch.optim.lr_scheduler.StepLR(optimiser, step_size= 1, gamma= 0.995)



#%%
max_accuracy = 0
num_epoch = 120000
epoch = 0


#%%
with open("test_on_"+text_name+".txt", "a") as f:  # 打开文件
        f.write('train_num_sample:' + str(len_train_set)+ '\n')
        f.write('temperature:' + str(model_vit.temperature)+ '\n')
        f.write('weight_decay:' + str(weight_decay)+ '\n')

print('temperature:',model_vit.temperature)
print('weight_decay:' ,(weight_decay))

while epoch < num_epoch:
    accuracy = [0]*5
    
    drop_sample = [0]*2
    
    drop_test_sample = [0]*2

    
    accuracy_val = [0]*5
    
    loss_train = 0
    # loss_test_all = 0
    with tqdm(total=len(train_loader)  + len(val_loader)) as pbar:
        model.train()  #启用 Batch Normalization 和 Dropout。
        for x_train, label,  label_in, idx in train_loader:


            # Model training
            
            
            #梯度清零

            x_train = t(x_train).float().to(device)


            idx = (idx).long().to(device)

            label = label.long().to(device)

            label_in = label_in.long().to(device)
            
            
            
            drop_sample[0] += label.eq(7775).sum()
            
            drop_sample[1] += label_in.eq(7775).sum()
            
            
            
            
            
            x_train = x_train/255.

            out_train = model(x_train) #输出

            

            loss, right, right_in, choose_right = (model.module.loss_function(*out_train, target_shape = label, target_line = label_in, idx = idx) )

            

            # choose_right, choose_right_in, choose_right_all = model.module.choose_accuracy(*out_train, idx = idx)
            model.zero_grad()
            loss.backward()#回传
            optimiser.step()  
            #dist.barrier()
            loss = reduce_value(loss, average=True)
            
            right = reduce_value(right, average = False)
            
            right_in = reduce_value(right_in, average = False)
            
            choose_right = reduce_value(choose_right, average = False)
            
            with torch.no_grad():

                accuracy[0] += right.cpu().numpy()
                accuracy[1] += right_in.cpu().numpy()
                accuracy[2] += choose_right.cpu().numpy()
                



            loss_train += loss.item()
            pbar.set_postfix(loss_batch = loss.item())#进度条
            pbar.update(1)






        lr_decay.step()
        
        if num_train - drop_sample[0] == 0:
            accuracy[0] = 0
        else:
            accuracy[0] /= (num_train - reduce_value(drop_sample[0], average = False).item())
            
            
        if num_train - drop_sample[1] == 0:
            accuracy[1] = 0
        else:
            accuracy[1] /= (num_train - reduce_value(drop_sample[1], average = False).item())
            
            
        accuracy[2] /= num_train
       

    
        loss_train /= len(train_loader)

        num_train = len(train_set)


        # num_train = len(train_set)
        
        # 测试错误率和损失函数
        
        model.eval()#不启用 Batch Normalization 和 Dropout。
        with torch.no_grad():
            
            for index, (x_test, label,  label_in, idx)in enumerate(val_loader):
                
                x_test = t(x_test).float().to(device)

                label = label.long().to(device)
                label_in = label_in.long().to(device)
                
                
                drop_test_sample[0] += label.eq(7775).sum()
                
                drop_test_sample[1] += label_in.eq(7775).sum()
                
                
                
                idx = (idx).long().to(device)
                x_test = x_test/255.
                
                out_test = model(x_test)

                _, right, right_in, choose_right = (model.module.loss_function(*out_test, target_shape = label, target_line = label_in, idx = idx) )
                

                # choose_right, choose_right_in, choose_right_all = model.module.choose_accuracy(*out_test, idx = idx)
                
                # loss = reduce_value(loss, average=True)
                #dist.barrier()
                right = reduce_value(right, average = False)
                
                right_in = reduce_value(right_in, average = False)
                
                choose_right = reduce_value(choose_right, average = False)
                
                accuracy_val[0] += right.cpu().numpy()
                accuracy_val[1] += right_in.cpu().numpy()
                accuracy_val[2] += choose_right.cpu().numpy()

                pbar.update(1)



            if num_val - drop_test_sample[0] == 0:
                accuracy_val[0] = 0
            else:
                accuracy_val[0] /= (num_val - reduce_value(drop_test_sample[0], average = False).item())
                
                
            if num_val - drop_test_sample[1] == 0:
                accuracy_val[1] = 0
            else:
                accuracy_val[1] /= (num_val - reduce_value(drop_test_sample[1], average = False).item())
           
            
            accuracy_val[2] /= num_val
            

    
    # Stores model
    if dist.get_rank() == 0:
        if accuracy_val[2] > max_accuracy: 
            torch.save(model.module.state_dict() , './model_'+name+'_best.pt')
            torch.save(optimiser.state_dict(), './optimiser_'+name+'_best.pt')
    
            max_accuracy = accuracy_val[2]
            
    
        torch.save(model.module.state_dict()  , './model_'+name+'_now.pt')
        torch.save(optimiser.state_dict(), './optimiser_'+name+'_now.pt')

    # Print and log some results
    """
    if accuracy[2] > accuracy_val[2]:
        for param_group in optimiser.param_groups:
            #param_group["lr"] = 5e-4
            param_group["weight_decay"] = 1e-4
    
    if accuracy[2] > accuracy_val[2] + 0.5:
        for param_group in optimiser.param_groups:
            #param_group["lr"] = 5e-4
            param_group["weight_decay"] = 2e-4


    if accuracy[2] > accuracy_val[2] + 0.8:
        for param_group in optimiser.param_groups:
            #param_group["lr"] = 5e-4
            param_group["weight_decay"] = 3e-4

    """

    
    
    if dist.get_rank() == 0:
        print("epoch:{}\t loss_train:{:.4f}\n accuracy_train: shape:{:.4f}\t line:{:.4f}\t  choose:{:.4f}\n  accuracy_val: shape:{:.4f}\t line:{:.4f}\t choose:{:.4f}\n learning_rate:{:.8f}\n".format(epoch, loss_train,  *accuracy[:3],
                                                                                                                                                                                         *accuracy_val[:3],lr_decay.get_lr()[0]))
                                                                                                                                                                                         
    if dist.get_rank() == 0:
        with open("test_on_"+text_name+".txt", "a") as f:  # 打开文件
            
            f.write("epoch:{}\t loss_train:{:.4f}\n accuracy_train: shape:{:.4f}\t line:{:.4f}\t  choose:{:.4f}\n  accuracy_val: shape:{:.4f}\t line:{:.4f}\t choose:{:.4f}\n learning_rate:{:.8f}\n".format(epoch, loss_train,  *accuracy[:3],
                                                                                                                                                                                             *accuracy_val[:3],lr_decay.get_lr()[0]))
    epoch += 1
    print(model.module.temperature)
    
    
    #%%











    
# from torchvision import transforms
# from matplotlib import pyplot as plt  
# t = transforms.ToPILImage()
# from einops.layers.torch import Rearrange
# arrenge = Rearrange('c h w -> h (c w)')
# num = 6
# plt.imshow(t(arrenge(x_train[num][:3])), cmap = 'gray')
# plt.show() 


# num = 6

# for i in range(8):

#     plt.subplot(3,3,i+1)
    
#     plt.imshow(t(x_train[num][i]), cmap = 'gray')
# plt.show() 
    


# for i in range(8):
    
#     plt.subplot(2,4,i+1)
    
#     plt.imshow(t(x_train[num][i+8]), cmap = 'gray')

    
    
# plt.show()

# print(y_train[num])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

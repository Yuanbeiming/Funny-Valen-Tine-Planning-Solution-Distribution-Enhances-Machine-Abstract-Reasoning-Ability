# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:46:20 2021

@author: yuanbeiming
"""



import torch
import torch.nn as nn
from tqdm import tqdm


import Valen as model_vit
import argparse




import numpy as np
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader



from bongard_data import ShapeBongard_V2 as SV
root_path='/home/user/ybm/ShapeBongard_V2_save1/coda/ShapeBongard_V2_2'






parser = argparse.ArgumentParser()

# General Arguments
parser.add_argument("-lr", help="Learning Rate",
                    type=float, default=0.001)
parser.add_argument("-batch_size", help="Batch Size",
                    type=int, default=120)
parser.add_argument("-epoch", help="Epochs to Train",
                    type=int, default=400)


parser.add_argument("-weight_decay", help="Weight Decay value",
                    type=float, default=0)

parser.add_argument("-start_epoch", help="Epoch to start and load save file",
                    type=int, default=0)

# Network Arguments
parser.add_argument("-base_model", help="Resnet model (layers)",
                    type=int, default=18)

parser.add_argument("-projection_size", help="Size of output of encoder projection",
                    type=int, default=1024)
parser.add_argument("-layer_size", help="Size of layers between feature and output",
                    type=int, default=512)

# Dataset Arguments
parser.add_argument("-num_classes", help="Modify dataset to have lesser classes, -1 is original num",
                    type=int, default=2)
parser.add_argument("-root_path", help="Choose optimizer",
                    type=str, default=root_path)

args = parser.parse_args()


HD = False

#%%

total_train = SV(args.root_path, type='total_train')
total_train_loader = DataLoader(total_train, batch_size=args.batch_size, shuffle=True, num_workers=8, prefetch_factor=4,pin_memory=True)


train_loader, num_train, data_name= total_train_loader, len(total_train), "Total "
#%%

BA_test = SV(args.root_path, type='BA_test')
BA_test_loader = DataLoader(BA_test, batch_size=args.batch_size, shuffle=True, num_workers=8, prefetch_factor=4,pin_memory=True)



val_loader, num_val,data_name_1 = BA_test_loader,len(BA_test),"BA "
#%%

FF_test = SV(args.root_path, type='FF_test')
FF_test_loader = DataLoader(FF_test, batch_size=args.batch_size, shuffle=True, num_workers=8, prefetch_factor=4,pin_memory=True)



val_loader_1,num_val_1,data_name_2 = FF_test_loader,len(FF_test),"FF "
#%%

NV_test = SV(args.root_path, type='NV_test')
NV_test_loader = DataLoader(NV_test, batch_size=args.batch_size, shuffle=True, num_workers=8, prefetch_factor=4,pin_memory=True)



val_loader_2,num_val_2,data_name_3 = NV_test_loader,len(NV_test),"NV "
#%%

CM_test = SV(args.root_path, type='CM_test')
CM_test_loader = DataLoader(CM_test, batch_size=args.batch_size, shuffle=True, num_workers=8, prefetch_factor=4,pin_memory=True)


val_loader_3,num_val_3,data_name_4 = CM_test_loader,len(CM_test),"CM "
#%%



print('train:', num_train, 'test:', num_val)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model_vit.wq()

# if model_vit.big == True:
# 	name =  'big_' + model.name +  '_' +str(len_train_set) + '_' + make_data.aaa
# 	text_name =  'big_' + model.name  + make_data.aaa
# else:
name =  model.name + 'on_' +  data_name
text_name =  model.name +  'on_' + data_name
print(name)
import os

	
#model.load_state_dict(torch.load('./model/model_'+name+'_now.pt', map_location = 'cpu'))
# /home/yuanbeiming/python_work/vit_for_raven/vit_for_raven_92.pt
#%%
if torch.cuda.device_count() > 1:



  model = nn.DataParallel(model)
  print( torch.cuda.device_count())

model = model.to(device)


#%%

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0)
#optimiser.load_state_dict(torch.load('./opt/optimiser_'+name+'_now.pt'))

for param_group in optimiser.param_groups:
    #param_group["lr"] = 7e-4
    param_group["weight_decay"] = args.weight_decay

print('lr:', optimiser.state_dict()['param_groups'][0]['lr'])
print('w_d:', optimiser.state_dict()['param_groups'][0]['weight_decay'])

lr_decay = torch.optim.lr_scheduler.StepLR(optimiser, step_size= 1, gamma= 0.999)



#%%
max_accuracy = 0
num_epoch = 120000
epoch = 0


#%%
with open("test_on_"+text_name+".txt", "a") as f:  # 打开文件
        f.write('train_num_sample:' + str(num_train) + '\n')
        f.write('temperature:' + str(model_vit.temperature)+ '\n')
        f.write('weight_decay:' + str(args.weight_decay)+ '\n')

print('temperature:',model_vit.temperature)
print('weight_decay:' ,(args.weight_decay))

while epoch < num_epoch:
    accuracy = [0]*3

    
    accuracy_val = [0]*5
    
    loss_train = 0
    
    loss_test = 0
    
    loss_test_1 = 0
    
    loss_test_2 = 0
    
    loss_test_3 = 0
    # loss_test_all = 0
    with tqdm(total=len(train_loader)  + len(val_loader) + len(val_loader_1) + len(val_loader_2) + len(val_loader_3)) as pbar:
        model.train()  #启用 Batch Normalization 和 Dropout。
        for _, x_train, _ in train_loader:


            # Model training
           
            
            #梯度清零
            model.zero_grad()

            x_train = (x_train).float().to(device)


            
            # x_train = x_train/255.

            out_train = model(x_train) #输出

            

            loss = (model.module.loss_function(out_train) if isinstance(model, nn.DataParallel) 
                    else model.loss_function(out_train))

            choose_right = model.module.choose_accuracy(out_train)
            
            loss.backward()#回传
            optimiser.step()  
            
            with torch.no_grad():

                # accuracy[0] += right.cpu().numpy()
                accuracy[1] += choose_right.cpu().numpy()



            loss_train += loss.item()
            pbar.set_postfix(loss_batch = loss.item())#进度条
            pbar.update(1)
        lr_decay.step()
        
        # accuracy[0] /= (num_train*8)
        accuracy[1] /= (num_train)

    
        loss_train /= len(train_loader)
        

        
        model.eval()#不启用 Batch Normalization 和 Dropout。
        with torch.no_grad():
            
            for index, (_, x_test, _)in enumerate(val_loader):
 
                x_test = (x_test).float().to(device)

                # x_test = x_test/255.
                
                out_test = model(x_test)

                # loss = (model.module.loss_function(out_test) if isinstance(model, nn.DataParallel) 
                #         else model.loss_function(out_test))

                choose_right = model.module.choose_accuracy(out_test)
                


                accuracy_val[0] += choose_right.cpu().numpy()
                # loss_test += loss.item()
                pbar.update(1)

                
            # accuracy_val[0] /= (num_val*8)
            accuracy_val[0] /= (num_val)
            
            
            
            loss_test /= len(val_loader)
            

            for index, (_, x_test, _)in enumerate(val_loader_1):
 
                x_test = (x_test).float().to(device)

                # x_test = x_test/255.
                
                out_test = model(x_test)

                # loss = (model.module.loss_function(out_test) if isinstance(model, nn.DataParallel) 
                #         else model.loss_function(out_test))

                choose_right = model.module.choose_accuracy(out_test)
                
                # accuracy_val[0] += right.cpu().numpy() 

                accuracy_val[1] += choose_right.cpu().numpy()
                # loss_test_1 += loss.item()
                pbar.update(1)


            accuracy_val[1] /= (num_val_1)
                
                
                
            loss_test_1 /= len(val_loader_1)
            
            for index, (_, x_test, _)in enumerate(val_loader_2):
 
                x_test = (x_test).float().to(device)

                # x_test = x_test/255.
                
                out_test = model(x_test)

                # loss = (model.module.loss_function(out_test) if isinstance(model, nn.DataParallel) 
                #         else model.loss_function(out_test))

                choose_right = model.module.choose_accuracy(out_test)
                
                # accuracy_val[0] += right.cpu().numpy() 

                accuracy_val[2] += choose_right.cpu().numpy()
                # loss_test_2 += loss.item()
                pbar.update(1)


            accuracy_val[2] /= (num_val_2)
                
                
                
            loss_test_2 /= len(val_loader_2)
            
            for index, (_, x_test, _)in enumerate(val_loader_3):
 
                x_test = (x_test).float().to(device)

                # x_test = x_test/255.
                
                out_test = model(x_test)

                # loss = (model.module.loss_function(out_test) if isinstance(model, nn.DataParallel) 
                #         else model.loss_function(out_test))

                choose_right = model.module.choose_accuracy(out_test)
                
                # accuracy_val[0] += right.cpu().numpy() 

                accuracy_val[3] += choose_right.cpu().numpy()
                # loss_test_3 += loss.item()
                pbar.update(1)


            accuracy_val[3] /= (num_val_3)
                
                
                
            loss_test_3 /= len(val_loader_3)
            
            

            
            
            

    
    # Stores model
    if accuracy_val[0] > max_accuracy: 
        torch.save(model.module.state_dict()  if isinstance(model, nn.DataParallel) else model.state_dict(), './model/model_'+name+'_best.pt')
        torch.save(optimiser.state_dict(), './opt/optimiser_'+name+'_best.pt')

        max_accuracy = accuracy_val[0]
        

    torch.save(model.module.state_dict()  if isinstance(model, nn.DataParallel) else model.state_dict(), './model/model_'+name+'_now.pt')
    torch.save(optimiser.state_dict(), './opt/optimiser_'+name+'_now.pt')

    # Print and log some results
    print(("epoch:{}\t loss_train:{:.4f}\t " + data_name + "choose_accuracy_train:{:.4f}\t  " + data_name_1 + "choose_accuracy_val:{:.4f}\t  " + data_name_2 + "choose_accuracy_val:{:.4f}\t "+ data_name_3 + "choose_accuracy_val:{:.4f}\t "+ data_name_4 + "choose_accuracy_val:{:.4f}\t learning_rate:{:.8f}").\
          format(epoch, loss_train,  accuracy[1],  accuracy_val[0],  accuracy_val[1],  accuracy_val[2],  accuracy_val[3],  lr_decay.get_lr()[0]))

    with open("test_on_"+text_name+".txt", "a") as f:  # 打开文件
        
        f.write(("epoch:{}\t loss_train:{:.4f}\t " + data_name + "choose_accuracy_train:{:.4f}\t  " + data_name_1 + "choose_accuracy_val:{:.4f}\t  " + data_name_2 + "choose_accuracy_val:{:.4f}\t "+ data_name_3 + "choose_accuracy_val:{:.4f}\t "+ data_name_4 + "choose_accuracy_val:{:.4f}\t learning_rate:{:.8f}").\
              format(epoch, loss_train,  accuracy[1],  accuracy_val[0],  accuracy_val[1],  accuracy_val[2],  accuracy_val[3],  lr_decay.get_lr()[0]))
    epoch += 1
    
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

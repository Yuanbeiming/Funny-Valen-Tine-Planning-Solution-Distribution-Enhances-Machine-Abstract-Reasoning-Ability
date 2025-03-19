# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import os
import csv
from PIL import Image
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import random
from torchvision.transforms import ToPILImage
import torch
from torch.utils.data import Dataset
from torchvision import transforms

show=ToPILImage()

class ShapeBongard_V2(Dataset):

    def __init__(self, root_path, type = 'bd_train',image_size=512, box_size=512):#
        self.bong_size = 7
        if box_size is None:
            box_size = image_size
        self.root_path = root_path
        self.image_size=image_size
        self.type = type
        self.tasks = {}  #
        self.tasks_per_prob_type = sorted(os.listdir(os.path.join(root_path, type)))
        self.tf = transforms.Compose([
                lambda x:Image.open(x).convert('RGB'), # string path=> image data
                transforms.Resize((int(self.image_size), int(self.image_size))),

                transforms.Grayscale(num_output_channels=1),
        
                transforms.ToTensor(),

            ])
        random.shuffle(self.tasks_per_prob_type)
        #print(self.tasks_per_prob_type)
        self.problems,self.images, self.labels = self.load_csv('image.csv')

       

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root_path, self.type, filename)):
            images = []
            for problem in self.tasks_per_prob_type:
                #print(problem)
                for x in os.listdir(os.path.join(self.root_path, self.type, problem)):
                #print(label)
                #./ShapeBongard_V2\\bd\\bd_open_band_three_arcs4-no_two_parts_sector2_0000\\1\\6.png'
                    images += glob.glob(os.path.join(self.root_path, self.type, problem, x, '*.png'))
            with open(os.path.join(self.root_path, self.type, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  #./ShapeBongard_V2\\bd\\bd_open_band_three_arcs4-no_two_parts_sector2_0000\\1\\6.png'
                    problem = img.split(os.sep)[-3]
                    #bd_open_band_three_arcs4-no_two_parts_sector2_0000,
                    label = img.split(os.sep)[-2]
                    # ./ShapeBongard_V2\\bd\\bd_open_band_three_arcs4-no_two_parts_sector2_0000\\1\\6.png', 1
                    writer.writerow([problem, img, label])
                print('writen into csv file:', filename)
    #
         # #read from csv file
        problems, images, labels = [], [], []
        with open(os.path.join(self.root_path, self.type, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # bd_open_band_three_arcs4-no_two_parts_sector2_0000,
                # ./ShapeBongard_V2\\bd\\bd_open_band_three_arcs4-no_two_parts_sector2_0000\\1\\6.png', 1
                problem ,img, label =  row
                problems.append(problem)
                images.append(img.replace('./ShapeBongard_V2_2',self.root_path))
                labels.append(label)
        assert  len(images) == len(labels)
        
        # print(self.images)
    #
        return problems, images, labels
    # #
    def __len__(self):  #样本的数量
        return int(len(self.images)/14)

    def __getitem__(self, idx): #返回当前idx元素的值，当前image的数据

        img = [None]*14
        label = [None]*14
        indx = int(idx*14)

        problem =  self.problems[indx].replace('\\','/')
        for i in range(14):
            img[i], label[i] = self.tf(self.images[indx+i].replace('\\','/')), torch.Tensor([int(self.labels[indx+i].replace('\\','/'))])

        return problem,  torch.cat(img, dim = 0), torch.cat(label, dim = 0)





# def main():
#     db = ShapeBongard_V2(root_path,type = 'bd_train')
#     # x, y, z = db[0]
#     # print(z.shape)
#     # print(torch.Tensor(0))
#     loader = DataLoader(db, batch_size=400, shuffle=False)
#     for step, (x, y, z) in enumerate(loader):
#         print(x[0])
#         print(y.shape)
#         print(z.shape)
#         y = y.unsqueeze(2)
# #         show(y[0,7,...]).show()
# if __name__ == '__main__':
#     main()
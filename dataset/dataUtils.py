import numpy as np
import os
import json

import torch
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)

def alignDataList(fileRoot, folder, imagesList,dataset):
    'this funciton is to make sure image list and skeleton list are aligned'
    allFiles = os.listdir(os.path.join(fileRoot, folder)) # get json files
    allFiles.sort()
    newJson_list = []
    newImage_list = []

    for i in range(0, len(imagesList)):
        if dataset == 'N-UCLA':
            json_file = imagesList[i].split('.jpg')[0] + '_keypoints.json'
        else:
            image_num = imagesList[i].split('.jpg')[0].split('_')[1]
            json_file = folder + '_rgb_0000000' + str(image_num)+'_keypoints.json'
        if json_file in allFiles:
            newJson_list.append(json_file)
            newImage_list.append(imagesList[i])

    return newJson_list, newImage_list


def getJsonData(fileRoot, folder, jsonList):
    skeleton = []
    # allFiles = os.listdir(os.path.join(fileRoot, folder))
    # allFiles.sort()
    usedID = []
    confidence = []
    mid_point_id1 = [2,3,5,6,8,9,10,12,13]
    mid_point_id2 = [3,4,6,7,1,10,11,13,14]
    for i in range(0, len(jsonList)):
        # json_file = imagesList[i].split('.jpg')[0] + '_keypoints.json'
        with open(os.path.join(fileRoot, folder, jsonList[i])) as f:
            data = json.load(f)
        # print(len(data['people']))
        if len(data['people']) != 0:
            # print('check')
            usedID.append(i)
            temp = np.asarray(data['people'][0]['pose_keypoints_2d']).reshape(25,3)
            pose = np.expand_dims(temp[:,0:2], 0)
            # midPoint = (pose[:,mid_point_id1]+pose[:,mid_point_id2])/2
            # pose = np.concatenate((pose,midPoint),1)
            s = np.array([temp[:,-1], temp[:,-1]])
            score = np.expand_dims(s.transpose(1,0), 0)
            skeleton.append(pose)
            confidence.append(score)

        else:
            continue

    skeleton = np.concatenate((skeleton))
    confidence = np.concatenate((confidence))
    return skeleton, usedID, confidence

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
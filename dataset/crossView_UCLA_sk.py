import sys
sys.path.append('../')
sys.path.append('../data')
sys.path.append('.')

import os
import numpy as np
import random
from PIL import Image, ImageFilter, ImageEnhance

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset.dataUtils import GaussianBlur, getJsonData, alignDataList
from utils import Gaussian, DrawGaussian

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

""" ground truth: 
   Hip_Center = 1;   Spine = 2;       Shoulder_Center = 3; Head = 4;           Shoulder_Left = 5;
   Elbow_Left = 6;   Wrist_Left = 7;  Hand_Left = 8;       Shoulder_Right = 9; Elbow_Right = 10;
   Wrist_Right = 11; Hand_Right = 12; Hip_Left = 13;       Knee_Left = 14;     Ankle_Left = 15;
   Foot_Left = 16;   Hip_Right = 17;  Knee_Right = 18;     Ankle_Right = 19;   Foot_Right = 20;
"""

class NUCLA_CrossView_ske(Dataset):
    """Northeastern-UCLA Dataset Skeleton Dataset, cross view experiment,
        Access input skeleton sequence, GT label
        When T=0, it returns the whole
    """
    def __init__(self, root_list, dataType, sampling, phase, T, maskType, setup, nClip):
        self.root_list = root_list
        self.data_root = '/data/N-UCLA_MA_3D/multiview_action'
        self.dataType = dataType
        self.sampling = sampling
        self.phase = phase
        self.T = T
        self.ds = 2
        self.clips = nClip
        self.maskType = maskType
        cam = '1,2,3' 
        if self.dataType == '2D':
            self.root_skeleton = '/data/N-UCLA_MA_3D/openpose_est'
        else:
            self.root_skeleton = '/data/N-UCLA_MA_3D/VideoPose3D_est/3d_est'
        self.view = []
        if setup == 'setup1':       self.test_view = 'view_3'
        elif setup == 'setup2':     self.test_view = 'view_2'
        else:                       self.test_view = 'view_1'
        for name_cam in cam.split(','): 
            if name_cam in self.test_view: continue
            self.view.append('view_' + name_cam)

        self.num_samples = 0
        self.num_action = 10
        self.action_list = {'a01': 0, 'a02': 1, 'a03': 2, 'a04': 3, 'a05': 4,
                            'a06': 5, 'a08': 6, 'a09': 7, 'a11': 8, 'a12': 9}
        self.actions = {'a01': 'pick up with one hand', 'a02': "pick up with two hands", 'a03': "drop trash",
                        'a04': "walk around", 'a05': "sit down",
                        'a06': "stand up", 'a08': "donning", 'a09': "doffing", 'a11': "throw", 'a12': "carry"}
        self.actionId = list(self.action_list.keys())
        # Get the list of files according to cam and phase
        self.test_list = []
        self.samples_list = []
        for view in self.view:
            file_list = self.root_list + view + '.list'
            list_samples = np.loadtxt(file_list, dtype=str)
            for name_sample in list_samples:
                self.samples_list.append((view, name_sample))

        self.test_list= np.loadtxt(os.path.join(self.root_list, f"{self.test_view}_test.list"), dtype=str)
        if self.phase == 'test':
            self.samples_list = self.test_list


    def __len__(self):
      return len(self.samples_list)


    def get_uniNorm(self, skeleton):
        'skeleton: T X 25 x 2, norm[0,1], (x-min)/(max-min)'
        # nonZeroSkeleton = []
        if self.dataType == '2D':  dim = 2
        else:                      dim = 3
        normSkeleton = np.zeros_like(skeleton)
        visibility = np.zeros(skeleton.shape)
        bbox = np.zeros((skeleton.shape[0], 4))
        for i in range(0, skeleton.shape[0]):
            nonZeros = []
            ids = []
            normPose = np.zeros_like((skeleton[i]))
            for j in range(0, skeleton.shape[1]):
                point = skeleton[i,j]

                if point[0] !=0 and point[1] !=0:

                    nonZeros.append(point)
                    ids.append(j)

            nonzeros = np.concatenate((nonZeros)).reshape(len(nonZeros), dim)
            minX, minY = np.min(nonzeros[:,0]), np.min(nonzeros[:,1])
            maxX, maxY = np.max(nonzeros[:,0]), np.max(nonzeros[:,1])
            normPose[ids,0] = (nonzeros[:,0] - minX)/(maxX-minX)
            normPose[ids,1] = (nonzeros[:,1] - minY)/(maxY-minY)
            if dim == 3:
                minZ, maxZ = np.min(nonzeros[:,2]), np.max(nonzeros[:,2])
                normPose[ids,2] = (nonzeros[:,1] - minZ)/(maxZ-minZ)
            normSkeleton[i] = normPose
            visibility[i,ids] = 1
            bbox[i] = np.asarray([minX, minY, maxX, maxY])

        return normSkeleton, visibility, bbox


    def get_rgbList(self, view, name_sample):
        data_path = os.path.join(self.data_root, view, name_sample)
        # print(data_path)
        # fileList = np.loadtxt(os.path.join(data_path, 'fileList.txt'))
        imgId = []
        imageList = []

        for item in os.listdir(data_path):
            if item.find('_rgb.jpg') != -1:
                id = int(item.split('_')[1])
                imgId.append(id)

        imgId.sort()

        for i in range(0, len(imgId)):
            for item in os.listdir(data_path):
                if item.find('_rgb.jpg') != -1:
                    if int(item.split('_')[1]) == imgId[i]:
                        imageList.append(item)
        # imageList.sort()

        'make sure it is sorted'
        return imageList, data_path


    def getAffineTransformation(self, skeleton):
        ''' For cross-sub, sample rates
            skeleton: T x 25 x 2',
        '''
        tx, ty = random.sample(range(-10,10),1)[0], random.sample(range(-10,10),1)[0]
        theta = random.sample(range(-180, 180),1)[0]
        scs = list(np.linspace(0.1, 10, 20))
        sx , sy = random.sample(scs, 1)[0], random.sample(scs, 1)[0]

        Translation = np.asarray([[1,0, tx],
                       [0,1,ty],
                       [0, 0, 1]])

        Rotation =np.asarray( [[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0,1]])

        scale = np.asarray([[sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1]])

        M1 = np.matmul(Translation, Rotation)
        M2 = np.matmul(scale, Rotation)

        affine1 = np.zeros_like(skeleton)
        affine2 = np.zeros_like(skeleton)

        for i in range(0, skeleton.shape[0]):
            pose = np.concatenate((skeleton[i].transpose(1, 0),np.ones((1,skeleton.shape[1]))))  # to homo
            aff1 = np.matmul(M1, pose)
            aff2 = np.matmul(M2, pose)

            affine1[i] = aff1[0:2,:].transpose(1,0)
            affine2[i] = aff2[0:2,:].transpose(1,0)

        affine1Norm,_,_ = self.get_uniNorm(affine1)
        affine2Norm,_,_ = self.get_uniNorm(affine2)

        affineSkeletons = np.concatenate((np.expand_dims(affine1Norm, 0), np.expand_dims(affine2Norm,0)))
        return affineSkeletons


    def paddingSeq_ske(self, skeleton, normSkeleton, visibility, affineSkeleton):
        Tadd = abs(skeleton.shape[0] - self.T)

        last = np.expand_dims(skeleton[-1, :, :], 0)
        copyLast = np.repeat(last, Tadd, 0)
        skeleton_New = np.concatenate((skeleton, copyLast), 0)  # copy last frame Tadd time

        lastNorm = np.expand_dims(normSkeleton[-1, :, :], 0)
        copyLastNorm = np.repeat(lastNorm, Tadd, 0)
        normSkeleton_New = np.concatenate((normSkeleton, copyLastNorm), 0)

        lastAffine = np.expand_dims(affineSkeleton[:,-1,:,:], 1)
        copyLastAff = np.repeat(lastAffine, Tadd, 1)
        affineSkeleton_new = np.concatenate((affineSkeleton, copyLastAff),1)

        lastMask = np.expand_dims(visibility[-1,:,:], 0)
        copyLastMask = np.repeat(lastMask, Tadd, 0)
        visibility_New = np.concatenate((visibility, copyLastMask), 0)

        return skeleton_New, normSkeleton_New, visibility_New, affineSkeleton_new


    def paddingSeq(self, skeleton, normSkeleton, imageSequence, ROIs, visibility, affineSkeleton,augImageSeqPair):
        Tadd = abs(skeleton.shape[0] - self.T)

        last = np.expand_dims(skeleton[-1, :, :], 0)
        copyLast = np.repeat(last, Tadd, 0)
        skeleton_New = np.concatenate((skeleton, copyLast), 0)  # copy last frame Tadd time

        lastNorm = np.expand_dims(normSkeleton[-1, :, :], 0)
        copyLastNorm = np.repeat(lastNorm, Tadd, 0)
        normSkeleton_New = np.concatenate((normSkeleton, copyLastNorm), 0)

        lastAffine = np.expand_dims(affineSkeleton[:,-1,:,:], 1)
        copyLastAff = np.repeat(lastAffine, Tadd, 1)
        affineSkeleton_new = np.concatenate((affineSkeleton, copyLastAff),1)

        lastMask = np.expand_dims(visibility[-1,:,:], 0)
        copyLastMask = np.repeat(lastMask, Tadd, 0)
        visibility_New = np.concatenate((visibility, copyLastMask), 0)

        lastImg = imageSequence[-1, :, :, :].unsqueeze(0)
        copyLastImg = lastImg.repeat(Tadd, 1, 1, 1)
        imageSequence_New = torch.cat((imageSequence, copyLastImg), 0)

        lastROI = ROIs[-1, :,:,:].unsqueeze(0)
        copyLastROI = lastROI.repeat(Tadd, 1, 1, 1)
        ROIs_New = torch.cat((ROIs, copyLastROI), 0)

        lastImagePair = augImageSeqPair[:,-1,:,:,:].unsqueeze(1)
        copyLastImagePair = lastImagePair.repeat(1,Tadd, 1,1,1)
        # print(lastImagePair.shape, copyLastImagePair.shape)
        augImageSeqPair_new = torch.cat((augImageSeqPair, copyLastImagePair),1)

        return skeleton_New, normSkeleton_New, imageSequence_New, ROIs_New, visibility_New,affineSkeleton_new, augImageSeqPair_new


    def get_data(self, view, name_sample):
        """RETURN:\\
        skeletonData = {\\
            'normSkeleton': [1, T,num_joint,dim_joint]\\
            'unNormSkeleton': [1, T,num_joint,dim_joint]\\
            'visibility':visibility_input,\\
            'affineSkeletons':affineSkeletons_input}
        """
        imagesList, data_path = self.get_rgbList(view, name_sample)
        jsonList, imgList = alignDataList(os.path.join(self.root_skeleton, view), name_sample, imagesList,'N-UCLA')
        assert  len(imgList) == len(jsonList)

        if self.dataType == '2D':
            skeleton, _, confidence = getJsonData(os.path.join(self.root_skeleton, view), name_sample, jsonList)
        else:
            skeleton = np.load(os.path.join(self.root_skeleton, view, name_sample + '.npy'), allow_pickle=True)
            confidence = np.ones_like(skeleton)
        
        T_sample, num_joints, dim = skeleton.shape
        normSkeleton, binaryMask, bboxes = self.get_uniNorm(skeleton)
        affineSkeletons = self.getAffineTransformation(skeleton)

        if self.maskType == 'binary': visibility = binaryMask
        else:                         visibility = confidence  # mask is from confidence score
        
        # Invalid data
        if self.T == 0:
            skeleton_input = skeleton
            normSkeleton_input = normSkeleton
            visibility_input = visibility
            affineSkeletons_input = affineSkeletons
            # imgSequence = np.zeros((T_sample, 3, 224, 224))
            details = {'name_sample': name_sample, 'T_sample': T_sample, 'time_offset': range(T_sample), 'view':view}
        else:
            if T_sample <= self.T:
                skeleton_input = skeleton
                normSkeleton_input = normSkeleton
                visibility_input = visibility
                affineSkeletons_input = affineSkeletons
            else:
                stride = T_sample / self.T
                ids_sample = []
                for i in range(self.T):
                    id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
                    ids_sample.append(id_sample)

                skeleton_input = skeleton[ids_sample, :, :]
                normSkeleton_input = normSkeleton[ids_sample,:,:]
                visibility_input = visibility[ids_sample,:,:]
                affineSkeletons_input = affineSkeletons[:,ids_sample,:,:]


            if skeleton_input.shape[0] != self.T:
                skeleton_input, normSkeleton_input, visibility_input, affineSkeletons_input \
                    = self.paddingSeq_ske(skeleton_input, normSkeleton_input, visibility_input, affineSkeletons_input)
        # Set clip as 1
        normSkeleton_input = np.expand_dims(normSkeleton_input,0)
        skeleton_input = np.expand_dims(skeleton_input,0)
        affineSkeletons_input = np.expand_dims(affineSkeletons_input,0)

        skeletonData = {'normSkeleton': normSkeleton_input, 'unNormSkeleton': skeleton_input,
                        'visibility':visibility_input, 'affineSkeletons':affineSkeletons_input}
        assert normSkeleton_input.shape[1] == self.T

        return skeletonData


    def get_data_multiSeq(self, view, name_sample):
        """RETURN:\\
        skeletonData = {\\
                'normSkeleton': [num_clips, T,num_joint,dim_joint]\\
                'unNormSkeleton': [num_clips, T,num_joint,dim_joint]\\
                'visibility':visibility_input,\\
                'affineSkeletons':affineSkeletons_input}
        """
        imagesList, data_path = self.get_rgbList(view, name_sample)
        jsonList, imgList = alignDataList(os.path.join(self.root_skeleton, view), name_sample, imagesList,'N-UCLA')

        assert len(imgList) == len(jsonList)

        if self.dataType == '2D':
            skeleton, usedID, confidence = getJsonData(os.path.join(self.root_skeleton, view), name_sample, jsonList)
        else:
            skeleton = np.load(os.path.join(self.root_skeleton, view, name_sample + '.npy'), allow_pickle=True)
            confidence = np.ones_like(skeleton)

        normSkeleton, binaryMask, bboxes = self.get_uniNorm(skeleton)
        affineSkeletons = self.getAffineTransformation(skeleton)

        if self.maskType == 'binary': visibility = binaryMask
        else:                         visibility = confidence  # mask is from confidence score

        T_sample, num_joints, dim = normSkeleton.shape
        stride = T_sample / self.clips
        ids_sample = []

        for i in range(self.clips):
            id_sample = random.randint(int(stride * i), int(stride * (i + 1)) - 1)
            ids_sample.append(id_sample)
        if T_sample <= self.T:
            skeleton_input, normSkeleton_input, visibility_input, affineSkeletons_input = \
                self.paddingSeq_ske(skeleton, normSkeleton, visibility, affineSkeletons)
            temp = np.expand_dims(normSkeleton_input, 0)
            inpSkeleton_all = np.repeat(temp, self.clips, 0)

            tempMask = np.expand_dims(visibility_input, 0)
            visibility_input = np.repeat(tempMask, self.clips, 0)

            temp_skl = np.expand_dims(skeleton_input, 0)
            skeleton_all = np.repeat(temp_skl, self.clips, 0)

            temAff = np.expand_dims(affineSkeletons_input,0)
            affineSkeletons_input = np.repeat(temAff,self.clips, 0)

        else: # T_sample > self.T

            inpSkeleton_all = []
            visibility_input = []
            skeleton_all = []
            affineSkeletons_input = []

            for id in ids_sample:

                if (id - int(self.T / 2)) <= 0 < (id + int(self.T / 2)) < T_sample:
                    temp = np.expand_dims(normSkeleton[0:self.T], 0)
                    temp_skl = np.expand_dims(skeleton[0:self.T], 0)
                    tempMask = np.expand_dims(visibility[0:self.T], 0)
                    temAff = np.expand_dims(affineSkeletons[:,0:self.T],0)
                elif 0 < (id-int(self.T/2)) <= (id + int(self.T / 2)) < T_sample:
                    temp = np.expand_dims(normSkeleton[id - int(self.T / 2):id + int(self.T / 2)], 0)
                    temp_skl = np.expand_dims(skeleton[id - int(self.T / 2):id + int(self.T / 2)], 0)
                    tempMask = np.expand_dims(visibility[id - int(self.T / 2):id + int(self.T / 2)],0)
                    temAff = np.expand_dims(affineSkeletons[:,id - int(self.T / 2):id + int(self.T / 2)],0)

                elif (id - int(self.T/2)) < T_sample <= (id+int(self.T / 2)):
                    temp = np.expand_dims(normSkeleton[T_sample - self.T:], 0)
                    temp_skl = np.expand_dims(skeleton[T_sample - self.T:], 0)
                    tempMask = np.expand_dims(visibility[T_sample - self.T:], 0)
                    temAff = np.expand_dims(affineSkeletons[:, T_sample - self.T:],0)
                else:
                    temp = np.expand_dims(normSkeleton[T_sample - self.T:], 0)
                    temp_skl = np.expand_dims(skeleton[T_sample - self.T:], 0)
                    tempMask = np.expand_dims(visibility[T_sample - self.T:], 0)
                    temAff = np.expand_dims(affineSkeletons[:, T_sample - self.T:], 0)

                inpSkeleton_all.append(temp)
                skeleton_all.append(temp_skl)
                visibility_input.append(tempMask)
                affineSkeletons_input.append(temAff)

            inpSkeleton_all = np.concatenate((inpSkeleton_all), 0)
            skeleton_all = np.concatenate((skeleton_all), 0)
            visibility_input = np.concatenate((visibility_input), 0)
            affineSkeletons_input = np.concatenate((affineSkeletons_input),0)
        skeletonData = {'normSkeleton':inpSkeleton_all, 'unNormSkeleton': skeleton_all,
                        'visibility':visibility_input, 'affineSkeletons':affineSkeletons_input}
        
        return skeletonData


    def __getitem__(self, index):
        """
        Return:
            skeletons: FloatTensor, [T, num_joints, 2]
            label_action: int, label for the action
            info: dict['sample_name', 'T_sample', 'time_offset']
        """
        # ipdb.set_trace()
        if self.phase == 'test':
            name_sample = self.samples_list[index]
            view = self.test_view
        else:
            view, name_sample = self.samples_list[index]

        if self.sampling == 'Single':
            # [1, T, num_joints, 2]
            skeletons = self.get_data(view, name_sample) 
        else:
            # [nClip, T, num_joints, 2]
            skeletons = self.get_data_multiSeq(view, name_sample) 
        # output affine skeletons
        label_action = self.action_list[name_sample[:3]]
        dicts = {'input_skeletons': skeletons,
                 'action': label_action,
                 'sample_name':name_sample,
                 'cam':view}

        return dicts


if __name__ == "__main__":
    setup = 'setup1'  # v1,v2 train, v3 test;
    path_list = '../data_list/NUCLA/CV/' + setup + '/'
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    trainSet = NUCLA_CrossView_ske(root_list=path_list, dataType='2D', sampling='Single', phase='train', T=36,maskType='score',
                               setup=setup, nClip=1)

    # pass

    trainloader = DataLoader(trainSet, batch_size=12, shuffle=False, num_workers=4)

    for i,sample in enumerate(trainloader):
        print('sample:', i)
        # heatmaps = sample['heat']
        # images = sample['input_images']
        inp_skeleton = sample['input_skeletons']['normSkeleton']
        visibility = sample['input_skeletons']['visibility']
        label = sample['action']
        # ROIs = sample['input_rois']
        # augImagePair = sample['input_imagePair']
        # ipdb.set_trace()
        # print(inp_skeleton.shape)

    print('done')
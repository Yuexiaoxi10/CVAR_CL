import torch.nn as nn
import torch

import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms
from DIR_CL import *
 
from utils import *
from actRGB import *
from gumbel_module import *
from scipy.spatial import distance

class fusionLayers(nn.Module):
    def __init__(self, num_class, in_chanel_x, in_chanel_y):
        super(fusionLayers, self).__init__()
        self.num_class = num_class
        self.in_chanel_x = in_chanel_x
        self.in_chanel_y = in_chanel_y
        self.cat = nn.Linear(self.in_chanel_x + self.in_chanel_y, 128)
        self.cls = nn.Linear(128, self.num_class)
        self.relu = nn.LeakyReLU()
    def forward(self, feat_x, feat_y):
        twoStreamFeat = torch.cat((feat_x, feat_y), 1)
        out = self.relu(self.cat(twoStreamFeat))
        label = self.cls(out)
        return label, out


class twoStreamClassification(nn.Module):
    def __init__(self, num_class, Npole, Drr, Dtheta, data_dim, gpu_id, gumbel_inference_mode,mode,
                  nClip, fistaLam, dataType, kinetics_pretrain):
        super(twoStreamClassification, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.Drr = Drr
        self.Dtheta = Dtheta
        # self.PRE = PRE
        self.gpu_id = gpu_id
        self.dataType = dataType
        self.data_dim = data_dim
        self.kinetics_pretrain = kinetics_pretrain
        self.Inference = gumbel_inference_mode
        self.nClip = nClip
        self.fistaLam = fistaLam
        
        self.dynamicsClassifier = contrastiveNet(Npole=self.Npole, Drr=self.Drr, Dtheta=self.Dtheta, gumbel_inference_mode=self.Inference, gpu_id=self.gpu_id, data_dim=self.data_dim,mode=mode, dataType=self.dataType, fistaLam=fistaLam, fineTune=True, nClip=self.nClip, useCL=False )
        self.RGBClassifier = RGBAction(self.num_class, self.kinetics_pretrain)

        self.lastPred = fusionLayers(self.num_class, in_chanel_x=512, in_chanel_y=128)

    def forward(self,skeleton, image, rois, fusion, bi_thresh):
        # stream = 'fusion'
        bz = skeleton.shape[0]
        if bz == 1:
            skeleton = skeleton.repeat(2,1,1,1)
            image = image.repeat(2,1,1,1,1)
            rois = rois.repeat(2,1,1,1,1)
        label1,lastFeat_DIR, binaryCode, Reconstruction = self.dynamicsClassifier(skeleton, bi_thresh)
        label2, lastFeat_CIR,_ = self.RGBClassifier(image, rois)

        if fusion:
            label = {'RGB':label1, 'Dynamcis':label2}
            feats = lastFeat_DIR
        else:
            # label = 0.5 * label1 + 0.5 * label2
            label, feats= self.lastPred(lastFeat_DIR, lastFeat_CIR)
        if bz == 1 :
            nClip = int(label.shape[0]/2)
            return label[0:nClip], binaryCode[0:nClip], Reconstruction[0:nClip], feats
        else:
            return label, binaryCode, Reconstruction, feats
        


if __name__ == '__main__':
    gpu_id = 0

    N = 2*80
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    kinetics_pretrain = '../pretrained/i3d_kinetics.pth'
    
    net = twoStreamClassification(num_class=10, Npole=161, num_binary=161, Drr=Drr, Dtheta=Dtheta,
                                  dim=2, gpu_id=gpu_id,gumbel_inference_mode=True, fistaLam=0.1, dataType='2D',
                                  kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)
    
    x = torch.randn(5, 36, 50).cuda(gpu_id)
    xImg = torch.randn(5, 20, 3, 224, 224).cuda(gpu_id)
    T = x.shape[1]
    xRois = xImg
    label, _, _ = net(x, xImg, xRois, T, False)



    print('check')



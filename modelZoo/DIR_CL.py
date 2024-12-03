import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms
from gumbel_module import *
from utils import *
from actRGB import *
from sparseCoding import *

class contrastiveNet(nn.Module):
    def __init__(self, Npole, Drr, Dtheta, gumbel_inference_mode, gpu_id, data_dim, mode, dataType,fistaLam,fineTune,nClip, useCL):
        super(contrastiveNet, self).__init__()

        # self.dim_in = dim_in
        self.Npole = Npole
        self.nClip = nClip
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.Inference = gumbel_inference_mode
        self.gpu_id = gpu_id
        self.data_dim = data_dim
        self.mode = mode
        self.fistaLam = fistaLam
        self.dim_embed = 128
        self.dataType = dataType
        
        self.num_class = 10
        self.fineTune = fineTune
        self.useCL = useCL
        if self.mode == 'rgb':
            self.backbone = RGBAction(num_class=self.num_class, kinetics_pretrain='./pretrained/i3d_kinetics.pth')
            dim_mlp = self.backbone.cls.in_features
        else:
            self.backbone = Fullclassification(self.num_class, self.Npole, self.Drr, self.Dtheta, self.data_dim, self.dataType, self.Inference, self.gpu_id, self.fistaLam, self.useCL)
        # if self.useCL == False:
        #     dim_mlp = self.backbone.Classifier.cls.in_features
        # else:
            dim_mlp = self.backbone.Classifier.cls[0].in_features
        self.proj = nn.Linear(dim_mlp,self.dim_embed)
        self.relu = nn.LeakyReLU()
        

    def forward(self, x, y):
        'if x: affine skeleton, then y:bi_thresh'
        'if x: img, then y: roi'
        bz = x.shape[0]
        # if len(x.shape) == 3:
        #     x = x.unsqueeze(0)
        if self.fineTune == False:
            
            if self.mode == 'rgb':
                if bz < 2:
                    x = x.repeat(2, 1, 1, 1, 1,1)
                    bz = x.shape[0]
                x1_img, x2_img = x[:,0], x[:,1]
                x1_roi, x2_roi = x[:,2], x[:,3]

                _, lastFeat1, _ = self.backbone(x1_img, x1_roi)
                _, lastFeat2, _ = self.backbone(x2_img, x2_roi)
            else:
                if bz < 2:
                    x = x.repeat(2,1,1,1,1)
                    bz = x.shape[0]
                # x = x.reshape(x.shape[0]* x.shape[1], x.shape[2], x.shape[3])
                
                x1 = x[:,0]
                x2 = x[:,1]
                _, lastFeat1,_,_ = self.backbone(x1, y)
                _, lastFeat2,_,_ = self.backbone(x2, y)

            embedding1 = self.relu(self.proj(lastFeat1))
            embedding2 = self.relu(self.proj(lastFeat2))


            embed1 = torch.mean(embedding1.reshape(bz, self.nClip, embedding1.shape[-1]),1)
            embed2 = torch.mean(embedding2.reshape(bz, self.nClip, embedding2.shape[-1]),1)
            z1 = F.normalize(embed1, dim=1)
            z2 = F.normalize(embed2, dim=1)

            features = torch.cat([z1,z2], dim=0)
            labels = torch.cat([torch.arange(bz) for i in range(2)], dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda(self.gpu_id)

            simL_matrix = torch.matmul(features, features.T)
            mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.gpu_id)
            labels = labels[~mask].view(labels.shape[0],-1)
            simL_matrix = simL_matrix[~mask].view(simL_matrix.shape[0], -1)
            positives = simL_matrix[labels.bool()].view(labels.shape[0], -1)
            negatives = simL_matrix[~labels.bool()].view(simL_matrix.shape[0], -1)

            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.gpu_id)
            temper = 0.07 #default
            logits = logits/temper

            return logits, labels
        else:
            if self.mode == 'rgb':
                
                return self.backbone(x,y )
            else:
                
                return self.backbone(x, y)

class MLP(nn.Module):
    def __init__(self,  dim_in):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(dim_in, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        # self.gelu = nn.GELU()
        self.relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x_out = self.relu(self.bn1(self.layer1(x)))
        x_out = self.relu(self.bn2(self.layer2(x_out)))
        # x_out = self.sig(x_out)

        return x_out


class Fullclassification(nn.Module):
    def __init__(self, num_class, Npole, Drr, Dtheta,dim, dataType, gumbel_inference_mode, gpu_id, fistaLam,useCL):
        super(Fullclassification, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.Inference = gumbel_inference_mode
        self.gpu_id = gpu_id
        self.dim = dim
        self.useCL = useCL
        self.dataType = dataType
        
        self.fistaLam = fistaLam
      
        self.BinaryCoding = GumbelSigmoid()

        self.sparseCoding = DyanEncoder(self.Drr, self.Dtheta, lam=fistaLam, gpu_id=self.gpu_id)
        self.Classifier = classificationGlobal(num_class=self.num_class, Npole=self.Npole, dataType=self.dataType, useCL=self.useCL)

    def forward(self, x, bi_thresh):
       
        T = x.shape[1]
     
        sparseCode, Dict, _ = self.sparseCoding(x, T) # w.RH

        'for GUMBEL'
        binaryCode = self.BinaryCoding(sparseCode**2, bi_thresh, force_hard=True, temperature=0.1, inference=self.Inference)
        temp1 = sparseCode * binaryCode
      
        Reconstruction = torch.matmul(Dict, temp1)
        sparseFeat = binaryCode
       
        label, lastFeat = self.Classifier(sparseFeat)

        return label, lastFeat, binaryCode, Reconstruction

class classificationGlobal(nn.Module):
    def __init__(self, num_class, Npole, dataType, useCL):
        super(classificationGlobal, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
    
        self.useCL = useCL
        self.dataType = dataType
        self.conv1 = nn.Conv1d(self.Npole, 256, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(num_features=256, eps=1e-05, affine=True)

        self.conv2 = nn.Conv1d(256, 512, 1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(num_features=512, eps=1e-5, affine=True)

        self.conv3 = nn.Conv1d(512, 1024, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=1024, eps=1e-5, affine=True)

       
        self.conv4 = nn.Conv2d(self.Npole + 1024, 1024, (3, 1), stride=1, padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(num_features=1024, eps=1e-5, affine=True)

        self.conv5 = nn.Conv2d(1024, 512, (3, 1), stride=1, padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(num_features=512, eps=1e-5, affine=True)

        self.conv6 = nn.Conv2d(512, 256, (3, 3), stride=2, padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=256, eps=1e-5, affine=True)
        if self.dataType == '2D' :
            self.njts = 25
            self.fc = nn.Linear(256*10*2, 1024) #njts = 25
        elif self.dataType == 'rgb':
            self.njts = 512 # for rgb
            # self.fc = nn.Linear(256*61*2, 1024) #for rgb
            self.fc = nn.Linear(256*253*2, 1024) # for att rgb
        elif self.dataType == '2D+rgb':
            self.njts = 512+25
            self.fc = nn.Linear(256*266*2,1024)
            
        self.pool = nn.AvgPool1d(kernel_size=(self.njts))
        # self.fc = nn.Linear(7168,1024) #njts = 34
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)

        # self.linear = nn.Sequential(nn.Linear(256*10*2,1024),nn.LeakyReLU(),nn.Linear(1024,512),nn.LeakyReLU(), nn.Linear(512, 128), nn.LeakyReLU())
        if self.useCL == False:

            # self.cls = nn.Linear(128, self.num_class)
            self.cls = nn.Sequential(nn.Linear(128,128),nn.LeakyReLU(),nn.Linear(128,self.num_class))
        else:

            self.cls = nn.Sequential(nn.Linear(128, self.num_class))
        self.relu = nn.LeakyReLU()

        'initialize model weights'
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu' )

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)


    def forward(self,x):
        inp = x
        if self.dataType == '2D' or 'rgb':
            dim = 2
        else:
            dim = 3

        bz = inp.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # ipdb.set_trace()
        x_gl = self.pool(self.relu(self.bn3(self.conv3(x))))
        # ipdb.set_trace()
        x_new = torch.cat((x_gl.repeat(1,1,inp.shape[-1]),inp),1).reshape(bz,1024+self.Npole, self.njts,dim)

        x_out = self.relu(self.bn4(self.conv4(x_new)))
        x_out = self.relu(self.bn5(self.conv5(x_out)))
        x_out = self.relu(self.bn6(self.conv6(x_out)))

        'MLP'
        # ipdb.set_trace()
        x_out = x_out.view(bz,-1)  #flatten
        
        x_out = self.relu(self.fc(x_out))
        x_out = self.relu(self.fc2(x_out))
        x_out = self.relu(self.fc3(x_out)) #last feature before cls

        out = self.cls(x_out)

        return out, x_out

import sys
sys.path.append('.')
sys.path.append('./modelZoo')
from torch.utils.data import DataLoader
import torch
from dataset.crossView_UCLA import *
from torch.optim import lr_scheduler
from modelZoo.DIR_CL import *
from modelZoo.twoStream import *
# from testClassifier_CV import testing, getPlots
import time
from matplotlib import pyplot as plt
from testClassifier_CV import *
import configurations
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def train_one_epoch(sample, mode, net):
    skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(configurations.gpu_id)
           
    images = sample['input_images'].float().cuda(configurations.gpu_id)
    ROIs = sample['input_rois'].float().cuda(configurations.gpu_id)
    

    t = skeletons.shape[2]
    input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1) #bz,clip, T, 25, 2 --> bz*clip, T, 50
   
    input_images = images.reshape(images.shape[0]*images.shape[1], t, 3, 224, 224)
    input_rois = ROIs.reshape(ROIs.shape[0]* ROIs.shape[1], t, 3, 224, 224)
    nClip = skeletons.shape[1]

    if mode == 'DIR_CL':
        # ipdb.set_trace()
        # input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1)
        actPred, lastFeat, binaryCode, output_skeletons = net(input_skeletons, configurations.gumbel_thresh)

        actPred = actPred.reshape(skeletons.shape[0], nClip, configurations.num_class)
        actPred = torch.mean(actPred, 1)
        target_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1],t,-1)

    else:
        actPred, binaryCode, output_skeletons, lastFeat = net(input_skeletons, input_images, input_rois, fusion=False, bi_thresh=configurations.gumbel_thresh)
        actPred = actPred.reshape(int(actPred.shape[0]/nClip), nClip, configurations.num_class)

        actPred = torch.mean(actPred, 1)
        target_skeletons = skeletons.reshape(skeletons.shape[0]* skeletons.shape[1], t, -1)

    return actPred, binaryCode, output_skeletons, target_skeletons
        

def trainer(Epoch, trainloader, testloader, mode):
    'hyper parameter initialized'

    P,Pall = gridRing(configurations.Npoles*2)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()

    if configurations.dataType == '2D':
        data_dim = 2
    else:
        data_dim = 3

    map_loc = "cuda:" + str(configurations.gpu_id)

    if mode == 'DIR_CL':
        net = contrastiveNet(Npole=configurations.Npoles*2+1, Drr=Drr, Dtheta=Dtheta, gumbel_inference_mode=True, gpu_id=configurations.gpu_id, data_dim=data_dim, dataType=configurations.dataType, fistaLam=configurations.fista_lam, mode=configurations.mode, fineTune=configurations.fineTune, nClip=configurations.nClip, useCL=configurations.useCL).cuda(configurations.gpu_id)
        pretrained_model = configurations.pretrained_model_DIR
        state_dict = torch.load(pretrained_model, map_location=map_loc)
        net = load_fineTune_model(state_dict, net)
        'frozen all layers except last classification layer'
        # for p in net.backbone.sparseCoding.parameters():
        #     p.requires_grad = False

        # for p in net.backbone.Classifier.parameters():
        #     p.requires_grad = False

        for p in net.backbone.Classifier.cls[-1].parameters():
            p.requires_grad = True
        
        optimizer = torch.optim.SGD(
        [{'params': filter(lambda x: x.requires_grad, net.backbone.sparseCoding.parameters()), 'lr': configurations.lr_sparseCoding},
         {'params': filter(lambda x: x.requires_grad, net.backbone.Classifier.parameters()), 'lr': configurations.lr_classifier}], weight_decay=1e-3,
        momentum=0.9)


    else:
        # kinetics_pretrain = './pretrained/i3d_kinetics.pth'
        net = twoStreamClassification(num_class=configurations.num_class, Npole=configurations.Npoles*2+1,
                                       Drr=Drr, Dtheta=Dtheta, data_dim=data_dim, gpu_id=configurations.gpu_id,gumbel_inference_mode=True, fistaLam=configurations.fista_lam, dataType=configurations.dataType,
                                  kinetics_pretrain=configurations.kinetics_pretrain,mode='DIR', nClip=configurations.nClip).cuda(configurations.gpu_id)
        pretrained_model_DIR = configurations.pretrained_model_DIR
        state_dict_DIR = torch.load(pretrained_model_DIR, map_location=map_loc)
        net.dynamicsClassifier = load_fineTune_model(state_dict_DIR, net.dynamicsClassifier)

        pretrained_model_CIR = configurations.pretrained_model_CIR
        state_dict_CIR = torch.load(pretrained_model_CIR, map_location=map_loc)
        net.RGBClassifier = load_fineTune_model(state_dict_CIR, net.RGBClassifier)

        # ipdb.set_trace()
        optimizer = torch.optim.SGD(
        [{'params': filter(lambda x: x.requires_grad, net.parameters()), 'lr': 1e-3}], weight_decay=0.001,
        momentum=0.9)


    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
    Criterion = torch.nn.CrossEntropyLoss()
    mseLoss = torch.nn.MSELoss()
    L1loss = torch.nn.SmoothL1Loss()    
    
    for epoch in range(1, Epoch+1):
        print('start training epoch:', epoch)
        lossVal = []
        lossCls = []
        lossBi = []
        lossMSE = []
        start_time = time.time()
        for i, sample in enumerate(trainloader):
            # print('sample:', i)
            optimizer.zero_grad()
            gt_label = sample['action'].cuda(configurations.gpu_id)

            actPred, binaryCode, output_skeletons, target_skeletons = train_one_epoch(sample, mode, net)

            bi_gt = torch.zeros_like(binaryCode).cuda(configurations.gpu_id)
            loss = configurations.lam1 * Criterion(actPred, gt_label) + configurations.lam2 * mseLoss(output_skeletons, target_skeletons.squeeze(-1)) \
                    + configurations.Alpha * L1loss(binaryCode, bi_gt)
            
            lossMSE.append(mseLoss(output_skeletons, target_skeletons.squeeze(-1)).data.item())
            # print('output shape:', output_skeletons.shape, 'mse:', mseLoss(output_skeletons, input_skeletons).data.item())
        
            lossBi.append(L1loss(binaryCode, bi_gt).data.item())

            loss.backward()
            # ipdb.set_trace()
            # print('rr.grad:', net.sparseCoding.rr.grad, 'mse:', lossMSE[-1])
            optimizer.step()
            lossVal.append(loss.data.item())

            lossCls.append(Criterion(actPred, gt_label).data.item())

        loss_val = np.mean(np.array(lossVal))
        
        print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|mse:', np.mean(np.array(lossMSE)),
            '|bi:', np.mean(np.array(lossBi)))
        end_time = time.time()
        print('training time(h):', (end_time - start_time) / 3600)


        scheduler.step()
        if epoch % 1 == 0:
            # torch.save({'state_dict': net.state_dict(),
            #            'optimizer': optimizer.state_dict()}, configurations.save_model_path + str(epoch) + '.pth')

            Acc = validating(testloader, net, gpu_id, configurations.mode,configurations.gumbel_thresh)
            print('testing epoch:', epoch, 'Acc:%.4f' % Acc)

    torch.cuda.empty_cache()
    print('done')


if __name__ == '__main__':


    trainSet = NUCLA_CrossView(input_data_list=configurations.input_path_list, input_data_root=configurations.input_data_path, input_skeleton_path=configurations.input_skeleton_path, 
                               dataType=configurations.dataType, sampling=configurations.sampling, phase='train', T=configurations.T, setup=configurations.setup, nClip=configurations.nClip)

    trainloader = DataLoader(trainSet, batch_size=configurations.batch_size, shuffle=True, num_workers=configurations.num_workers)
    

    testSet = NUCLA_CrossView(input_data_list=configurations.input_path_list, input_data_root=configurations.input_data_path, input_skeleton_path=configurations.input_skeleton_path, 
                               dataType=configurations.dataType, sampling=configurations.sampling, phase='test', T=configurations.T, setup=configurations.setup, nClip=configurations.nClip)
    testloader = DataLoader(testSet, batch_size=configurations.batch_size, shuffle=False, num_workers=configurations.num_workers)

    print('gumbel tresh:', configurations.gumbel_thresh)
    trainer(configurations.EPOCH, trainloader, testloader, configurations.mode)









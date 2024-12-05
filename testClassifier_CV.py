from torch.utils.data import DataLoader
import sys
sys.path.append('.')
sys.path.append('./modelZoo')
from dataset.crossView_UCLA import NUCLA_CrossView 
from dataset.crossView_UCLA_sk import NUCLA_CrossView_ske
from torch.optim import lr_scheduler
# from modelZoo.BinaryCoding import *
import matplotlib.pyplot as plt
import configurations
from modelZoo.DIR_CL import *
from modelZoo.twoStream import *
random.seed(configurations.seed)
np.random.seed(configurations.seed)
torch.manual_seed(configurations.seed)

def validating(dataloader,net, gpu_id, mode, gumbel_thresh):
    count = 0
    pred_cnt = 0
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            # print('sample:', i)
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
            
            gt = sample['action'].cuda(configurations.gpu_id)
            t = skeletons.shape[2]
            input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1) #bz,clip, T, 25, 2 --> bz*clip, T, 50
            nClip = configurations.nClip
            # ipdb.set_trace()
            if mode == 'DIR_CL':
                # input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1)
                actPred, _,_, _ = net(input_skeletons, gumbel_thresh)
            else:

                images = sample['input_images'].float().cuda(gpu_id)
                ROIs = sample['input_rois'].float().cuda(gpu_id)

                input_images = images.reshape(images.shape[0]*images.shape[1], t, 3, 224, 224)
                input_rois = ROIs.reshape(ROIs.shape[0]* ROIs.shape[1], t, 3, 224, 224)
                actPred, _, _ , _ = net(input_skeletons, input_images, input_rois, fusion=False, bi_thresh=gumbel_thresh)

            actPred = actPred.reshape(skeletons.shape[0], nClip, actPred.shape[-1])
            actPred = torch.mean(actPred, 1)
            pred = torch.argmax(actPred, 1)


            correct = torch.eq(gt, pred).int()
            count += gt.shape[0]
            pred_cnt += torch.sum(correct).data.item()

        Acc = pred_cnt/count

    return Acc

def inference(model_checkpoint):
    'model initialization'
    P, Pall = gridRing(configurations.Npoles*2)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()

    if configurations.dataType == '2D':
        data_dim = 2
    else:
        data_dim = 3
    
    if configurations.mode == 'DIR_CL':
        net = contrastiveNet(Npole=configurations.Npoles*2+1, Drr=Drr, Dtheta=Dtheta, gumbel_inference_mode=True, gpu_id=configurations.gpu_id, data_dim=data_dim, mode=configurations.mode, dataType=configurations.dataType, fistaLam=configurations.fista_lam, fineTune=configurations.fineTune, nClip=configurations.nClip, useCL=configurations.useCL).cuda(configurations.gpu_id)
        testSet = NUCLA_CrossView_ske(root_list=configurations.input_path_list, dataType=configurations.dataType, sampling=configurations.sampling, phase='test', T=36,maskType='score',
                               setup=configurations.setup, nClip=configurations.nClip)

        testloader = DataLoader(testSet, batch_size=configurations.batch_size, shuffle=False, num_workers=configurations.num_workers)

    else:
        testSet = NUCLA_CrossView(input_data_list=configurations.input_path_list, input_data_root=configurations.input_data_path, input_skeleton_path=configurations.input_skeleton_path, 
                               dataType=configurations.dataType, sampling=configurations.sampling, phase='test', T=configurations.T, setup=configurations.setup, nClip=configurations.nClip)
        testloader = DataLoader(testSet, batch_size=configurations.batch_size, shuffle=False, num_workers=configurations.num_workers)

        net = twoStreamClassification(num_class=configurations.num_class, Npole=configurations.Npoles*2+1, 
                                       Drr=Drr, Dtheta=Dtheta, data_dim=data_dim, gpu_id=configurations.gpu_id,gumbel_inference_mode=True, fistaLam=configurations.fista_lam, dataType=configurations.dataType,
                                  kinetics_pretrain=configurations.kinetics_pretrain, nClip=configurations.nClip).cuda(configurations.gpu_id)

    map_loc = "cuda:" + str(configurations.gpu_id)
    stateDict = torch.load(model_checkpoint, map_location=map_loc)['state_dict']
    net.load_state_dict(stateDict)
   
    Acc = validating(testloader, net, configurations.gpu_id, configurations.mode,configurations.gumbel_thresh)

    print('Acc:%.4f' % Acc)
    print('done')

if __name__ == '__main__':

    inference(configurations.model_checkpoint)






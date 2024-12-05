import os
'dataset configuration'
seed = 0
dataset = 'NUCLA'
num_class = 10
dataType = '2D' # input type for the model
sampling = 'Multi'
setup = 'setup1'
baseline = False # set for 'True' if doing classification without constrastive learning step, otherwise set to 'False'

input_path_list = './data_list/' + dataset + '/CV/' + setup + '/'
input_skeleton_path = '/data/N-UCLA_MA_3D/openpose_est' # the input 2D skeletons are generated with openpose
input_data_path = '/data/N-UCLA_MA_3D/multiview_action'

kinetics_pretrain = './pretrained/i3d_kinetics.pth'

constrastive = True
fusion = False 
num_class = 10
T = 36 
if sampling == 'Single':
    nClip = 1
else:
    nClip = 6 # for multi-sampling 
# mode = 'DIR_CL' 
mode = '2stream_CL'

if baseline:
    useCL = False # set for 'False' 
else:
    useCL = True # set for 'True' if pre-trained on contrastive learning step

fineTune = True # set for 'False' when doing contrastive learning step, 'True' for classification step
endToend = False # training end to end, set to 'False' if only fine-tune last classification layer

'experiment configuration'

if not baseline:
    pretrained_model_DIR = './pretrained/' + dataset + '/' + setup + '/' + sampling + '/pretrainedDIR_CL_ep100.pth' 
else:   
    pretrained_model_DIR = './pretrained/' + dataset + '/' + setup + '/' + sampling + '/pretrainedRHDYAN_for_CL.pth'
   
pretrained_model_CIR = './pretrained/' + dataset + '/' + setup + '/' + sampling + '/pretrainedCIR_CL.pth'

save_model_path = '/home/yuexi/Documents/CVAR_checkpoint/Multi/2Stream_CL/'
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)

'training configuration'
lr_classifier = 1e-3 
lr_sparseCoding = 1e-4 
gumbel_thresh = 0.502

if endToend:
    EPOCH = 100
else:
    EPOCH = 100 # if only training last layer, converging fast

Npoles = 80 # number of poles in the dictionary
gpu_id = 4
fista_lam = 0.1
Alpha = 0.1 # for bi loss
lam1 = 2 # cls loss
lam2 = 1 # mse loss


if mode == 'DIR_CL':

    # model_checkpoint = 'checkpoints/' + dataset + '/' + sampling + '/' + 'ckpt_dir_cl.pth'
    model_checkpoint = 'checkpoints/' + dataset + '/' + sampling + '/' + 'ckpt_dir.pth'

    num_workers = 8
    batch_size = 12
    
else:
    model_checkpoint = 'checkpoints/' + dataset + '/' + sampling + '/' + 'ckpt_2stream_cl.pth'
    num_workers = 2
    batch_size = 4

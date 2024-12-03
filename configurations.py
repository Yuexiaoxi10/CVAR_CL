'dataset configuration'
dataset = 'NUCLA'
num_class = 10
dataType = '2D' # input type for the model
sampling = 'Multi'
setup = 'setup1'
input_path_list = './data_list/' + dataset + '/CV/' + setup + '/'
input_skeleton_path = '/data/N-UCLA_MA_3D/openpose_est' # the input 2D skeletons are generated with openpose
input_data_path = '/data/N-UCLA_MA_3D/multiview_action'

'experiment configuration'
pretrained_model_DIR = './pretrained/' + dataset + '/' + setup + '/' + sampling + '/pretrainedDIR_CL.pth'
pretrained_model_CIR = './pretrained/' + dataset + '/' + setup + '/' + sampling + '/pretrainedCIR_CL.pth'
kinetics_pretrain = './pretrained/i3d_kinetics.pth'

constrastive = True
fusion = False 
num_class = 10
T = 36 
nClip = 4 # for multi-sampling 
mode = 'DIR_CL' 
# mode = '2stream_CL'
useCL = False # set for 'True' when doing constrative learning step
fineTune = True # set for 'False' when doing contrastive learning step

# save_model_path = 'your/path/to/save/models/' 
save_model_path = ''

'training configuration'
lr_classifier = 1e-3 
lr_sparseCoding = 1e-4 
gumbel_thresh = 0.501
EPOCH = 70
num_workers = 3
batch_size = 8
Npoles = 80 # number of poles in the dictionary
gpu_id = 4
fista_lam = 0.1
Alpha = 0.1 # for bi loss
lam1 = 2 # cls loss
lam2 = 1 # mse loss

'testing configurations'
if mode == 'DIR_CL':
    model_checkpoint = 'checkpoints/' + dataset + '/' + sampling + '/' + 'ckpt_dir_cl.pth'
    num_workers = 4
    batch_size = 8
    
else:
    model_checkpoint = 'checkpoints/' + dataset + '/' + sampling + '/' + 'ckpt_2stream_cl.pth'
    num_workers = 2
    batch_size = 2







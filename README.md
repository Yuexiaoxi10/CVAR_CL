# CVAR_CL
  * This repo is the official implementation for: Cross-view Action Recogintion via Constractive Dynamic View-Invariant Representations
  * This repo is designed to showcase the usage of our pipeline, and all experiments from this repo is based on **N-UCLA dataset**, cross-view experiment with **'setup1'** configuration, and **MULTI-CLIP** sampling strategy
      * Setup1 configuration: training on 'view1, view2', and testing on 'view3'
      * Input data: 2D skeleton, and rgb images 
  * All pre-trained models and provided checkpoints only support **'setup1'** configuration with **MULTI-CLIP** sampling strategy


## Software Requirement
  * python>=3.6
  * pytorch>=1.10.1
  
## Repo Structure :
  * checkpoints: checkpoints for inference (you will need to download this folder from the google drive link that we shared in the below) 
  * data_list: lists of samples for each view
  * dataset : dataloader
  * modelZoo: model scripts
  * pretrained: pretrained models before training the final classifier (you will need to download this folder from the google drive link that we shared in the below)
  * trainClassifier_CV.py (training script, cross-view)
  * testClassifier_CV.py (inference script, cross-view)
  * configurations.py (experiment configurations, cross-view)

## Pretrained Models
  * The Pretrained 2D skeleton based RHS are needed for all training step, please load accordingly
    
  |Sampling| Model name | Description| Model Path | Gumbel Threshold|
  |---| --- | --- | --- | --- |
  |Multi| pretrainedRHDYAN_for_CL.pth| Loading this model to DIR stream for classification without applying constrastive learning step| pretrained/UCLA/setup1/Multi | 0.505 |
  |Multi| pretrainedDIR_CL_ep100.pth | Loading this model to DIR stream for final classification step| pretrained/NUCLA/setup1/Multi | 0.502 |
  |Multi| pretrainedCIR_CL.pth | Loading this model to CIR stream for final classification step| pretrained/NUCLA/setup1/Multi | 0.502 |
  
## Model Download:
  * Go to https://drive.google.com/drive/folders/1UTm2twdRtXKJoxx0lqUSlf1y8IYkqSDs?usp=sharing to download our pre-trained models and  best checkpoints for each experiment.
  * Note: there are two folders, 'pretrained' and 'checkpoints', please download the entire folders and paste to the working directory

## Input Modalities and Dataset Downloading
  * This repo supports 2D skeletons and rgb images as inputs to the model 
  * 2D skeletons: generated by using openpose (referring: https://github.com/CMU-Perceptual-Computing-Lab/openpose)
  * RGB images are obtained directly from the official datasets, refere this link to download: https://www.kaggle.com/datasets/akshayjain22/n-ucla/data 
  * After downloading the dataset, change the data path in 'configurations.py': 
    ```
    input_data_path = 'path/of/your/dataset'
    ```


## Experiment Results
  
  | Architecture name | Sampling| Description | Accuracy |
  | --- | --- | --- | --- |
  | DIR | Multi | DIR stream only, baseline experiment, classifier is not trained with constrastive learning | 92.89 % (reproduced)| 
  | CL-DIR | Multi | DIR stream only, classifier is trained with constrastive learning  | 96.12 %(reproduced) |
  | CL-DIR + CL-CIR | Multi | 2 stream pipeline, both DIR and CIR stream are trained with constrastive learning | 99.3% |

## Training steps
 * All default hypter-parameters and paths are defined in 'configurations.py'
 * Firstly, modify the 'mode' and 'sampling' accordingly
    ```
    mode = 'DIR_CL' for DIR stream only, mode = '2stream_CL' for 2-stream pipeline
    sampling = 'Single' for single clip, sampling = 'Multi' for multi clips
    ```

  * Second, modify the path of saving your models
    ```
    save_model_path = '/path/to/save/model'
    ```
  * Then, running
    ```
    python trainClassifier_CV.py
    ```

## Inference steps
 * All default hypter-parameters and paths are defined in 'configurations.py'
 * We provided the best checkpoints each of experiment
 
  | Architecture name | Sampling | checkpoint path | Usage |
  | --- | --- | --- | --- | 
  | DIR Stream(DIR) | Multi | checkpoints/NUCLA/Single/ckpt_dir.pth | for DIR stream only, no constrastive learning step |
  | DIR Stream(CL-DIR) | Multi | checkpoints/NUCLA/Multi/ckpt_dir_cl.pth | for DIR stream only, with constrastive learning |
  | 2 Stream(CL-DIR + CL-CIR) | Multi | checkpoints/NUCLA/Multi/ckpt_2stream_cl.pth | for 2Stream, with constrastive learning |

 * Firstly, in 'configurations.py', replace 'model_path' to your own path and choose the model you want to apply 
    ```
    model_path = "/path/to/checkpoint/" + "xxx.pth"
    ```
    We provided our best checkpoints under 'checkpoints/', applying those models to reproduce results showing in the above table. 
 * Then, runing
   ```
   python testClassifier_CV.py
   ```


## Other Versions
  * 01: https://github.com/Yuexiaoxi10/Cross_view_actionRecognition
      * This is repo use regular DYAN as sparsecode generator, baseline experiments can be found here. (Experiment configuration is 'setup1', 'single clip')
  * Other datasets and experiments: https://github.com/DanLuoNEU/CVARDIF (Complete pipeline, and 3D input skeleton)
    


  




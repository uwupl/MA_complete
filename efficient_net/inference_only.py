from efficientad import train_transform, test, quantize_model, config_helper
from tqdm import tqdm
from time import perf_counter
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_auc_score
import platform

if __name__ == '__main__':
    import sys
    sys.path.append('/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_complete')
    # from path_definitions import MVTEC_DIR
    from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
        ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
else:
    from .common import get_autoencoder, get_pdn_small, get_pdn_medium, \
        ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader

# constants
seed = 42
on_gpu = torch.cuda.is_available()
on_gpu_init = on_gpu#.copy()
out_channels = 384
image_size = 256
raspberry_pi = False if platform.machine().__contains__('x86') else True
if raspberry_pi:
    torch.backends.quantized.engine = 'qnnpack'
else:
    torch.backends.quantized.engine = 'fbgemm'
    
def main():
    '''
    Performs inference on the test set of the specified dataset and subdataset.
    Loads quantized (!) teacher, student and autoencoder models from the specified weights. TODO not up to date
    '''

    from efficientad import config
    config.measure_inference_time = True
    config.subdataset = 'cable'
    
    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    # test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
    #                             config.dataset, config.subdataset, 'test')
    if raspberry_pi:
        model_dir = os.path.join(config.model_base_dir, #config.output_dir, 'trainings', config.dataset,
                                    config.subdataset)
    else:
        model_dir = config.model_base_dir
    # /mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_complete/results/efficientned_ad/trainings/mvtec_ad/screw
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))

    # create models - default models which means that the actual weights do not matter
    # the weights are loaded later, we just need the model architecture
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception('Unknown config.model_size')
    
    autoencoder = get_autoencoder(out_channels)
    
    # teacher, student, autoencoder = quantize_model(teacher, student, autoencoder, calibration_loader=None)
    
    # load weights
    phase = 'final' # or 'final'

    teacher = torch.load(os.path.join(model_dir, f'teacher_{phase}.pth'), map_location=torch.device('cpu'))
    student = torch.load(os.path.join(model_dir, f'student_{phase}.pth'), map_location=torch.device('cpu'))
    autoencoder = torch.load(os.path.join(model_dir, f'autoencoder_{phase}.pth'), map_location=torch.device('cpu'))
        # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))
    # if config.dataset == 'mvtec_ad':
    # mvtec dataset paper recommend 10% validation set
    train_size = int(0.90 * len(full_train_set)) # edited from 0.9 to 0.98
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(seed) # random number generator
    _, validation_set = torch.utils.data.random_split(full_train_set,
                                                    [train_size,
                                                        validation_size],
                                                    rng)
    
    validation_loader = DataLoader(validation_set, batch_size=1)
    
    t_0 = perf_counter()
    print('Quantizing models...')
    teacher_q, student_q, autoencoder_q = quantize_model(teacher, student, autoencoder, calibration_loader=validation_loader, backend=config.backend)
    print(teacher)
    out = teacher(torch.randn(1, 3, 256, 256))
    t_1 = perf_counter()
    print(f'Quantization took {t_1 - t_0} seconds')

    # import json
    if raspberry_pi:
        
        this_dir = model_dir
    else:
        this_dir = os.path.join(config.output_dir, config.run_id, 'model_statistics')
    import json
    with open(os.path.join(this_dir, f'statistics_{phase}.json'), 'rb') as f: # _{phase}
        statistics = json.load(f)#, allow_pickle=True)
   
    teacher_mean = torch.tensor(statistics['teacher_mean'])
    teacher_std = torch.tensor(statistics['teacher_std'])
    q_st_start = torch.tensor(statistics['q_st_start'])
    q_st_end = torch.tensor(statistics['q_st_end'])
    q_ae_start = torch.tensor(statistics['q_ae_start'])
    q_ae_end = torch.tensor(statistics['q_ae_end'])

    teacher_q_mean = torch.tensor(statistics['teacher_q_mean'])
    teacher_q_std = torch.tensor(statistics['teacher_q_std'])
    q_st_start_q = torch.tensor(statistics['q_st_start_q'])
    q_st_end_q = torch.tensor(statistics['q_st_end_q'])
    q_ae_start_q = torch.tensor(statistics['q_ae_start_q'])
    q_ae_end_q = torch.tensor(statistics['q_ae_end_q'])

    auc_train = statistics['auc']
    auc_train_q = statistics['auc_q']
    # quantized models
    t_0 = perf_counter()
    auc_q, teacher_inference_q, student_inference_q, autoencoder_inference_q, map_normalization_inference_q = test(
        test_set=test_set, teacher=teacher_q, student=student_q,
        autoencoder=autoencoder_q, teacher_mean=teacher_q_mean,
        teacher_std=teacher_q_std, q_st_start=q_st_start,
        q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=None, desc='Intermediate inference',
        q_flag=True)
    t_1 = perf_counter()
    print(f'Inference took {t_1 - t_0} seconds')
    # fp32 models
    t_0 = perf_counter()
    auc, teacher_inference, student_inference, autoencoder_inference, map_normalization_inference = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start,
        q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=None, desc='Intermediate inference',
        q_flag=False)
    # auc, teacher_inference, student_inference, autoencoder_inference, map_normalization_inference = 0, 0, 0, 0, 0 # tmp
    t_1 = perf_counter()
    print(f'Inference took {t_1 - t_0} seconds')
    
    print(f'auc: {auc} %')
    print(f'auc_q: {auc_q} %')
    print(f'auc_train: {auc_train} %')
    print(f'auc_train_q: {auc_train_q} %')
    print(f'auc_q - auc_train: {auc_q - auc_train} %')
    
    print(f'\nteacher_inference: {teacher_inference*1e3} ms')
    print(f'teacher_inference_q: {teacher_inference_q*1e3} ms')
    print(f'student_inference: {student_inference*1e3} ms')
    print(f'student_inference_q: {student_inference_q*1e3} ms')
    print(f'autoencoder_inference: {autoencoder_inference*1e3} ms')
    print(f'autoencoder_inference_q: {autoencoder_inference_q*1e3} ms')
    print(f'map_normalization_inference: {map_normalization_inference*1e3} ms')
    print(f'map_normalization_inference_q: {map_normalization_inference_q*1e3} ms')
    
    # save runtimes as json
    import json
    with open(os.path.join(config.output_dir, f'runtime_{phase}.json'), 'w') as f:
        json.dump({'teacher_inference': teacher_inference*1e3,
                   'teacher_inference_q': teacher_inference_q*1e3,
                   'student_inference': student_inference*1e3,
                   'student_inference_q': student_inference_q*1e3,
                   'autoencoder_inference': autoencoder_inference*1e3,
                   'autoencoder_inference_q': autoencoder_inference_q*1e3,
                   'map_normalization_inference': map_normalization_inference*1e3,
                   'map_normalization_inference_q': map_normalization_inference_q*1e3}, f)

if __name__ == '__main__':
    main()
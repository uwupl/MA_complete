from efficientad import train_transform, test, quantize_model
from tqdm import tqdm
from time import perf_counter
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    import sys
    sys.path.append('/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_complete')
    from path_definitions import MVTEC_DIR
    from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
        ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
else:
    from .common import get_autoencoder, get_pdn_small, get_pdn_medium, \
        ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='own',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_complete/results/efficientned_ad')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default=MVTEC_DIR,
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    
# constants
seed = 42
on_gpu = torch.cuda.is_available()
on_gpu_init = on_gpu#.copy()
out_channels = 384
image_size = 256

def main():
    '''
    Performs inference on the test set of the specified dataset and subdataset.
    Loads quantized (!) teacher, student and autoencoder models from the specified weights.
    '''
    config = get_argparse()
    
    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                config.dataset, config.subdataset, 'test')

    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))

        # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception('Unknown config.model_size')
    
    autoencoder = get_autoencoder(out_channels)
    
    teacher, student, autoencoder = quantize_model(teacher, student, autoencoder, calibrate=False)
    
    # load weights
    teacher.load_state_dict(torch.load(config.weights))
    student.load_state_dict(torch.load(config.weights))
    autoencoder.load_state_dict(torch.load(config.weights))
    
    # inference
    auc_q = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start,
        q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=None, desc='Intermediate inference',
        q_flag=True)
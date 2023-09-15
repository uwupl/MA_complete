#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from time import perf_counter
import json

from sklearn.metrics import roc_auc_score
import sys
sys.path.append('/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_complete')
from path_definitions import MVTEC_DIR
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
# else:
#     from .common import get_autoencoder, get_pdn_small, get_pdn_medium, \
#         ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
# from sklearn.metrics import roc_auc_score
# 
# MVTEC_DIR = r'/mnt/crucial/UNI/IIIT_Muen/MA/MVTechAD/'

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='screw',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_complete/results/efficientned_ad')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_complete/efficient_net/models/teacher_small.pth')
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
    parser.add_argument('-t', '--train_steps', type=int, default=5000)
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
on_gpu_init = on_gpu#.copy()
out_channels = 384
image_size = 256
test_interval = 2500

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # create output dir
    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, config.subdataset)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, 'test')
    if not os.path.exists(train_output_dir) and not os.path.exists(test_output_dir):
        os.makedirs(train_output_dir)
        os.makedirs(test_output_dir)

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))
    if config.dataset == 'mvtec_ad':
        # mvtec dataset paper recommend 10% validation set
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed) # random number generator
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                           [train_size,
                                                            validation_size],
                                                           rng)
    elif config.dataset == 'mvtec_loco':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'validation'),
            transform=transforms.Lambda(train_transform))
    else:
        raise Exception('Unknown config.dataset')


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception('Unknown config.model_size')
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader) # TODO: check what exactly is done here

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                 autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)
    tqdm_obj = tqdm(range(config.train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu_init:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
            teacher = teacher.cuda()
            student = student.cuda()
            autoencoder = autoencoder.cuda()    
            
        # compute loss
        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std # normalized output of teacher --> TODO: What is the teacher?
        student_output_st = student(image_st)[:, :out_channels] # take the first half of channels
        distance_st = (teacher_output_st - student_output_st) ** 2 # compute the distance between teacher and student
        d_hard = torch.quantile(distance_st, q=0.999) # compute threshold. This is done in order to avoid the model to learn insignificant differences
        loss_hard = torch.mean(distance_st[distance_st >= d_hard]) # take only the values above the threshold and compute the mean

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae) 
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student(image_ae)[:, out_channels:] # take the second half of channels
        distance_ae = (teacher_output_ae - ae_output)**2 # compute the distance between 
        distance_stae = (ae_output - student_output_ae)**2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_total = loss_st + loss_ae + loss_stae
        # optimizer step
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description(
                "Current loss: {:.4f}  ".format(loss_total.item()))
        
        # intermediate evaluation
        if iteration % test_interval == 0 and iteration > 0:
            calibrate_eval_save(teacher, student, autoencoder, teacher_mean, teacher_std, train_loader, validation_loader, test_set, train_output_dir, phase = 'tmp')
    # final evaluation
    calibrate_eval_save(teacher, student, autoencoder, teacher_mean, teacher_std, train_loader, validation_loader, test_set, train_output_dir, phase = 'final')
        
    
def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference', q_flag=False):
    y_true = []
    y_score = []
    if not q_flag:
        on_gpu = True if next(student.parameters()).is_cuda else False
    else:
        on_gpu = False
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    device = teacher_output.device
    teacher_mean, teacher_std = teacher_mean.to(device), teacher_std.to(device)
    q_st_start = q_st_start.to(device) if q_st_start is not None else None
    q_st_end = q_st_end.to(device) if q_st_end is not None else None
    q_ae_start = q_ae_start.to(device) if q_ae_start is not None else None
    q_ae_end = q_ae_end.to(device) if q_ae_end is not None else None
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization', q_flag=False):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    if not q_flag:
        on_gpu = True if next(student.parameters()).is_cuda else False
    else:
        on_gpu = False
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader, q_flag=False):

    mean_outputs = []
    if not q_flag:
        on_gpu = True if next(teacher.parameters()).is_cuda else False
    else:
        on_gpu = False
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


### Quantization section ###

class RandomImageDataset(torch.utils.data.Dataset):
    def __init__(self, num_images, transform=None, image_size=224):
        self.num_images = num_images
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Generate a random image
        image = np.random.randint(0, 256, size=(self.image_size, self.image_size, 3), dtype=np.uint8)

        # Convert numpy array to PIL image
        image = transforms.ToPILImage()(image)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, 0, 0, 0, 0

def quantize_model(teacher, student, autoencoder, calibration_loader=None):
    import torch.quantization as tq
    import torch.ao.quantization as taoq
    
    fuse_list_teacher_student = [('0','1'),('3','4'),('6','7')]
    fuse_list_autoencoder = [('0','1'),('2','3'),('4','5'),('6','7'),('8','9'),('12','13'),('16','17'),('20','21'),('24','25'),('28','29'),('32','33'),('36','37')]
    
    teacher, student, autoencoder = teacher.to('cpu'), student.to('cpu'), autoencoder.to('cpu')
    
    teacher, student, autoencoder = tq.fuse_modules(teacher, fuse_list_teacher_student), tq.fuse_modules(student, fuse_list_teacher_student), tq.fuse_modules(autoencoder, fuse_list_autoencoder)   
    
    teacher, student, autoencoder = taoq.QuantWrapper(teacher), taoq.QuantWrapper(student), taoq.QuantWrapper(autoencoder)
    
    teacher.qconfig = tq.get_default_qconfig('fbgemm')
    student.qconfig = tq.get_default_qconfig('fbgemm')
    autoencoder.qconfig = tq.get_default_qconfig('fbgemm')
    
    tq.prepare(teacher, inplace=True)
    tq.prepare(student, inplace=True)
    tq.prepare(autoencoder, inplace=True)
    
    if calibration_loader is not None:
        def calibrate_model(model, loader, calib_item_idx=[0]):
            for idx in calib_item_idx:
                with torch.inference_mode():
                    for inputs in loader:
                        x = inputs[idx]
                        _ = model(x)

        # from PIL import Image
        # data_transforms = transforms.Compose([
        #             transforms.Resize((256, 256), Image.LANCZOS),
        #             transforms.ToTensor(),
        #             transforms.CenterCrop(256),
        #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])]) # from imagenet
        
        calibrate_model(teacher, calibration_loader, [0])
        calibrate_model(student, calibration_loader, [0,1])
        calibrate_model(autoencoder, calibration_loader, [1])
    
    teacher = tq.convert(teacher, inplace=True)
    student = tq.convert(student, inplace=True)
    autoencoder = tq.convert(autoencoder, inplace=True)
    
    return teacher.eval(), student.eval(), autoencoder.eval()

@torch.no_grad()  
def calibrate_eval_save(teacher, student, autoencoder, teacher_mean, teacher_std, train_loader, validation_loader, test_set, train_output_dir, phase = 'tmp'):

    # run intermediate evaluation
    teacher.eval()
    student.eval()
    autoencoder.eval()
    
    torch.save(teacher, os.path.join(train_output_dir,
                                        f'teacher_{phase}.pth'))
    torch.save(student, os.path.join(train_output_dir,
                                        f'student_{phase}.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir,
                                        f'autoencoder_{phase}.pth'))

    print('Quantizing models...')
    st = perf_counter()
    teacher_q, student_q, autoencoder_q = quantize_model(teacher, student, autoencoder, calibration_loader=validation_loader)
    print(f'Quantization took {round(perf_counter() - st,2)} seconds')
    
    teacher_q_mean, teacher_q_std = teacher_normalization(teacher_q, train_loader, q_flag=True)

    torch.save(teacher_q.state_dict(), os.path.join(train_output_dir,
                                        f'teacher_q_{phase}.pth'))
    torch.save(student_q.state_dict(), os.path.join(train_output_dir,
                                        f'student_q_{phase}.pth'))
    torch.save(autoencoder_q.state_dict(), os.path.join(train_output_dir,
                                        f'autoencoder_q_{phase}.pth'))
    
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization( # only done with non quantized model
        validation_loader=validation_loader, teacher=teacher,
        student=student, autoencoder=autoencoder,
        teacher_mean=teacher_mean, teacher_std=teacher_std,
        desc='Intermediate map normalization')
    
    q_st_start_q, q_st_end_q, q_ae_start_q, q_ae_end_q = map_normalization( # only done with quantized model
        validation_loader=validation_loader, teacher=teacher_q,
        student=student_q, autoencoder=autoencoder_q,
        teacher_mean=teacher_q_mean, teacher_std=teacher_q_std,
        desc='Intermediate map normalization', q_flag=True)

    print(q_st_start, q_st_end, q_ae_start, q_ae_end)
    print(q_st_start_q, q_st_end_q, q_ae_start_q, q_ae_end_q)
    # non quantized model
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start,
        q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=None, desc='Intermediate inference')
    print('Intermediate image auc - non quantized: {:.4f}'.format(auc))
    # quantized model
    auc_q_1 = test(
        test_set=test_set, teacher=teacher_q, student=student_q,
        autoencoder=autoencoder_q, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start,
        q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=None, desc='Intermediate inference',
        q_flag=True)
    print('Intermediate image auc - quantized (just models): {:.4f}'.format(auc_q_1))

    auc_q_2 = test(
        test_set=test_set, teacher=teacher_q, student=student_q,
        autoencoder=autoencoder_q, teacher_mean=teacher_q_mean,
        teacher_std=teacher_q_std, q_st_start=q_st_start_q,
        q_st_end=q_st_end_q, q_ae_start=q_ae_start_q,
        q_ae_end=q_ae_end_q, test_output_dir=None,
        desc='Intermediate inference', q_flag=True)
    print('Intermediate image auc - quantized (with map norm and teacher mean/std): {:.4f}'.format(auc_q_2))

    auc_q_3 = test(
        test_set=test_set, teacher=teacher_q, student=student_q,
        autoencoder=autoencoder_q, teacher_mean=teacher_q_mean,
        teacher_std=teacher_q_std, q_st_start=q_st_start,
        q_st_end=q_st_end, q_ae_start=q_ae_start,
        q_ae_end=q_ae_end, test_output_dir=None,
        desc='Intermediate inference', q_flag=True)
    print('Intermediate image auc - quantized (with teacher mean/std): {:.4f}'.format(auc_q_3))

    auc_q_4 = test(
        test_set=test_set, teacher=teacher_q, student=student_q,
        autoencoder=autoencoder_q, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start_q,
        q_st_end=q_st_end_q, q_ae_start=q_ae_start_q,
        q_ae_end=q_ae_end_q, test_output_dir=None,
        desc='Intermediate inference', q_flag=True)
    print('Intermediate image auc - quantized (with map norm): {:.4f}'.format(auc_q_4))
    
    # save statistics
    statistics = {
        'q_st_start': q_st_start.item(),
        'q_st_end': q_st_end.item(),
        'q_ae_start': q_ae_start.item(),
        'q_ae_end': q_ae_end.item(),
        'q_st_start_q': q_st_start_q.item(),
        'q_st_end_q': q_st_end_q.item(),
        'q_ae_start_q': q_ae_start_q.item(),
        'q_ae_end_q': q_ae_end_q.item(),
        'teacher_mean': teacher_mean.cpu().tolist(),
        'teacher_std': teacher_std.cpu().tolist(),
        'teacher_q_mean': teacher_q_mean.cpu().tolist(),
        'teacher_q_std': teacher_q_std.cpu().tolist(),
        'auc': auc,
        'auc_q': auc_q_3
    }
    with open(os.path.join(train_output_dir, f'statistics_{phase}.json'), 'w') as f:
        json.dump(statistics, f)

    # teacher frozen
            


if __name__ == '__main__':
    main()




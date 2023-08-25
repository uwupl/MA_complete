# import logging
from PIL import Image
import os
import pickle
from collections import OrderedDict
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import utils.common as common
from utils.utils import modified_kNN_score_calc
import utils.metrics as metrics
from utils.backbone import Backbone
from utils.embedding import _embed, _feature_extraction, PatchMaker
import os
from sklearn.metrics import roc_auc_score
from path_definitions import ROOT_DIR, RES_DIR, PLOT_DIR, MVTEC_DIR, EMBEDDING_DIR
from utils.datasets import MVTecDataset
from torch.utils.data import DataLoader
from time import perf_counter


def init_weight(m):
    '''
    Used to initialize the weights of the discriminator and projection networks.
    '''
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden # which is alsoe by default None ???
        print('Hidden is: ', _hidden)
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
            print('Hidden is: ', _hidden)
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    '''
    TODO
    '''    
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):
        
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x

class SimpleNet(torch.nn.Module):
    def __init__(self, device):
        """anomaly detection class."""
        super(SimpleNet, self).__init__()
        self.device = device
        self.input_shape = (3,224,224)
        self.only_img_lvl = True
        self.measure_inference = True
        # Backbone
        self.backbone_id = 'RN18' # analogous to model_id
        self.layers_to_extract_from = [2,3] # analogous to layers_needed
        self.quantize_qint8 = True
        if self.quantize_qint8:
            self.device = 'cpu'
            self.calibration_dataset = 'random'
            self.cpu_arch = 'x86'
            self.num_images_calib = 100
        # Embedding
        self.pretrain_embed_dimensions = 256 + 128 # for RN18
        self.target_embed_dimensions = 128 + 256 # for RN18
        self.patch_size = 3
        self.patch_stride = 1
        self.embedding_size = None # TODO --> What does that do?
        # Projection
        self.pre_proj = 1 # TODO
        self.proj_layer_type = 0 # TODO
        # Discriminator
        self.dsc_layers = 2
        self.dsc_hidden = int(self.target_embed_dimensions*0.75)#1024
        self.dsc_margin = 0.5 # TODO
        # Noise
        self.noise_std = 0.015
        self.auto_noise = [0, None] # TODO
        self.noise_type = 'GAU'
        self.mix_noise = 1 # TODO --> probably just the number of classes of noise. Usually one
        # Training
        self.meta_epochs = 40
        self.aed_meta_epochs = 0 # TODO
        self.gan_epochs = 4 # TODO
        self.batch_size = 8
        self.dsc_lr = 1e-5
        self.proj_lr = 1e-3
        self.lr_scheduler = True
        self.num_workers = 12
        # Scoring 
        self.adapted_score_calc = True # TODO
        self.top_k = 3
        self.batch_size_test = 8
        # Directory
        self.model_dir = r'/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_complete/results/simplenet'
        self.run_id = 'none'
        self.category = 'pill'
        self.dataset_path = MVTEC_DIR
        # Data transforms
        self.load_size = 256
        self.input_size = 224
        self.category_wise_statistics = False
        if self.category_wise_statistics:
            filename = 'statistics.json'
            with open(filename, "rb") as file:
                loaded_dict = pickle.load(file)
            statistics_of_categories = loaded_dict[self.category]
            means = statistics_of_categories['means']
            stds = statistics_of_categories['stds']
        else:
            means = [0.485, 0.456, 0.406] # Imagenet
            stds = [0.229, 0.224, 0.225]
        self.data_transforms = transforms.Compose([
                transforms.Resize((self.load_size, self.load_size), Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.CenterCrop(self.input_size),
                transforms.Normalize(mean=means,
                                    std=stds)]) # from imagenet  # for each category calculate mean and std TODO

        self.gt_transforms = transforms.Compose([
                        transforms.Resize((self.load_size, self.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(self.input_size)])
        self.inv_normalize = transforms.Normalize(mean=list(np.divide(np.multiply(-1,means), stds)), std=list(np.divide(1, stds))) 
        
        # initialize the network
        self.forward_modules = torch.nn.ModuleDict({})
        self.backbone = Backbone(model_id=self.backbone_id, layers_needed=self.layers_to_extract_from, layer_cut=True, quantize_qint8_prepared=self.quantize_qint8).to(self.device)
        feature_dimension = self.backbone.feature_dim
        if self.quantize_qint8:
            from utils.quantize import quantize_model_into_qint8
            self.backbone = quantize_model_into_qint8(model=self.backbone, layers_needed=self.layers_to_extract_from, calibrate=self.calibration_dataset, cpu_arch=self.cpu_arch, num_images=self.num_images_calib)#, dataset_path=self.dataset_path)
        self.forward_modules['backbone'] = self.backbone
        self.patch_maker = PatchMaker(patchsize=self.patch_size,top_k=self.top_k, stride=self.patch_stride)
        preprocessing = common.Preprocessing(input_dims = feature_dimension, output_dim = self.pretrain_embed_dimensions).to(self.device)
        self.forward_modules['preprocessing'] = preprocessing
        
        preadapt_aggregator = common.Aggregator(target_dim = self.target_embed_dimensions).to(self.device)
        self.forward_modules['preadapt_aggregator'] = preadapt_aggregator
        
        self.anomaly_segmentor = common.RescaleSegmentor(device = self.device, target_size = self.input_shape[1])#.to(self.device)
        
        self.embedding_size = self.embedding_size if self.embedding_size is not None else self.target_embed_dimensions
        
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimensions, self.target_embed_dimensions, self.pre_proj, self.proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), self.proj_lr)
        
        self.discriminator = Discriminator(self.target_embed_dimensions, self.dsc_layers, self.dsc_hidden).to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt, self.meta_epochs, eta_min=self.dsc_lr*.4)
        
    def set_model_dir(self):
        # print("model_dir: ",self.model_dir)
        # self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, self.run_id, self.category)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        # os.makedirs(self.tb_dir, exist_ok=True)
        # self.logger = TBWrapper(self.tb_dir) #SummaryWriter(log_dir=tb_dir)
    
    # def embed(self, data):
    #     if isinstance(data, torch.utils.data.DataLoader):
    #         features = []
    #         for image in data:
    #             if isinstance(image, dict):
    #                 image = image["image"]
    #                 input_image = image.to(torch.float).to(self.device)
    #             with torch.no_grad():
    #                 features.append(_embed(input_image, self.forward_modules, self.patch_maker))
    #         return features
    #     return _embed(data, self.forward_modules, self.patch_maker)
    
    
    def train(self):#, training_data, test_data):
        '''
        main function
        '''
        self.set_model_dir()
        state_dict = {}
        training_data = self.train_dataloader()
        test_data = self.test_dataloader()
        
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                print('No discriminator in state_dict!')
                self.load_state_dict(state_dict, strict=False)

            self.predict(training_data, "train_")
            if not self.measure_inference:
                scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            else:
                scores, segmentations, features, labels_gt, masks_gt, t_1_cpu, t_2_cpu, t_3_cpu = self.predict(test_data)
                print('Time for feature extraction: ', t_1_cpu)
                print('Time for embedding: ', t_2_cpu)
                print('Time for discriminator: ', t_3_cpu)
            auroc, full_pixel_auroc, anomaly_pixel_auroc = self._evaluate(scores, segmentations, features, labels_gt, masks_gt)
            
            return auroc, full_pixel_auroc, anomaly_pixel_auroc
    
        def update_state_dict():
            
            state_dict["discriminator"] = OrderedDict({
                k:v.detach().cpu() 
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k:v.detach().cpu() 
                    for k, v in self.pre_projection.state_dict().items()})
        # actual training loop
        best_record = None
        for i_mepoch in range(self.meta_epochs):

            self._train_discriminator(training_data)
            if not self.only_img_lvl:
                if not self.measure_inference:
                    scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
                else:
                    scores, segmentations, features, labels_gt, masks_gt, t_1_cpu, t_2_cpu, t_3_cpu = self.predict(test_data)
                    print('Time for feature extraction: ', t_1_cpu)
                    print('Time for embedding: ', t_2_cpu)
                    print('Time for discriminator: ', t_3_cpu)
            else:
                if not self.measure_inference:
                    scores,_,_,labels_gt,_ = self.predict(test_data)
                else:
                    scores,_,_,labels_gt,_, t_1_cpu, t_2_cpu, t_3_cpu = self.predict(test_data)
                    print('Time for feature extraction: ', t_1_cpu)
                    print('Time for embedding: ', t_2_cpu)
                    print('Time for discriminator: ', t_3_cpu)
                segmentations, features, masks_gt = None, None, None
            auroc, full_pixel_auroc, pro = self._evaluate(scores, segmentations, features, labels_gt, masks_gt)

            if best_record is None:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict()
            elif auroc > best_record[0]:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict()
                # elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                #     best_record[1] = full_pixel_auroc
                #     best_record[2] = pro 
                #     update_state_dict(state_dict)

            print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                  f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                  f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----")
        
        torch.save(state_dict, ckpt_path)
        
        return best_record
    
    def _evaluate(self, scores, segmentations, features, labels_gt, masks_gt):
        if not self.only_img_lvl:
            print('Total pixel-level auc-roc score:')
            pixel_auc = roc_auc_score(masks_gt, segmentations)
            print(pixel_auc)
        else:
            pixel_auc = 0.0
        img_auc = roc_auc_score(labels_gt, scores)
        print('Total image-level auc-roc score:')
        print(img_auc)

        pro = 0.0
        
        return img_auc, pixel_auc, pro
        
    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval() # so they won't be trained and stay fixed. Mainly the backbone. 
        
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        # LOGGER.info(f"Training discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    i_iter += 1
                    img, _, _, _, _ = data_item#["image"]
                    img = img.to(torch.float).to(self.device)
                    # features = 
                    true_feats = _embed(_feature_extraction(img, self.forward_modules), self.forward_modules, self.patch_maker)
                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(true_feats)
                    # else:
                        # true_feats = _embed(img, self.forward_modules, self.patch_maker)
                    # print('feat', true_feats.shape)
                    noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
                    # print('noise_idxs', noise_idxs.shape)
                    noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device) # (N, K)
                    # print('noise_one_hot', noise_one_hot.shape)
                    noise = torch.stack([
                        torch.normal(0, self.noise_std * 1.1**(k), true_feats.shape)
                        for k in range(self.mix_noise)], dim=1).to(self.device) # (N, K, C)
                    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                    # print('noise', noise.shape)
                    fake_feats = true_feats + noise
                    # print('fake_feats', fake_feats.shape)
                    combined_features = torch.cat([true_feats, fake_feats])
                    scores = self.discriminator(combined_features)
                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(fake_feats):]
                    
                    th = self.dsc_margin
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                    true_loss = torch.clip(-true_scores + th, min=0)
                    fake_loss = torch.clip(fake_scores + th, min=0)

                    # self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                    # self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)

                    loss = true_loss.mean() + fake_loss.mean()
                    # self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    # self.logger.step()

                    loss.backward()
                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    # if self.train_backbone:
                    #     self.backbone_opt.step()
                    self.dsc_opt.step()

                    loss = loss.detach().cpu() 
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())
                
                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)
                
                if self.lr_scheduler:
                    self.dsc_schl.step()
                
                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)


    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        img_paths = []
        scores = []
        labels_gt = []
        
        if not self.only_img_lvl:
            masks = []
            features = []
            masks_gt = []
        
        if self.measure_inference:
            total_fe = []
            total_em = []
            total_sc = []

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            if not self.only_img_lvl:
                for data in data_iterator:
                    image, mask, label, img_path, img_type = data
                    # if isinstance(data, dict):
                    labels_gt.extend(label.numpy().tolist())
                    if mask is not None:
                        masks_gt.extend(mask.numpy().tolist())
                    # image = data["image"]
                    img_paths.extend(img_path)
                    if not self.measure_inference:
                        _scores, _masks, _feats = self._predict(image)
                    else:
                        _scores, _masks, _feats, t_1_cpu, t_2_cpu, t_3_cpu = self._predict(image)
                for score, mask in zip(_scores, _masks, ):
                    scores.append(score)
                    masks.append(mask)
                return scores, masks, features, labels_gt, masks_gt, t_1_cpu, t_2_cpu, t_3_cpu
            else:
                for data in data_iterator:
                    image, mask, label, img_path, img_type = data
                    # if isinstance(data, dict):
                    labels_gt.extend(label.numpy().tolist())
                    # image = image
                    img_paths.extend(img_path)
                    if not self.measure_inference:
                        _scores, _, _, _, _ = self._predict(image)
                    else:
                        _scores, _, _, _, _, t_fe, t_em, t_sc = self._predict(image)
                    for score, time_fe, time_em, time_sc in zip(_scores, time_fe, time_em, time_sc):
                        scores.append(score)
                        total_fe.append(time_fe)
                        total_em.append(time_em)
                        total_sc.append(time_sc)
                    t_fe = np.mean(total_fe)
                    t_em = np.mean(total_em)
                    t_sc = np.mean(total_sc)
                return scores, None, None, labels_gt, None, t_fe, t_em, t_sc
            
    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            # feature extraction
            if not self.measure_inference:

                features = _feature_extraction(images, 
                                            self.forward_modules)
                features, patch_shapes = _embed(features,
                                                self.forward_modules,
                                                self.patch_maker,
                                                provide_patch_shapes=True)#, 
                                                    #evaluation=True)
                if self.pre_proj > 0:
                    features = self.pre_projection(features) #torch.Size([6272, 1536])


                score_patches = -self.discriminator(features) #torch.Size([6272, 1]) --> for each patch one score
                score_patches = score_patches.cpu().numpy()
                image_scores = score_patches.copy()

                image_scores = self.patch_maker.unpatch_scores(
                    image_scores, batchsize=batchsize
                ) # (8, 784, 1)
                # image_scores = image_scores.reshape(*image_scores.shape[:2], -1) # redundant
                image_scores = self.patch_maker.score(image_scores)

                # score_patches = self.patch_maker.unpatch_scores(
                #     score_patches, batchsize=batchsize
                # )
                if not self.only_img_lvl:
                    scales = patch_shapes[0]
                    score_patches = score_patches.reshape(batchsize, scales[0], scales[1])
                    features = features.reshape(batchsize, scales[0], scales[1], -1)
                    masks, features = self.anomaly_segmentor.convert_to_segmentation(score_patches, features)
                    return list(image_scores), list(masks), list(features)
                else:
                    return list(image_scores), None, None, None, None
            else:
                # feature extraction
                t_0_cpu = perf_counter()
                features = _feature_extraction(images, 
                            self.forward_modules)
                # embedding
                t_1_cpu = perf_counter()
                features, patch_shapes = _embed(features,
                                                self.forward_modules,
                                                self.patch_maker,
                                                provide_patch_shapes=True)
                # projection
                if self.pre_proj > 0:
                    features = self.pre_projection(features)
                t_2_cpu = perf_counter()
                # discriminator
                score_patches = -self.discriminator(features)
                score_patches = score_patches.cpu().numpy()
                image_scores = score_patches.copy()

                image_scores = self.patch_maker.unpatch_scores(
                    image_scores, batchsize=batchsize
                )
                image_scores = self.patch_maker.score(image_scores)
                t_3_cpu = perf_counter()
                if not self.only_img_lvl:
                    scales = patch_shapes[0]
                    score_patches = score_patches.reshape(batchsize, scales[0], scales[1])
                    features = features.reshape(batchsize, scales[0], scales[1], -1)
                    masks, features = self.anomaly_segmentor.convert_to_segmentation(score_patches, features)
                    return list(image_scores), list(masks), list(features), t_1_cpu-t_0_cpu, t_2_cpu-t_1_cpu, t_3_cpu-t_2_cpu
                else:
                    return list(image_scores), None, None, None, None, t_1_cpu-t_0_cpu, t_2_cpu-t_1_cpu, t_3_cpu-t_2_cpu



    
    def train_dataloader(self):
        '''
        load training data
        uses attributes to determine which dataset to load
        '''
        image_datasets = MVTecDataset(root=os.path.join(self.dataset_path,self.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='train')#, half=self.quantization)
        train_loader = DataLoader(image_datasets, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader

    def test_dataloader(self):
        '''
        load test data
        uses attributes to determine which dataset to load
        '''
        test_datasets = MVTecDataset(root=os.path.join(self.dataset_path,self.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')#, half=self.quantization)
        test_loader = DataLoader(test_datasets, batch_size=self.batch_size_test, shuffle=False, num_workers=self.num_workers)
        return test_loader
    
    def calc_img_score(self, score_patches):
        '''
        calculates the image score based on score_patches
        '''
        if self.adapted_score_calc:
            score = modified_kNN_score_calc(score_patches=score_patches.astype(np.float64), n_next_patches=self.n_next_patches)
        else:
            if True: # outlier removal
                sum_of_each_patch = np.sum(score_patches,axis=1)
                threshold_val = 50*np.percentile(sum_of_each_patch, 50)
                non_outlier_patches = np.argwhere(sum_of_each_patch < threshold_val).flatten()#[0]
                if len(non_outlier_patches) < score_patches.shape[0]:
                    score_patches = score_patches[non_outlier_patches]
                    print('deleted outliers: ', sum_of_each_patch.shape[0]-len(non_outlier_patches))
            N_b = score_patches[np.argmax(score_patches[:,0])].astype(np.float128) # only the closest val is relevant for selection! # this changes with adapted version.
            w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
            score = w*max(score_patches[:,0]) # Image-level score #TODO --> meaning of numbers
        return score
    
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleNet(device)
    number_of_samples = [1, 10, 100, 1000, 10000]
    dataset_type = ['random', 'imagenet']
    model.category = 'pill'
    for dataset in dataset_type:
        for no_samples in number_of_samples:
            model.num_images_calib = no_samples
            model.calibration_dataset = dataset
            model.run_id = f'calibration_dataset_{dataset}_{no_samples}'
            model.train()#model.train_dataloader(), model.test_dataloader())
            model.train() # equal to test! Model is already trained, so this model is loaded and the inference is done directly
    # model.predict(model.test_d
    


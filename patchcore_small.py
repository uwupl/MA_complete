from utils.backbone import Backbone, prune_naive, prune_model_nni, prune_output_layer, quantize_model, compress_model_nni
from utils.feature_adaptor import get_feature_adaptor, FeatureAdaptor
from utils.datasets import MVTecDataset
from utils.utils import min_max_norm, heatmap_on_image, cvt2heatmap, record_gpu, modified_kNN_score_calc, calc_anomaly_map #  distance_matrix, softmax
from utils.pooling import adaptive_pooling
from utils.embedding import reshape_embedding, embedding_concat_frame
from utils.search import KNN
from utils.quantize import quantize_model_into_qint8
from utils.kcenter_greedy import kCenterGreedy
from path_definitions import ROOT_DIR, RES_DIR, PLOT_DIR, MVTEC_DIR, EMBEDDING_DIR

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import faiss
import pickle
from sklearn.neighbors import NearestNeighbors
import time
from time import perf_counter as record_cpu
from anomalib.models.components.sampling import k_center_greedy
from torchinfo import summary


FIXED_ARGS = {
    'load_size': 256,
    'crop_size': 224
}


class PatchCoreSmall(pl.LightningModule):
    def __init__(self, fixed_args=FIXED_ARGS):
        super(PatchCoreSmall, self).__init__()
        
        # Options
        self.category = 'bottle'
        
        # Fixed Options
        self.load_size = fixed_args['load_size']
        self.input_size = fixed_args['input_size']        
        
        # Feature Extraction
        # Backbone
        self.model_id = 'RN18' # other options are 'RN34' and 'WRN50'
        
# import numpy as np
# # import numba as nb
# from utils.utils import remove_uncomplete_runs, remove_test_dir
# from train_main_patchcore import PatchCore, one_run_of_model
# import pytorch_lightning as pl
# import os
# import torch
# import sys
# import traceback
# import gc
# import time
# from path_definitions import RES_DIR
from utils.testbench_utils import *

            
if __name__ == '__main__':
    
    import warnings
    warnings.filterwarnings("ignore") 

    model = get_default_PatchCoreModel()
    manager = TestContainer()
    manager.total_runs = 2
    run_id_prefix = 'default_model_test_2109'
    manager.this_run_id = run_id_prefix
    model.random_presampling = [False, 250000]
    model.pretrain_embed_dimensions = 1024
    model.target_embed_dimensions = 384
    
    
    model.multiple_coresets = [False, 5]
    # model.pooling_embedding = False
    # model.group_id = run_id_prefix + 'default_1_percent_pooling_embedding'
    # manager.run(model, only_accuracy=False)
    
    model.adapted_score_calc = False
    
    model.own_knn = False
    model.faiss_standard = False
    model.patchcore_scorer = True
    model.patchcore_score_patches = True
    # model.group_id = run_id_prefix + 'default_patchcore-10 percentage-own sampler'
    # model.coreset_sampling_method = 'k_center_greedy'
    # manager.run(model, only_accuracy=True)
    sampling_methods = ['random_selection', 'k_center_greedy', 'patchcore_greedy_approx']#, 'patchcore_greedy_exact']
    for sampling_method in sampling_methods:
        
                
        model.coreset_sampling_method = sampling_method
        model.coreset_sampling_ratio = 0.01
        model.pooling_embedding = False
        model.group_id = run_id_prefix + 'default-' + sampling_method + '-patchify-1 percent'
        manager.run(model, only_accuracy=True)
        model.pooling_embedding = True
        model.group_id = run_id_prefix + 'default-' + sampling_method + '-pooling embedding-1 percent'
        manager.run(model, only_accuracy=True)

        if not sampling_method == 'k_centet_greedy':
            model.coreset_sampling_ratio = 0.1
            model.pooling_embedding = False
            model.group_id = run_id_prefix + 'default-' + sampling_method + '-patchify-10 percent'
            manager.run(model, only_accuracy=True)
            model.pooling_embedding = True
            model.group_id = run_id_prefix + 'default-' + sampling_method + '-pooling embedding-10 percent'
            manager.run(model, only_accuracy=True)
        
        
        model.specific_number_of_examples = 1000
        model.pooling_embedding = False
        model.group_id = run_id_prefix + 'default-' + sampling_method + '-patchify-1000 Samples'
        manager.run(model, only_accuracy=True)
        model.pooling_embedding = True
        model.group_id = run_id_prefix + 'default-' + sampling_method + '-pooling embedding-1000 Samples'
        manager.run(model, only_accuracy=True)

    
    # model.own_knn = True
    # model.faiss_standard = False
    # model.group_id = run_id_prefix + 'default_1_percent_pooling_with_patchify-embed_dim_384-faiss-adapted_score_calc'
    # manager.run(model, only_accuracy=True)
    # # model.backbone_id = 'RN18'
    # model.adapted_score_calc = True
    # model.n_neighbors = 4
    # model.n_next_patches = 16
    # model.layer_cut = True
    # model.layers_needed = [2,3]
    # run_id_prefix = 'Backbone_Comparison_1909_2'
    # model.quantize_qint8 = False

    # model.pooling_embedding = True
    # model.calibration_dataset = 'random'
    # model.num_images_calib = 10
    # model.warm_up_reps = 1
    # model.number_of_reps = 3
    # model.own_knn = False
    # model.faiss_standard = True
    # model.multiple_coresets = [True, 5]
    # model.specific_number_of_examples = 1000
    # model.adapt_feature = False
    # model.measure_inference = True
    # # model.group_id = run_id_prefix + 'RN18_qint8-pooling_embedding-fa'
    # # model.backbone_id = 'RN18'
    # backbones = ['WRN50', 'RN34', 'RN18', 'pdn_small', 'pdn_medium']
    
    # for backbone in backbones:
    #     model.backbone_id = backbone
    #     model.group_id = run_id_prefix + f'-{backbone}'
    #     manager.run(model, only_accuracy=False)
    
    # model.pooling_embedding = False
    # model.group_id = run_id_prefix + 'RN18_qint8-pooling_with_patchify'
    # manager.run(model, only_accuracy=False)
    # manager.this_run_id = run_id_prefix
    # for dataset in ['imagenet', 'random']:
    #     for num_images in [1,10,100,1000]:
    #         model.calibration_dataset = dataset
    #         model.num_images_calib = num_images
    #         model.group_id = run_id_prefix + 'RN18_qint8-calib-' + dataset + '-#' + str(num_images)
    #         manager.run(model, only_accuracy=True)

    
    
    # default run
    # manager.this_run_id = run_id_prefix 
    # model.group_id = run_id_prefix + 'RN18_fp32-default'
    # # manager.run(model, only_accuracy=True)
    
    
    # ##############################
    # # define test loop here
    # ##############################
    
    # model.quantize_qint8 = True
    # model.calibration_dataset = 'imagenet'
    # model.group_id = run_id_prefix + 'RN18_qint8-calib_imagenet'
    # manager.run(model, only_accuracy=True)
    
    # model.calibration_dataset = 'target'
    # model.group_id = run_id_prefix + 'RN18_qint8-calib_target'
    # manager.run(model, only_accuracy=True)
    
    # model.calibration_dataset = 'mvtec'
    # model.group_id = run_id_prefix + 'RN18_qint8-calib_mvtec'
    # manager.run(model, only_accuracy=True)
    
    # model.calibration_dataset = 'random'
    # model.group_id = run_id_prefix + 'RN18_qint8-calib_random'
    # manager.run(model, only_accuracy=True)
    
    # model.calibration_dataset = None
    # model.group_id = run_id_prefix + 'RN18_qint8-calib_none'
    # manager.run(model, only_accuracy=True)    
    
    print(manager.get_summarization())

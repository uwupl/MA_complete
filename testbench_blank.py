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
    manager.total_runs = 6
    model.model_id = 'RN18'
    model.adapted_score_calc = True
    model.n_neighbors = 4
    model.n_next_patches = 16
    model.layer_cut = True
    model.layers_needed = [2,3]
    run_id_prefix = 'Calib_Comp'
    
    
    # default run
    manager.this_run_id = run_id_prefix 
    model.group_id = run_id_prefix + 'RN18_fp32-default'
    # manager.run(model, only_accuracy=True)
    
    
    ##############################
    # define test loop here
    ##############################
    
    model.quantize_qint8 = True
    model.calibration_dataset = 'imagenet'
    model.group_id = run_id_prefix + 'RN18_qint8-calib_imagenet'
    manager.run(model, only_accuracy=True)
    
    model.calibration_dataset = 'target'
    model.group_id = run_id_prefix + 'RN18_qint8-calib_target'
    manager.run(model, only_accuracy=True)
    
    model.calibration_dataset = 'mvtec'
    model.group_id = run_id_prefix + 'RN18_qint8-calib_mvtec'
    manager.run(model, only_accuracy=True)
    
    model.calibration_dataset = 'random'
    model.group_id = run_id_prefix + 'RN18_qint8-calib_random'
    manager.run(model, only_accuracy=True)
    
    model.calibration_dataset = None
    model.group_id = run_id_prefix + 'RN18_qint8-calib_none'
    manager.run(model, only_accuracy=True)    
    
    print(manager.get_summarization())

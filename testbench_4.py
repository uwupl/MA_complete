from utils.testbench_utils import *

            
if __name__ == '__main__':
    
    import warnings
    warnings.filterwarnings("ignore") 

    model = get_default_PatchCoreModel()
    layers_needed_list = [[2], [3], [4], [1], [2,3], [1,2], [1,3], [1,2,3]] 
    # layers_needed_list = [[1,3]]
    
    manager = TestContainer()
    run_id_prefix = 'Slide_4'
    manager.this_run_id = run_id_prefix
    model.multiple_coresets = [True, 5]
    model.specific_number_of_examples = 1000
    model.pooling_embedding = True
    model.backbone_id = 'CX_XS'
    for ln in layers_needed_list:
        model.layers_needed = ln
        model.group_id = run_id_prefix + model.backbone_id + str(ln)
        manager.run(model, only_accuracy=False)
    
    model.backbone_id = 'CX_S'
    for ln in layers_needed_list:
        model.layers_needed = ln
        model.group_id = run_id_prefix + model.backbone_id + str(ln)
        manager.run(model, only_accuracy=False)
    
    model.backbone_id = 'CX_M'
    for ln in layers_needed_list:
        model.layers_needed = ln
        model.group_id = run_id_prefix + model.backbone_id + str(ln)
        manager.run(model, only_accuracy=False)
    
    model.backbone_id = 'CX_L'
    for ln in layers_needed_list:
        model.layers_needed = ln
        model.group_id = run_id_prefix + model.backbone_id + str(ln)
        manager.run(model, only_accuracy=False)
        
    # model.backbone_id = 'pdn_small'
    # model.group_id = run_id_prefix + model.backbone_id
    # manager.run(model, only_accuracy=False)
    
    # model.backbone_id = 'pdn_medium'
    # model.group_id = run_id_prefix + model.backbone_id
    # manager.run(model, only_accuracy=False)
    
    print(manager.get_summarization())
    
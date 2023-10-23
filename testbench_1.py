from utils.testbench_utils import *

            
if __name__ == '__main__':
    
    import warnings
    warnings.filterwarnings("ignore") 

    model = get_default_PatchCoreModel()
    
    manager = TestContainer()
    run_id_prefix = 'Slide_1'
    manager.this_run_id = run_id_prefix
    model.multiple_coresets = [False, 5]
    model.target_embed_dimensions = 1024
    model.pretrain_embed_dimensions = 1024
    
    model.group_id = run_id_prefix + r'100%'
    model.coreset_sampling_ratio = 1.0
    manager.run(model, only_accuracy=False) # not done yet, multiple coreset has to be false
    
    model.multiple_coresets[0] = True
    model.group_id = run_id_prefix + r'10%'
    model.coreset_sampling_ratio = 0.1
    manager.run(model, only_accuracy=False)
    
    model.group_id = run_id_prefix + r'1%'
    model.coreset_sampling_ratio = 0.01
    manager.run(model, only_accuracy=False)
    
    model.group_id = run_id_prefix + r'0.1%'
    model.coreset_sampling_ratio = 0.001
    manager.run(model, only_accuracy=False)
    
    model.specific_number_of_examples = 1000
    model.group_id = run_id_prefix + r'#1000 (MW=0.6%)'
    manager.run(model, only_accuracy=False)
    
    print(manager.get_summarization())
from utils.testbench_utils import *

            
if __name__ == '__main__':
    
    import warnings
    warnings.filterwarnings("ignore") 

    model = get_default_PatchCoreModel()
    
    manager = TestContainer()
    run_id_prefix = 'Slide_2'
    manager.this_run_id = run_id_prefix
    model.multiple_coresets = [True, 5]
    model.specific_number_of_examples = 1000
    model.target_embed_dimensions = 384
    model.pretrain_embed_dimensions = 384
    model.group_id = run_id_prefix + r'PC-384-FAISS'
    manager.run(model, only_accuracy=False)
    
    model.n_neighbors = 9
    model.patchcore_score_patches = False
    model.own_knn = True
    model.patchcore_scorer = False
    model.group_id = run_id_prefix + r'PC-384-cdist'
    manager.run(model, only_accuracy=False)
    
    model.patchcore_scorer = True
    model.patchcore_score_patches = True
    model.own_knn = False
    model.pooling_embedding = True
    model.group_id = run_id_prefix + r'Einfaches Pooling-1536-FAISS'
    manager.run(model, only_accuracy=False)
    
    model.patchcore_scorer = False
    model.own_knn = True
    model.patchcore_score_patches = False
    model.pooling_embedding = True
    model.group_id = run_id_prefix + r'Einfaches Pooling-1536-cdist'
    manager.run(model, only_accuracy=False)
    
    
    
    print(manager.get_summarization())
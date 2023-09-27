from utils.utils import plot_results, get_plot_ready_data
res_path = r'/mnt/crucial/UNI/IIIT_Muen/MA/code/productive/MA_complete/results/'

this_run_id = '_Baseline_Slide_3_'

to_contain = []#'std + entropy']#'by_0.3']#'with_sigmoid_weight_by_entropy', 'reduce_via_entropy']#'RN18', 'layer 3']#'(not adapted)', 'avg311', '-RN34-L_2_normalized-entropy_normed-', 'reduced_by_95']
to_delete = []#'1_']#'std + entropy']#'with_1_promille', '(adapted)']
title = r'Manipulating the Output Layer (ResNet 18, Layer 2)'#Iterative Pruning utilizing L2 norm: 3% overall pruning after 2% of all channels left removed'#unstructured_prune_l1_RN34_L2'#1_percentage_layer_2_ResNet34_search_default'
plot_results(
    *get_plot_ready_data(this_run_id, res_path, to_contain, to_delete),# take_n_best=8),
    fig_size=(10,6),
    title=title,
    only_auc=False,
    width=0.4,
    save_fig=False,
    show_f_length=False,
    show_storage=False,
    loc_legend=(1.25,0.04),
    font_scaler=0.5,
    res_path=res_path,
    show=True,
    )

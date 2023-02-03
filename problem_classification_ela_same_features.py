import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from utils import *
from run_utils import *
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold



arguments=sys.argv
argument_algorithm_names = arguments[1]
if argument_algorithm_names == 'de_config':
    algorithm_names = [f'DE_CR_{CR}_crossover_{crossover}' for CR in [0.2, 0.4, 0.6, 0.8] for crossover in
                       ['exp', 'bin']]
else:
    algorithm_names = get_argument_elements_from_list(arguments[1], False)
train_on_seed= True if arguments[2].lower()=='true' else False
result_dir='problem_classification_ela_results_same_features'
seeds=[2,150,500,750,1000]
dimension=3
instance_min, instance_max=0,999
os.makedirs(result_dir, exist_ok=True)


feature_df=pd.DataFrame()
for algorithm in algorithm_names:
    for seed in seeds:
        df = pd.read_csv(f'ela/{algorithm}_dim_{dimension}_seed_{seed}_ELA.csv', index_col=[0])
        df['seed']=seed
        df = df.set_index(['problem_id','instance_id','seed'])
        
        feature_df=pd.concat([feature_df,df])
        

feature_names=['basic.costs_runtime',
 'ela_level.lda_mda_50',
 'ela_level.mmce_lda_25',
 'ela_level.mmce_lda_50',
 'ela_level.mmce_mda_50',
 'ela_meta.costs_runtime',
 'ela_meta.lin_simple.adj_r2',
 'ela_meta.lin_simple.coef.max',
 'ela_meta.lin_simple.coef.max_by_min',
 'ela_meta.lin_simple.coef.min',
 'ela_meta.lin_simple.intercept',
 'ela_meta.lin_w_interact.adj_r2',
 'ela_meta.quad_simple.adj_r2',
 'ela_meta.quad_simple.cond',
 'ela_meta.quad_w_interact.adj_r2',
 'pca.costs_runtime',
 'pca.expl_var.cor_init',
 'pca.expl_var.cor_x',
 'pca.expl_var.cov_init',
 'pca.expl_var.cov_x',
 'pca.expl_var_PC1.cor_init',
 'pca.expl_var_PC1.cor_x',
 'pca.expl_var_PC1.cov_init',
 'basic.lower_min',
 'pca.expl_var_PC1.cov_x',
 'disp.ratio_mean_25',
 'disp.diff_mean_05',
 'disp.ratio_mean_10',
 'disp.diff_median_25',
 'disp.ratio_median_02',
 'disp.diff_median_10',
 'disp.diff_median_05',
 'disp.diff_median_02',
 'disp.diff_mean_25',
 'disp.diff_mean_10',
 'disp.ratio_median_05',
 'disp.ratio_mean_02',
 'disp.diff_mean_02',
 'disp.costs_runtime',
 'disp.ratio_median_10',
 'disp.ratio_median_25',
 'ela_level.costs_runtime',
 'basic.upper_min',
 'basic.objective_min',
 'basic.objective_max',
 'disp.ratio_mean_05']



print(feature_df.shape)
print(feature_df.describe())
feature_df=feature_df[feature_names]
print(feature_df.shape)
print(feature_df.describe())

feature_df['y']=list(feature_df.reset_index()['problem_id'].values)


my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['black','#008b8b','#9acd32','#e3f8b7'])


instance_count=instance_max-instance_min if instance_max!=999 else 1000


for train_seed in seeds:

    if 'config' not in argument_algorithm_names:
        global_name=f'{"_".join(algorithm_names)}_instance_count_{instance_count}_{"train" if train_on_seed else "test"}_on_seed_{train_seed}'
    else:
        global_name=f'{argument_algorithm_names}_instance_count_{instance_count}_{"train" if train_on_seed else "test"}_on_seed_{train_seed}'

    kf = KFold(n_splits=10)
    df=feature_df.copy()
    fold=0
    instance_ids=np.array(feature_df.reset_index()['instance_id'].drop_duplicates())
    all_predictions=[]
    all_ys=[]
    accuracies=[]
    importances=[]

    for train_index, test_index in kf.split(instance_ids):

        run_name=f'{global_name}_fold_{fold}'
        report_location=f'{result_dir}/{run_name}_report.csv'
        if os.path.isfile(report_location) and False:
            print(f'Report already exists: {report_location}. Skipping run')
            continue
        train,test=get_split_data_for_problem_classification_generalization_testing(instance_ids, train_index, test_index, feature_df, result_dir, run_name, train_seed, train_on_seed)

        clf, train, test= train_random_forest(train,test)
        preds,report_dict=save_classification_report(clf,test,report_location)

        #save_feature_importance(run_name, clf, dimension, iteration_min, iteration_max, result_dir,feature_names)

        test_predictions=pd.DataFrame(list(zip(test['y'].values,preds)), index=test.index, columns=['y','preds'])
        test_predictions.to_csv(f'{result_dir}/{run_name}_test_preds.csv', compression='zip')
        feature_importance_df=pd.DataFrame(list(clf.feature_importances_)).T
        feature_importance_df.columns=feature_names
        feature_importance_df.to_csv(f'{result_dir}/{run_name}_feature_importance.csv', compression='zip')

        fold+=1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from algorithm_baseline_utils import *
from run_utils import *
from run_utils import *
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from config import *
stat_color_mapping = {s: c for s, c in zip(['mean', 'min', 'max', 'std'], color_palette_4)}


def save_feature_importance(run_name, clf, dimension, iteration_min, iteration_max,result_dir, feature_names,features_to_plot=10):
    
    features = feature_names
    importances = clf.feature_importances_
    indices = np.argsort(importances)
    indices = indices[0:features_to_plot]
    feature_names_to_plot = [features[i] for i in indices]
    feature_colors = [stat_color_mapping[f.split(' ')[0]] for f in feature_names_to_plot]

    plt.figure(figsize=(10, 10))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color=feature_colors, align='center')
    plt.yticks(range(len(indices)), feature_names_to_plot)
    plt.xlabel('Relative Importance')
    plt.savefig(f'{result_dir}/{run_name}_top_10_features.pdf')
    plt.show()


def extract_features(sample_df, dimension, iteration_min, iteration_max, task='algorithm_classification'):
    grouped = sample_df.groupby(['algorithm_name', 'problem_id', 'instance_id', 'seed', 'iteration'])
    df = pd.concat([grouped.mean(), grouped.min(), grouped.max(), grouped.std()], axis=1)

    xs = []
    ys = []
    ids = []
    for group_key, group_values in df.groupby(['algorithm_name', 'problem_id', 'instance_id', 'seed']).groups.items():
        xs += [df.loc[group_values].values.flatten()]
        ys += ([group_key[0]] if task=='algorithm_classification' else [group_key[1]])
        ids += [group_key]

    feature_names = [f'{j}_it_{it}_dim_{i}' for it in range(iteration_min, iteration_max + 1) for j in
                     ['mean', 'min', 'max', 'std'] for i in range(0, dimension + 1)]
    df = pd.DataFrame(xs)
    df.columns = feature_names
    df.index = pd.MultiIndex.from_tuples(ids, names=['algorithm_name', 'problem_id', 'instance_id', 'seed'])
    
    df['y'] = ys
    return df


def plot_features(feature_df, color_palette):
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_df_scaled = pd.DataFrame(scaler.fit_transform(feature_df.drop(columns=['y'])))
    X_2d = TSNE(n_components=2).fit_transform(feature_df_scaled)
    X_2d = pd.DataFrame(X_2d)
    X_2d.columns = ['x1', 'x2']
    X_2d['class'] = feature_df['y'].values
    sns.scatterplot(data=X_2d, x="x1", y="x2", hue='class', palette=color_palette)
    plt.show()


def get_arguments(arguments):
    print('Number of arguments:', len(arguments), 'arguments.')
    print('Argument List:', str(arguments))

    argument_algorithm_names = arguments[1]
    if argument_algorithm_names == 'de_config':
        algorithm_names = [f'DE_CR_{CR}_crossover_{crossover}' for CR in [0.2, 0.4, 0.6, 0.8] for crossover in
                           ['exp', 'bin']]
    else:
        algorithm_names = get_argument_elements_from_list(arguments[1], False)
    seeds = get_argument_elements_from_list(arguments[2], True)

    iteration_min, iteration_max, instance_min, instance_max, difference = 0, 29, 1, 999, False
    if len(arguments) > 3:
        iteration_min, iteration_max = get_argument_elements_from_list(arguments[3], True)
        print(iteration_min, iteration_max)
    if len(arguments) > 4:
        instance_min, instance_max = get_argument_elements_from_list(arguments[4], True)
        print(instance_min, instance_max)
    if len(arguments) > 5:
        difference = True if arguments[5].lower()=='true' else False
        print(instance_min, instance_max)

    return argument_algorithm_names, algorithm_names, seeds, iteration_min, iteration_max, instance_min, instance_max, difference


def train_random_forest(train, test, do_regression=False):
    train = train.fillna(train.mean())
    test = test.fillna(train.mean())
    clf = RandomForestRegressor() if do_regression else RandomForestClassifier()
    clf.fit(train.drop(columns=['y']), train['y'])
    return clf,train,test


def save_classification_report(clf, test, report_location):
    preds = clf.predict(test.drop(columns=['y']))
    report_dict = classification_report(test['y'], preds,  output_dict=True)
    report_df = pd.DataFrame(report_dict)
    report_df.to_csv(report_location)
    return preds, report_dict


def get_split_data_for_algorithm_classification(problem_ids,train_index, test_index, feature_df, result_dir, run_name):
    
    
    train_ids, test_ids = problem_ids[train_index], problem_ids[test_index]
   
    train = feature_df.copy().query("problem_id in @train_ids")
    print('Train/test shape')
    print(train.shape)
    test = feature_df.copy().query("problem_id in @test_ids")
    #train.to_csv(f'{result_dir}/{run_name}_train.csv')
    #test.to_csv(f'{result_dir}/{run_name}_test.csv')
    print(test.shape)
    return train, test


def get_split_data_for_problem_classification(instance_ids,train_index, test_index, feature_df, result_dir, run_name):
    
    train_ids, test_ids = instance_ids[train_index], instance_ids[test_index]
    train = feature_df.copy().query("instance_id in @train_ids")
    print('Train/test shape')
    print(train.shape)
    test = feature_df.copy().query("instance_id in @test_ids")
    #train.to_csv(f'{result_dir}/{run_name}_train.csv')
    #test.to_csv(f'{result_dir}/{run_name}_test.csv')
    print(test.shape)
    return train, test

def read_trajectory_data(algorithm_names, seeds, dimension):
    sample_df = pd.DataFrame()
    for algorithm_name in algorithm_names:
        for seed in seeds:
            sample_df_1 = pd.read_csv(f'algorithm_run_data/{algorithm_name}_dim_{dimension}_seed_{seed}.csv',
                                      index_col=[0])
            print(sample_df_1.shape)
            sample_df_1['seed'] = seed
            sample_df = sample_df.append(sample_df_1)

    sample_df = sample_df.drop(columns=['problem_name'])
    return sample_df


def get_split_data_for_algorithm_classification_generalization_testing(problem_ids, train_index, test_index, feature_df, result_dir, run_name, seed, train_on_seed=True):
    train_ids, test_ids = problem_ids[train_index], problem_ids[test_index]
    train = feature_df.query("seed==@seed and problem_id in @train_ids") if train_on_seed else feature_df.query("seed!=@seed and problem_id in @train_ids")
    print('Train/test shape')
    print(train.shape)
    test = feature_df.copy().query("problem_id in @test_ids")
    #train.to_csv(f'{result_dir}/{run_name}_train.csv')
    #test.to_csv(f'{result_dir}/{run_name}_test.csv')
    print(test.shape)
    return train, test

def get_split_data_for_problem_classification_generalization_testing(instance_ids, train_index, test_index, feature_df, result_dir, run_name, seed, train_on_seed=True):
    train_ids, test_ids = instance_ids[train_index], instance_ids[test_index]
    train = feature_df.query("seed==@seed and instance_id in @train_ids") if train_on_seed else feature_df.query("seed!=@seed and instance_id in @train_ids")
    print('Train/test shape')
    print(train.shape)
    test = feature_df.copy().query("instance_id in @test_ids")
    #train.to_csv(f'{result_dir}/{run_name}_train.csv')
    #test.to_csv(f'{result_dir}/{run_name}_test.csv')
    print(test.shape)
    return train, test

def difference_features(feature_df,max_iteration):
    columns_values=[]
    columns_names=[]
    for c_index, c in enumerate(feature_df.columns):
        if len(c.split('_'))>2:
            iteration=int(c.split('_')[2])
            if iteration<=max_iteration-1:
                c_next=c.replace(f'_it_{iteration}_',f'_it_{iteration+1}_')
                columns_names+=[(f'{c_next} - {c}')]
                columns_values+=[(feature_df[c_next]- feature_df[c])]
    differenced_features=pd.concat(columns_values,axis=1)
    differenced_features.columns=columns_names
    differenced_features['y']=feature_df['y'].values
    #differenced_features['seed']=feature_df.reset_index()['seed'].values
    return differenced_features


import argparse
import os

import numpy as np
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature
from torch_rechub.models.multi_task import DCMT, DCN4MMOE, DeepFM4MMOE, DR, ESMM, IPS, MMOE, UCVRLC
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# hyper parameters
epoch = 10
learning_rate = 1e-3
weight_decay = 1e-5
batch_size=2048
cvr_params = {"dims": [512, 256, 128, 64]}
counterfactual_cvr_params = {"dims": [512, 256, 128, 64]}
ctr_params = {"dims": [512, 256, 128, 64]}
imputation_params = {"dims": [512, 256, 128, 64]}


def load_csv(party, data_path = '../../data/kuaipure'):
    print('Loading training data...')
    df_train = pd.read_csv(data_path + '/filtered_train.csv') 
    print('Loading testing data...')
    df_test = pd.read_csv(data_path + '/filtered_test.csv')  

    return df_train, df_test


def preprocess(party, df_train, df_test):
    df_train.rename(columns={'is_like': 'cvr_label', 'is_click': 'ctr_label'}, inplace=True)
    df_train["ctcvr_label"] = df_train['cvr_label'] * df_train['ctr_label']

    df_test.rename(columns={'is_like': 'cvr_label', 'is_click': 'ctr_label'}, inplace=True)
    df_test["ctcvr_label"] = df_test['cvr_label'] * df_test['ctr_label']

    df_all = pd.concat([df_train, df_test], axis=0)

    col_names = df_all.columns.values.tolist()
    sparse_cols = [col for col in col_names if col not in ['cvr_label', 'ctr_label', 'ctcvr_label']]

    label_cols = ['cvr_label', 'ctr_label', "ctcvr_label"]  # the order of 3 labels must fixed as this

    used_cols = sparse_cols # ESMM only for sparse features in origin paper

    item_cols = ['video_id', 'author_id',
                   'video_type', 'upload_type', 'video_duration', 'server_width',
                   'server_height', 'music_type', 'column_0', 'column_1', 'column_10',
                   'column_11', 'column_12', 'column_13', 'column_14', 'column_15',
                   'column_16', 'column_17', 'column_18', 'column_19', 'column_2',
                   'column_20', 'column_21', 'column_22', 'column_23', 'column_24',
                   'column_25', 'column_26', 'column_27', 'column_28', 'column_29',
                   'column_3', 'column_30', 'column_34', 'column_35', 'column_36',
                   'column_37', 'column_38', 'column_39', 'column_4', 'column_40',
                   'column_42', 'column_43', 'column_5', 'column_54', 'column_56',
                   'column_6', 'column_60', 'column_62', 'column_65', 'column_67',
                   'column_68', 'column_7', 'column_8', 'column_9']

    user_cols = [col for col in used_cols if col not in item_cols]

    user_features = [SparseFeature(col, df_all[col].max() + 1, embed_dim=16) for col in user_cols]
    item_features = [SparseFeature(col, df_all[col].max() + 1, embed_dim=16) for col in item_cols]

    return used_cols, label_cols, user_features, item_features


def get_dataloader(df_train, df_test, used_cols, label_cols):
    x_train, y_train = {name: df_train[name].values for name in used_cols}, df_train[label_cols].values
    x_test, y_test = {name: df_test[name].values for name in used_cols}, df_test[label_cols].values
    dg = DataGenerator(x_train, y_train)
    train_dataloader, _, test_dataloader = dg.generate_dataloader(x_val=x_train, 
                                                                  y_val=y_train, 
                                                                  x_test=x_test, 
                                                                  y_test=y_test, 
                                                                  batch_size=batch_size,
                                                                  num_workers=0
                                                                  )
    
    return train_dataloader, test_dataloader


def train(party, seed, weight, strategy, ablation_weight, device, user_features, item_features, train_dataloader, test_dataloader, cvr_params, counterfactual_cvr_params, ctr_params):

    if 'dcmt' in strategy:
        model = DCMT(user_features, item_features, cvr_params, counterfactual_cvr_params, ctr_params)
        task_types = ["classification", "classification", "classification", "classification"] 
    elif 'ucvrlc' in strategy:
        model = UCVRLC(user_features, item_features, cvr_params, ctr_params)
        task_types = ["classification", "classification", "classification"] 
    elif 'ips' in strategy:
        model = IPS(user_features, item_features, cvr_params, ctr_params)
        task_types = ["classification", "classification", "classification"] 
    elif 'dr' in strategy:
        model = DR(user_features, item_features, cvr_params, imputation_params, ctr_params)
        task_types = ["classification", "classification", "classification", "regression"] 
    elif 'mmoe' in strategy:
        task_types = ["classification", "classification"] 
        if 'dcn' in strategy:
            model = DCN4MMOE(user_features+item_features, task_types, 4, expert_params={"dims": [128]}, tower_params_list=[{"dims": [64, 32, 16]}, {"dims": [64, 32, 16]}])
        elif 'deepfm' in strategy:
            model = DeepFM4MMOE(user_features+item_features, task_types, 4, tower_params_list=[{"dims": [16]}, {"dims": [16]}])
        else:
            model = MMOE(user_features+item_features, task_types, 4, expert_params={"dims": [128]}, tower_params_list=[{"dims": [64, 32, 16]}, {"dims": [64, 32, 16]}])
    else:
        model = ESMM(user_features, item_features, cvr_params, ctr_params)
        task_types = ["classification", "classification", "classification"] 

    save_dir = 'logs/kuaipure/rebuttal/mlp/{}'.format(strategy)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if strategy == 'ctr':
        _earlystop_taskid = 1
    else:
        _earlystop_taskid = 0

    mtl_trainer = MTLTrainer(model, 
                             task_types=task_types, 
                             optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, 
                             n_epoch=epoch, 
                             earlystop_taskid=_earlystop_taskid,
                             earlystop_patience=3, 
                             device=device, 
                             model_path=save_dir)
        
    total_log = mtl_trainer.fit(train_dataloader, test_dataloader, party, seed, weight, strategy, ablation_weight)

    if 'dcmt' in strategy:
        column_names = ['cvr_loss', 'counterfactual_cvr_loss', 'ctr_loss', 'ctcvr_loss', 'cvr_auc', 'counterfactual_cvr_auc', 'ctr_auc', 'ctcvr_auc', 'cvr_ks', 'counterfactual_cvr_ks', 'ctr_ks', 'ctcvr_ks', 'cvr_log_loss', 'counterfactual_cvr_log_loss', 'ctr_log_loss', 'ctcvr_log_loss']
    elif 'dr' in strategy:
        column_names = ['cvr_loss', 'ctr_loss', 'ctcvr_loss', 'low_variance_loss', 'cvr_auc', 'ctr_auc', 'ctcvr_auc', 'cvr_ks', 'ctr_ks', 'ctcvr_ks', 'cvr_log_loss', 'ctr_log_loss', 'ctcvr_log_loss']
    elif 'mmoe' in strategy:
        column_names = ['cvr_loss', 'ctr_loss', 'cvr_auc', 'ctr_auc', 'cvr_ks', 'ctr_ks', 'cvr_log_loss', 'ctr_log_loss']
    else:
        column_names = ['cvr_loss', 'ctr_loss', 'ctcvr_loss', 'cvr_auc', 'ctr_auc', 'ctcvr_auc', 'cvr_ks', 'ctr_ks', 'ctcvr_ks', 'cvr_log_loss', 'ctr_log_loss', 'ctcvr_log_loss']

    total_log = pd.DataFrame(total_log, columns=column_names)
    total_log.to_csv(save_dir + '/{}_{}_{}.csv'.format(party, seed, strategy))



def main(party, seed, weight, strategy, ablation_weight, device):
    print('Loading csv files......')
    df_train, df_test = load_csv(party)

    print('Begin preprocess data......')
    used_cols, label_cols, user_features, item_features = preprocess(party, df_train, df_test)

    train_dataloader, test_dataloader = get_dataloader(df_train, df_test, used_cols, label_cols)

    torch.manual_seed(seed)
    print('Training starts......')
    train(party, seed, weight, strategy, ablation_weight, device, user_features, item_features, train_dataloader, test_dataloader, cvr_params, counterfactual_cvr_params, ctr_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--party', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight', type=int, default=50)
    parser.add_argument('--strategy', type=str, default='esmm')
    parser.add_argument('--ablation_weight', type=float, default=1)
    parser.add_argument('--device', type=str, required=True)

    args = parser.parse_args()

    print('Local model training with party = {}, seed = {}, weight = {}, strategy = {} and ablation_weight = {}'.format(args.party, args.seed, args.weight, args.strategy, args.ablation_weight))

    main(args.party, args.seed, args.weight, args.strategy, args.ablation_weight, args.device)

    
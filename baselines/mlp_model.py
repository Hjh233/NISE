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
cvr_params = {"dims": [160, 80]}
counterfactual_cvr_params = {"dims": [160, 80]}
ctr_params = {"dims": [160, 80]}
imputation_params = {"dims": [160, 80]}


def load_csv(party, data_path = '../../data/ali-ccp'):
    if party == 'both':
        print('Loading training data...')
        df_train = pd.read_csv(data_path + '/ali_ccp_train.csv') 
        print('Loading testing data...')
        df_test = pd.read_csv(data_path + '/ali_ccp_test.csv') 
    else:
        print('Loading training data...')
        df_train = pd.read_csv(data_path + '/ali_ccp_train_{}.csv'.format(party)) 
        print('Loading testing data...')
        df_test = pd.read_csv(data_path + '/ali_ccp_test_{}.csv'.format(party))    

    return df_train, df_test


def preprocess(party, df_train, df_test, sample=False, frac=1.0, random_state=0):
    df_train.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)
    df_train["ctcvr_label"] = df_train['cvr_label'] * df_train['ctr_label']

    df_test.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)
    df_test["ctcvr_label"] = df_test['cvr_label'] * df_test['ctr_label']

    if sample:
        df_clicked_samples = df_train[df_train['ctr_label'] == 1]
        df_unclicked_samples = df_train[df_train['ctr_label'] == 0].sample(frac=frac, random_state=random_state)

        df_train = pd.concat([df_clicked_samples, df_unclicked_samples], axis=0)

    print('df_train.shape', len(df_train))
    print('df_clicked_samples', len(df_train[df_train['ctr_label'] == 1]))
    print('df_unclicked_samples', len(df_train[df_train['ctr_label'] == 0]))
    print('df_converted_samples', len(df_train[df_train['cvr_label'] == 1]))

    df_all = pd.concat([df_train, df_test], axis=0)

    col_names = df_all.columns.values.tolist()
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['cvr_label', 'ctr_label', 'ctcvr_label']]
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))

    label_cols = ['cvr_label', 'ctr_label', "ctcvr_label"]  # the order of 3 labels must fixed as this

    used_cols = sparse_cols # ESMM only for sparse features in origin paper

    if party == 'both':
        item_cols = ['129', '205', '206', '207', '210', '216']  
    elif party == 'platform':
        item_cols = ['129', '205', '206', '207']
    elif party == 'advertiser':
        item_cols = ['205', '210', '216']  
    user_cols = [col for col in used_cols if col not in item_cols]

    user_features = [SparseFeature(col, df_all[col].max() + 1, embed_dim=16) for col in user_cols]
    item_features = [SparseFeature(col, df_all[col].max() + 1, embed_dim=16) for col in item_cols]

    return used_cols, label_cols, user_features, item_features, df_train


def get_dataloader(df_train, df_test, used_cols, label_cols):
    x_train, y_train = {name: df_train[name].values for name in used_cols}, df_train[label_cols].values
    x_test, y_test = {name: df_test[name].values for name in used_cols}, df_test[label_cols].values
    dg = DataGenerator(x_train, y_train)
    train_dataloader, _, test_dataloader = dg.generate_dataloader(x_val=x_train, 
                                                                  y_val=y_train, 
                                                                  x_test=x_test, 
                                                                  y_test=y_test, 
                                                                  batch_size=batch_size
                                                                  )
    
    return train_dataloader, test_dataloader


def train(party, seed, weight, strategy, ablation_weight, device, user_features, item_features, train_dataloader, test_dataloader, cvr_params, counterfactual_cvr_params, ctr_params, frac):

    if 'dcmt' in strategy:
        model = DCMT(user_features, item_features, cvr_params, counterfactual_cvr_params, ctr_params)
        task_types = ["classification", "classification", "classification", "classification"] 
    elif 'ucvrlc' in strategy or 'dwa' in strategy or 'dtp' in strategy or 'naive' in strategy:
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
            model = DCN4MMOE(user_features+item_features, task_types, 2, expert_params={"dims": [80]}, tower_params_list=[{"dims": [40]}, {"dims": [40]}])
        elif 'deepfm' in strategy:
            model = DeepFM4MMOE(user_features+item_features, task_types, 2, tower_params_list=[{"dims": [16]}, {"dims": [16]}])
        else:
            model = MMOE(user_features+item_features, task_types, 2, expert_params={"dims": [80]}, tower_params_list=[{"dims": [40]}, {"dims": [40]}])
    else:
        model = ESMM(user_features, item_features, cvr_params, ctr_params)
        task_types = ["classification", "classification", "classification"] 

    save_dir = 'logs/ali_ccp/local/rebuttal/mlp/{}/{}'.format(strategy, int(weight * 10))
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


def inference(party, strategy, device, user_features, item_features, test_dataloader, cvr_params, counterfactual_cvr_params, ctr_params):
    save_dir = 'logs/ali_ccp/local/rebuttal/mlp/bias/{}'.format(strategy)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    f = open(save_dir + '/bias.txt', 'a')

    for seed in range(5):
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
            model = MMOE(user_features+item_features, task_types, 2, expert_params={"dims": [80]}, tower_params_list=[{"dims": [40]}, {"dims": [40]}])
        else:
            model = ESMM(user_features, item_features, cvr_params, ctr_params)
            task_types = ["classification", "classification", "classification"] 

        model.load_state_dict(torch.load('logs/ali_ccp/local/mlp/{}/{}/model_both_{}.pth'.format(strategy, party, seed)))

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
            
        _bias = (mtl_trainer.get_bias_in_click_space(test_dataloader) / 9195) - 1

        print('_bias', _bias)

        f.write('{}'.format(_bias))
        f.write(',')


def main(party, seed, weight, strategy, ablation_weight, device, frac):
    print('Loading csv files......')
    df_train, df_test = load_csv(party)

    print('Begin preprocess data......')
    used_cols, label_cols, user_features, item_features, df_sampled_train = preprocess(party, df_train, df_test, sample=False, frac=frac, random_state=seed)

    train_dataloader, test_dataloader = get_dataloader(df_sampled_train, df_test, used_cols, label_cols)

    torch.manual_seed(seed)
    print('Training starts......')
    train(party, seed, weight, strategy, ablation_weight, device, user_features, item_features, train_dataloader, test_dataloader, cvr_params, counterfactual_cvr_params, ctr_params, frac)
    # inference(party, strategy, device, user_features, item_features, test_dataloader, cvr_params, counterfactual_cvr_params, ctr_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--party', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight', type=float, default=50)
    parser.add_argument('--strategy', type=str, default='esmm')
    parser.add_argument('--ablation_weight', type=float, default=1)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--frac', type=float, default=1)

    args = parser.parse_args()

    print('Local model training with party = {}, seed = {}, weight = {}, strategy = {}, ablation_weight = {} and frac = {}'.format(args.party, args.seed, args.weight, args.strategy, args.ablation_weight, args.frac))

    main(args.party, args.seed, args.weight, args.strategy, args.ablation_weight, args.device, args.frac)

    
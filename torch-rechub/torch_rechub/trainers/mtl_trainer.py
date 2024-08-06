import os
import tqdm
import numpy as np
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.callback import EarlyStopper
from ..basic.metric import log_loss
from ..utils.data import get_loss_func, get_metric_func
from ..models.multi_task import DCMT, DR, ESMM, IPS, MMOE, PLE, UCVRLC
from ..utils.mtl import shared_task_layers, gradnorm, MetaBalance


torch.autograd.set_detect_anomaly(True)


def calculate_ks(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    ks_value = max(tpr - fpr)
    return ks_value


class MTLTrainer(object):
    """A trainer for multi task learning.

    Args:
        model (nn.Module): any multi task learning model.
        task_types (list): types of tasks, only support ["classfication", "regression"].
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        adaptive_params (dict): parameters of adaptive loss weight method. Now only support `{"method" : "uwl"}`. 
        n_epoch (int): epoch number of training.
        earlystop_taskid (int): task id of earlystop metrics relies between multi task (default = 0).
        earlystop_patience (int): how long to wait after last time validation auc improved (default = 10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
        self,
        model,
        task_types,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        adaptive_params=None,
        n_epoch=10,
        earlystop_taskid=0,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
    ):
        self.model = model
        if gpus is None:
            gpus = []
        if optimizer_params is None:
            optimizer_params = {
                "lr": 1e-3,
                "weight_decay": 1e-5
            }
        self.task_types = task_types
        self.n_task = len(task_types)
        self.loss_weight = None
        self.adaptive_method = None

        if adaptive_params is not None:
            if adaptive_params["method"] == "uwl":
                self.adaptive_method = "uwl"
                self.loss_weight = nn.ParameterList(nn.Parameter(torch.zeros(1)) for _ in range(self.n_task))
                self.model.add_module("loss weight", self.loss_weight)
            elif adaptive_params["method"] == "metabalance":
                self.adaptive_method = "metabalance"
                share_layers, task_layers = shared_task_layers(self.model)
                self.meta_optimizer = MetaBalance(share_layers)
                self.share_optimizer = optimizer_fn(share_layers, **optimizer_params)
                self.task_optimizer = optimizer_fn(task_layers, **optimizer_params)
            elif adaptive_params["method"] == "gradnorm":
                self.adaptive_method = "gradnorm"
                self.alpha = adaptive_params.get("alpha", 0.16)
                share_layers = shared_task_layers(self.model)[0]
                # gradnorm calculate the gradients of each loss on the last fully connected shared layer weight(dimension is 2)
                for i in range(len(share_layers)):
                    if share_layers[-i].ndim == 2:
                        self.last_share_layer = share_layers[-i]
                        break
                self.initial_task_loss = None
                self.loss_weight = nn.ParameterList(nn.Parameter(torch.ones(1)) for _ in range(self.n_task))
                self.model.add_module("loss weight", self.loss_weight)

        if self.adaptive_method != "metabalance":
            self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default Adam optimizer

        self.scheduler = None

        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.loss_fns = [get_loss_func(task_type) for task_type in task_types]
        self.evaluate_fns = [get_metric_func(task_type) for task_type in task_types]
        self.n_epoch = n_epoch
        self.earlystop_taskid = earlystop_taskid
        self.early_stopper = EarlyStopper(patience=earlystop_patience, epoch=n_epoch)

        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_path = model_path


    def train_one_epoch(self, data_loader, weight, strategy, epoch=-1, ablation_weight=1, loss_history=[], kappa_history=[]):
        '''
            loss_history: [loss_cvr_prev_2, loss_ctr_prev_2, loss_cvr_prev_1, loss_ctr_prev_1] for dynamic weight average
        '''
        temperature = 2.0
        alpha = 0.5 # not specified in the original paper
        targets, predicts = list(), list()

        self.model.train()
        total_loss = np.zeros(self.n_task)
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)

        kappa_cvr, kappa_ctr = kappa_history[0], kappa_history[1]

        for iter_i, (x_dict, ys) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  # tensor to GPU
            ys = ys.to(self.device) # cvr, ctr, ctcvr
            y_preds = self.model(x_dict)

            if isinstance(self.model, ESMM):
                loss_list = [self.loss_fns[i](y_preds[:, i], ys[:, i].float()) for i in range(self.n_task)]
                loss = sum(loss_list[1:]) 

            elif isinstance(self.model, MMOE) or isinstance(self.model, PLE):
                # loss_list = [self.loss_fns[i](y_preds[:, i], ys[:, i].float()) for i in range(self.n_task)]
                # loss = sum(loss_list)  
                '''
                    y_preds = [cvr, ctr]
                    ys = [cvr, ctr]
                '''
                loss_ctr = self.loss_fns[1](y_preds[:, 1], ys[:, 1].float()) 

                if 'ucvrlc' in strategy:
                    clicked_cvr_weight = ys[:, 1] 
                    unclicked_cvr_weight = 1 - ys[:, 1]

                    clicked_cvr_loss_fn = torch.nn.BCELoss(weight=clicked_cvr_weight.to(self.device).detach())
                    loss_clicked_cvr = clicked_cvr_loss_fn(y_preds[:, 0], ys[:, 0].float())

                    unclicked_cvr_loss_fn = torch.nn.BCELoss(weight=unclicked_cvr_weight.to(self.device).detach())
                    loss_unclicked_cvr = unclicked_cvr_loss_fn(y_preds[:, 0], y_preds[:, 0].float())    

                    loss_cvr = loss_clicked_cvr + loss_unclicked_cvr

                    cvr_weight_per_batch = loss_ctr.item() / loss_cvr.item()
                    cvr_weight_per_batch = ablation_weight * min(cvr_weight_per_batch, weight)
                    loss = loss_ctr + cvr_weight_per_batch * loss_cvr

                else:
                    _loss_cvr_weight = ys[:, 1] # o_{u,i} 

                    cvr_loss_fn = torch.nn.BCELoss(weight=_loss_cvr_weight.to(self.device).detach())
                    loss_cvr = cvr_loss_fn(y_preds[:, 0], ys[:, 0].float())

                    loss = loss_ctr + loss_cvr

                loss_list = [loss_cvr, loss_ctr]

            elif isinstance(self.model, DCMT):
                '''
                    y_preds = [cvr, counterfactual_cvr, ctr, ctcvr]
                    ys = [cvr, ctr, ctcvr]
                '''
                actual_weight = ys[:, 1] / y_preds[:, 2]
                counterfactual_weight = (1 - ys[:, 1]) / (1 - y_preds[:, 2])

                '''
                    self.loss_fns[i]: i = n_tasks.index(task)
                    y_preds[:, i]: i = y_preds.index(task)
                    ys[:, j]: j = ys.index(task)
                    e.g. ctr_i = [cvr, counterfactual_cvr, ctr, ctcvr].index(ctr) = 2, ctr_j = [cvr, ctr, ctcvr].index(ctr) = 1          
                '''
                loss_ctr = self.loss_fns[2](y_preds[:, 2], ys[:, 1].float()) 

                cvr_loss_fn = torch.nn.BCELoss(weight=actual_weight.to(self.device).detach())
                loss_cvr = cvr_loss_fn(y_preds[:, 0], ys[:, 0].float())

                counterfactual_cvr_loss_fn = torch.nn.BCELoss(weight=counterfactual_weight.to(self.device).detach())
                loss_counterfactual_cvr = counterfactual_cvr_loss_fn(y_preds[:, 1], 1 - ys[:, 0].float())
                
                soft_constraint_value = y_preds[:, 0] + y_preds[:, 1]
                hard_constraint_value = torch.ones_like(y_preds[:, 0])
                l1_loss_fn = torch.nn.L1Loss()
                loss_constraint = l1_loss_fn(soft_constraint_value, hard_constraint_value)

                loss_dcmt = loss_cvr + loss_counterfactual_cvr + 1e-3 * loss_constraint
                loss_ctcvr = self.loss_fns[3](y_preds[:, 3], ys[:, 2].float())

                loss = loss_ctr + loss_dcmt + loss_ctcvr

                loss_list = [loss_cvr, loss_counterfactual_cvr, loss_ctr, loss_ctcvr]

            elif isinstance(self.model, IPS):
                '''
                    y_preds = [cvr, ctr, ctcvr]
                    ys = [cvr, ctr, ctcvr]
                '''
                _loss_cvr_weight = ys[:, 1] / y_preds[:, 1] # o_{u,i} / \hat{o}_{u,i}

                loss_ctr = self.loss_fns[1](y_preds[:, 1], ys[:, 1].float()) 
                loss_ctcvr = self.loss_fns[2](y_preds[:, 2], ys[:, 2].float()) 

                cvr_loss_fn = torch.nn.BCELoss(weight=_loss_cvr_weight.to(self.device).detach())
                loss_cvr = cvr_loss_fn(y_preds[:, 0], ys[:, 0].float())

                loss = loss_ctr + loss_cvr + loss_ctcvr

                loss_list = [loss_cvr, loss_ctr, loss_ctcvr]

            elif isinstance(self.model, DR):
                '''
                    y_preds = [cvr, ctr, ctcvr, imputation_error]
                    ys = [cvr, ctr, ctcvr]
                    reference: https://github.com/DongHande/AutoDebias/blob/main/baselines/DR.py
                '''
                # 1. update imputation error model
                cvr_loss_in_exposure = F.binary_cross_entropy_with_logits(y_preds[:, 0], ys[:, 0].float(), reduction='none').detach()
                imputation_loss_fn_in_click = torch.nn.MSELoss(reduction='none')
                loss_imputation_in_click = imputation_loss_fn_in_click(y_preds[:, 3], cvr_loss_in_exposure) * ys[:, 1]
                imputation_loss = torch.sum(loss_imputation_in_click) / ys.shape[0]

                params_to_update = list(self.model.tower_imputation.parameters())

                _optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
                _optimizer = torch.optim.Adam(params_to_update, **_optimizer_params)  

                self.model.tower_imputation.zero_grad()
                imputation_loss.backward(retain_graph=True)
                _optimizer.step()

                # 2. update the whole model
                self.model.train()
                y_preds_ = self.model(x_dict)
                _loss_cvr_weight = ys[:, 1] / y_preds_[:, 1] # o_{u,i} / \hat{o}_{u,i}
                _low_variance_weight = (y_preds_[:, 1] - ys[:, 1]) / y_preds_[:, 1] # (\hat{o}_{u,i} - o_{u,i}) / \hat{o}_{u,i}
                _low_variance_weight.to(self.device).detach()

                loss_ctr = self.loss_fns[1](y_preds_[:, 1], ys[:, 1].float()) 
                loss_ctcvr = self.loss_fns[2](y_preds_[:, 2], ys[:, 2].float()) 

                cvr_loss_fn = torch.nn.BCELoss(weight=_loss_cvr_weight.to(self.device).detach())
                loss_cvr = cvr_loss_fn(y_preds_[:, 0], ys[:, 0].float())

                loss_low_variance_array = _low_variance_weight * y_preds_[:, 3]
                loss_low_variance = torch.mean(loss_low_variance_array)

                loss = loss_ctr + loss_cvr + loss_low_variance + loss_ctcvr

                loss_list = [loss_cvr, loss_ctr, loss_ctcvr, loss_low_variance]

            elif isinstance(self.model, UCVRLC):
                clicked_cvr_weight = ys[:, 1] 
                unclicked_cvr_weight = 1 - ys[:, 1]

                loss_ctr = self.loss_fns[1](y_preds[:, 1], ys[:, 1].float()) 
                loss_ctcvr = self.loss_fns[2](y_preds[:, 2], ys[:, 2].float()) 

                clicked_cvr_loss_fn = torch.nn.BCELoss(weight=clicked_cvr_weight.to(self.device).detach())
                loss_clicked_cvr = clicked_cvr_loss_fn(y_preds[:, 0], ys[:, 0].float())

                unclicked_cvr_loss_fn = torch.nn.BCELoss(weight=unclicked_cvr_weight.to(self.device).detach())
                loss_unclicked_cvr = unclicked_cvr_loss_fn(y_preds[:, 0], y_preds[:, 0].float())    

                loss_cvr = loss_clicked_cvr + loss_unclicked_cvr

                if 'ablation_1' in strategy:
                    loss = loss_cvr

                if 'ablation_2' in strategy: # weighting
                    loss = loss_ctr + loss_cvr

                if 'ablation_3' in strategy: # unclicked samples  
                    loss_cvr = loss_clicked_cvr
                        
                    cvr_weight_per_batch = loss_ctr.item() / loss_cvr.item()
                    cvr_weight_per_batch = ablation_weight * min(cvr_weight_per_batch, weight)
                    loss = loss_ctr + cvr_weight_per_batch * loss_cvr

                if strategy == 'adaptive_ucvrlc':
                    cvr_weight_per_batch = loss_ctr.item() / loss_cvr.item()
                    cvr_weight_per_batch = ablation_weight * min(cvr_weight_per_batch, weight)
                    loss = loss_ctr + cvr_weight_per_batch * loss_cvr
                elif strategy == 'naive':
                    loss = loss_ctr + weight * loss_cvr
                elif strategy == 'dwa':
                    '''
                        Paper reference: End-to-End Multi-Task Learning with Attention
                        https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf
                        Code reference: https://github.com/lorenmt/auto-lambda/blob/main/trainer_dense.py
                    '''
                    if epoch == 0 or epoch == 1:
                        loss = loss_ctr + loss_cvr
                    else:
                        loss_cvr_prev_2, loss_ctr_prev_2, loss_cvr_prev_1, loss_ctr_prev_1 = loss_history[0], loss_history[1], loss_history[2], loss_history[3]
                        w_ctr_prev_1, w_cvr_prev_1 = loss_ctr_prev_1 / loss_ctr_prev_2, loss_cvr_prev_1 / loss_cvr_prev_2
                        w = [w_ctr_prev_1, w_cvr_prev_1]
                        w = torch.softmax(torch.tensor(w) / temperature, dim = 0)
                        loss = w.numpy()[0] * loss_ctr + w.numpy()[1] * loss_cvr
                elif strategy == 'dtp':
                    '''
                        Paper reference: Dynamic Task Prioritization for Multitask Learning
                        https://openaccess.thecvf.com/content_ECCV_2018/papers/Michelle_Guo_Focus_on_the_ECCV_2018_paper.pdf
                    '''
                    targets.extend(ys.tolist())
                    predicts.extend(y_preds.tolist())
                    if (iter_i + 1) % 100 == 0:
                        '''
                            We use AUC as the metric for kappa.
                            Since in one batch, there may be ZERO positive samples, which raise the ValueError: Only one class present in y_true. ROC AUC score is not defined in that case,
                            we calculate the AUC score per 100 batches instead and update the weights per 100 iterations accordingly.
                            Also, in the original paper, the value of alpha is not specified, we thus choose alpha = 0.5.
                            Moreover, gamma is set to be 1, so - (1 - kappa_cvr) ** gamma * np.log2(kappa_cvr) is reduced to - (1 - kappa_cvr) * np.log2(kappa_cvr)
                        '''
                        targets, predicts = np.array(targets), np.array(predicts)
                        kappa_cvr = alpha * kappa_cvr + (1 - alpha) * self.evaluate_fns[0](targets[:, 0], predicts[:, 0])
                        kappa_ctr = alpha * kappa_ctr + (1 - alpha) * self.evaluate_fns[1](targets[:, 1], predicts[:, 1])
                        targets, predicts = list(), list()
                    FL_cvr = - (1 - kappa_cvr) * np.log2(kappa_cvr)
                    FL_ctr = - (1 - kappa_ctr) * np.log2(kappa_ctr)
                    loss = FL_cvr * loss_cvr + FL_ctr * loss_ctr

                loss_list = [loss_cvr, loss_ctr, loss_ctcvr]

            else:
                if self.adaptive_method != None:
                    if self.adaptive_method == "uwl":
                        loss = 0
                        for loss_i, w_i in zip(loss_list, self.loss_weight):
                            w_i = torch.clamp(w_i, min=0)
                            loss += 2 * loss_i * torch.exp(-w_i) + w_i
                else:
                    loss = sum(loss_list) / self.n_task

            if self.adaptive_method == 'metabalance':
                self.share_optimizer.zero_grad()
                self.task_optimizer.zero_grad()
                self.meta_optimizer.step(loss_list)
                self.share_optimizer.step()
                self.task_optimizer.step()
            elif self.adaptive_method == "gradnorm":
                self.optimizer.zero_grad()
                if self.initial_task_loss is None:
                    self.initial_task_loss = [l.item() for l in loss_list]
                gradnorm(loss_list, self.loss_weight, self.last_share_layer, self.initial_task_loss, self.alpha)
                self.optimizer.step()
                # renormalize
                loss_weight_sum = sum([w.item() for w in self.loss_weight])
                normalize_coeff = len(self.loss_weight) / loss_weight_sum
                for w in self.loss_weight:
                    w.data = w.data * normalize_coeff
            else:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += np.array([l.item() for l in loss_list])
        log_dict = {"task_%d:" % (i): total_loss[i] / (iter_i + 1) for i in range(self.n_task)}
        loss_logs = [total_loss[i] / (iter_i + 1) for i in range(self.n_task)]
        print("train loss: ", log_dict)
        if self.loss_weight:
            print("loss weight: ", [w.item() for w in self.loss_weight])

        if 'dtp' in strategy:
            return loss_logs, [kappa_cvr, kappa_ctr]
        else:
            return loss_logs


    def fit(self, train_dataloader, val_dataloader, party = 'base', seed = 0, weight = 1, strategy = 'esmm', ablation_weight = 1):
        total_log = []
        loss_history = []
        kappa_history = [0.1, 0.1]

        for epoch_i in range(self.n_epoch):
            if 'dtp' not in strategy:
                _log_per_epoch = self.train_one_epoch(train_dataloader, weight, strategy, epoch=epoch_i, ablation_weight=ablation_weight, loss_history=loss_history[-4:], kappa_history=kappa_history)
            else:
                _log_per_epoch, _kappa_history = self.train_one_epoch(train_dataloader, weight, strategy, epoch=epoch_i, ablation_weight=ablation_weight, loss_history=loss_history[-4:], kappa_history=kappa_history)
                kappa_history = _kappa_history
            loss_history.extend(_log_per_epoch[0:2])

            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  # update lr in epoch level by scheduler
            scores = self.evaluate(val_dataloader, strategy)
            print('epoch:', epoch_i, 'validation scores: ', scores)

            for score in scores:
                _log_per_epoch.append(score)

            total_log.append(_log_per_epoch)

            if self.early_stopper.stop_training(scores[self.earlystop_taskid], self.model.state_dict(), epoch_i+1):
                print('validation best auc of main task %d: %.6f' %
                      (self.earlystop_taskid, self.early_stopper.best_auc))
                self.model.load_state_dict(self.early_stopper.best_weights)
                break
        
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model_{}_{}.pth".format(party, seed)))  #save best auc model
 
        return total_log


    def evaluate(self, data_loader, strategy):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, ys) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
                ys = ys.to(self.device)
                y_preds = self.model(x_dict)
                targets.extend(ys.tolist())
                predicts.extend(y_preds.tolist())
        targets, predicts = np.array(targets), np.array(predicts)
        if 'dcmt' in strategy:
            cvr_score = self.evaluate_fns[0](targets[:, 0], predicts[:, 0])
            counterfactual_cvr_score = self.evaluate_fns[1](1 - targets[:, 0], predicts[:, 1])
            ctr_score = self.evaluate_fns[2](targets[:, 1], predicts[:, 2])
            ctcvr_score = self.evaluate_fns[3](targets[:, 2], predicts[:, 3])
            scores = [cvr_score, counterfactual_cvr_score, ctr_score, ctcvr_score]

            cvr_ks_score = calculate_ks(targets[:, 0], predicts[:, 0])
            counterfactual_cvr_ks_score = calculate_ks(1 - targets[:, 0], predicts[:, 1])
            ctr_ks_score = calculate_ks(targets[:, 1], predicts[:, 2])
            ctcvr_ks_score = calculate_ks(targets[:, 2], predicts[:, 3])
            ks_scores = [cvr_ks_score, counterfactual_cvr_ks_score, ctr_ks_score, ctcvr_ks_score]

            cvr_log_loss = log_loss(targets[:, 0], predicts[:, 0])
            counterfactual_cvr_log_loss = log_loss(1 - targets[:, 0], predicts[:, 1])
            ctr_log_loss = log_loss(targets[:, 1], predicts[:, 2])
            ctcvr_log_loss = log_loss(targets[:, 2], predicts[:, 3])
            log_loss_scores = [cvr_log_loss, counterfactual_cvr_log_loss, ctr_log_loss, ctcvr_log_loss]

            scores.extend(ks_scores)
            scores.extend(log_loss_scores)

        elif 'dr' in strategy:
            scores = [self.evaluate_fns[i](targets[:, i], predicts[:, i]) for i in range(self.n_task - 1)]
            ks_scores = [calculate_ks(targets[:, 0], predicts[:, 0]), calculate_ks(targets[:, 1], predicts[:, 1]), calculate_ks(targets[:, 2], predicts[:, 2])]
            log_loss_scores = [log_loss(targets[:, 0], predicts[:, 0]), log_loss(targets[:, 1], predicts[:, 1]), log_loss(targets[:, 2], predicts[:, 2])]
            scores.extend(ks_scores)
            scores.extend(log_loss_scores)
            
        elif 'mmoe' in strategy:
            scores = [self.evaluate_fns[i](targets[:, i], predicts[:, i]) for i in range(self.n_task)]               
            ks_scores = [calculate_ks(targets[:, 0], predicts[:, 0]), calculate_ks(targets[:, 1], predicts[:, 1])]
            log_loss_scores = [log_loss(targets[:, 0], predicts[:, 0]), log_loss(targets[:, 1], predicts[:, 1])]
            scores.extend(ks_scores)
            scores.extend(log_loss_scores)
            
        else:
            scores = [self.evaluate_fns[i](targets[:, i], predicts[:, i]) for i in range(self.n_task)]               
            ks_scores = [calculate_ks(targets[:, 0], predicts[:, 0]), calculate_ks(targets[:, 1], predicts[:, 1]), calculate_ks(targets[:, 2], predicts[:, 2])]
            log_loss_scores = [log_loss(targets[:, 0], predicts[:, 0]), log_loss(targets[:, 1], predicts[:, 1]), log_loss(targets[:, 2], predicts[:, 2])]
            scores.extend(ks_scores)
            scores.extend(log_loss_scores)
            
        return scores


    def evaluate_log_loss(self, data_loader):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for _, (x_dict, ys) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
                ys = ys.to(self.device)
                y_preds = self.model(x_dict)
                targets.extend(ys.tolist())
                predicts.extend(y_preds.tolist())
        targets, predicts = np.array(targets), np.array(predicts)

        _log_loss = log_loss(targets[:, 0], predicts[:, 0])

        return _log_loss


    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y_preds = model(x_dict)
                predicts.extend(y_preds.tolist())
        return predicts


    def get_bias_in_click_space(self, data_loader):
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, ys) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
                ys = ys.to(self.device)
                y_preds = self.model(x_dict)
                selected_y_preds = y_preds[ys[:, 1] == 1, :]
                
                predicts.extend(selected_y_preds.tolist())
        predicts = np.array(predicts)

        return np.sum(predicts)
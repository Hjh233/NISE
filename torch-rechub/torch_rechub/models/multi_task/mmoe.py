"""
Date: create on 04/05/2022
References: 
    paper: (KDD'2018) Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
    url: https://dl.acm.org/doi/pdf/10.1145/3219819.3220007
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch
import torch.nn as nn

from ...basic.layers import DCN4FeatureExtraction, DeepFM4FeatureExtraction, MLP, EmbeddingLayer, PredictionLayer

clip_min = 1e-15
clip_max = 1-1e-15
emb_dim_per_feature = 16

class MMOE(nn.Module):
    """Multi-gate Mixture-of-Experts model.

    Args:
        features (list): the list of `Feature Class`, training by the expert and tower module.
        task_types (list): types of tasks, only support `["classfication", "regression"]`.
        n_expert (int): the number of expert net.
        expert_params (dict): the params of all the expert module, keys include:`{"dims":list, "activation":str, "dropout":float}.
        tower_params_list (list): the list of tower params dict, the keys same as expert_params.
    """

    def __init__(self, features, task_types, n_expert, expert_params, tower_params_list):
        super().__init__()
        self.features = features
        self.task_types = task_types
        self.n_task = len(task_types)
        self.n_expert = n_expert
        self.embedding = EmbeddingLayer(features)
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.experts = nn.ModuleList(
            MLP(self.input_dims, output_layer=False, **expert_params) for i in range(self.n_expert))
        self.gates = nn.ModuleList(
            MLP(self.input_dims, output_layer=False, **{
                "dims": [self.n_expert],
                "activation": "softmax"
            }) for i in range(self.n_task))  #n_gate = n_task
        self.towers = nn.ModuleList(MLP(expert_params["dims"][-1], **tower_params_list[i]) for i in range(self.n_task))
        self.predict_layers = nn.ModuleList(PredictionLayer(task_type) for task_type in task_types)

    def forward(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)  #[batch_size, input_dims]
        expert_outs = [expert(embed_x).unsqueeze(1) for expert in self.experts
                      ]  #expert_out[i]: [batch_size, 1, expert_dims[-1]]
        expert_outs = torch.cat(expert_outs, dim=1)  #[batch_size, n_expert, expert_dims[-1]]
        gate_outs = [gate(embed_x).unsqueeze(-1) for gate in self.gates]  #gate_out[i]: [batch_size, n_expert, 1]

        ys = []
        for gate_out, tower, predict_layer in zip(gate_outs, self.towers, self.predict_layers):
            expert_weight = torch.mul(gate_out, expert_outs)  #[batch_size, n_expert, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  #[batch_size, expert_dims[-1]]
            tower_out = tower(expert_pooling)  #[batch_size, 1]
            y = predict_layer(tower_out)  #logit -> proba
            ys.append(torch.clamp(y, min=clip_min, max=clip_max))
        return torch.cat(ys, dim=1)


class DCN4MMOE(nn.Module):
    """
    Args:
        features (list): the list of `Feature Class`, training by the expert and tower module.
        task_types (list): types of tasks, only support `["classfication", "regression"]`.
        n_expert (int): the number of expert net.
        expert_params (dict): the params of all the expert module, keys include:`{"dims":list, "activation":str, "dropout":float}.
        tower_params_list (list): the list of tower params dict, the keys same as expert_params.
    """

    def __init__(self, features, task_types, n_expert, expert_params, tower_params_list):
        super().__init__()
        self.features = features
        self.task_types = task_types
        self.n_task = len(task_types)
        self.n_expert = n_expert
        self.embedding = EmbeddingLayer(features)
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.experts = nn.ModuleList(DCN4FeatureExtraction(self.input_dims, 1, expert_params) for _ in range(self.n_expert))
        self.gates = nn.ModuleList(
            MLP(self.input_dims, output_layer=False, **{
                "dims": [self.n_expert],
                "activation": "softmax"
            }) for _ in range(self.n_task))  #n_gate = n_task
        self.towers = nn.ModuleList(MLP(expert_params["dims"][-1], **tower_params_list[i]) for i in range(self.n_task))
        self.predict_layers = nn.ModuleList(PredictionLayer(task_type) for task_type in task_types)

    def forward(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)  #[batch_size, input_dims]
        expert_outs = [expert(embed_x).unsqueeze(1) for expert in self.experts
                      ]  #expert_out[i]: [batch_size, 1, expert_dims[-1]]
        expert_outs = torch.cat(expert_outs, dim=1)  #[batch_size, n_expert, expert_dims[-1]]
        gate_outs = [gate(embed_x).unsqueeze(-1) for gate in self.gates]  #gate_out[i]: [batch_size, n_expert, 1]

        ys = []
        for gate_out, tower, predict_layer in zip(gate_outs, self.towers, self.predict_layers):
            expert_weight = torch.mul(gate_out, expert_outs)  #[batch_size, n_expert, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  #[batch_size, expert_dims[-1]]
            tower_out = tower(expert_pooling)  #[batch_size, 1]
            y = predict_layer(tower_out)  #logit -> proba
            ys.append(torch.clamp(y, min=clip_min, max=clip_max))
        return torch.cat(ys, dim=1)


class DeepFM4MMOE(nn.Module):
    """Multi-gate Mixture-of-Experts model.

    Args:
        features (list): the list of `Feature Class`, training by the expert and tower module.
        task_types (list): types of tasks, only support `["classfication", "regression"]`.
        n_expert (int): the number of expert net.
        expert_params (dict): the params of all the expert module, keys include:`{"dims":list, "activation":str, "dropout":float}.
        tower_params_list (list): the list of tower params dict, the keys same as expert_params.
    """

    def __init__(self, features, task_types, n_expert, tower_params_list):
        super().__init__()
        self.features = features
        self.task_types = task_types
        self.n_task = len(task_types)
        self.n_expert = n_expert
        self.embedding = EmbeddingLayer(features)
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.experts = nn.ModuleList(DeepFM4FeatureExtraction(self.input_dims, emb_dim_per_feature) for _ in range(self.n_expert))
        self.gates = nn.ModuleList(
            MLP(self.input_dims, output_layer=False, **{
                "dims": [self.n_expert],
                "activation": "softmax"
            }) for i in range(self.n_task))  #n_gate = n_task
        self.towers = nn.ModuleList(MLP(emb_dim_per_feature, **tower_params_list[i]) for i in range(self.n_task))
        self.predict_layers = nn.ModuleList(PredictionLayer(task_type) for task_type in task_types)

    def forward(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)  #[batch_size, input_dims]
        embed_x_fm = self.embedding(x, self.features, squeeze_dim=False)
        expert_outs = [expert(embed_x, embed_x_fm).unsqueeze(1) for expert in self.experts
                      ]  #expert_out[i]: [batch_size, 1, expert_dims[-1]]
        # print('expert_outs', expert_outs[0].shape, expert_outs[1].shape)
        expert_outs = torch.cat(expert_outs, dim=1)  #[batch_size, n_expert, expert_dims[-1]]
        # print('expert_outs', expert_outs.shape)
        gate_outs = [gate(embed_x).unsqueeze(-1) for gate in self.gates]  #gate_out[i]: [batch_size, n_expert, 1]

        ys = []
        for gate_out, tower, predict_layer in zip(gate_outs, self.towers, self.predict_layers):
            expert_weight = torch.mul(gate_out, expert_outs)  #[batch_size, n_expert, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  #[batch_size, expert_dims[-1]]
            # print('expert_pooling', expert_pooling.shape)
            tower_out = tower(expert_pooling)  #[batch_size, 1]
            y = predict_layer(tower_out)  #logit -> proba
            # print('y', y.shape)
            ys.append(torch.clamp(y, min=clip_min, max=clip_max))
        return torch.cat(ys, dim=1)

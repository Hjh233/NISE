"""
Date: create on 11/13/2023
References: 
    paper: (ICDE'2023) DCMT: A Direct Entire-Space Causal Multi-Task Framework for Post-Click Conversion Estimation
"""

import torch
import torch.nn as nn

from ...basic.layers import MLP, EmbeddingLayer

clip_min = 1e-15
clip_max = 1-1e-15

class DCMT(nn.Module):
    """

    Args:
        user_features (list): the list of `Feature Class`, training by shared bottom and tower module. It means the user features.
        item_features (list): the list of `Feature Class`, training by shared bottom and tower module. It means the item features.
        cvr_params (dict): the params of the CVR Tower module, keys include:`{"dims":list, "activation":str, "dropout":float`}
        counterfactual_cvr_params (dict): the params of the CounterFactual CVR Tower module, keys include:`{"dims":list, "activation":str, "dropout":float`}
        ctr_params (dict): the params of the CTR Tower module, keys include:`{"dims":list, "activation":str, "dropout":float`}
    """

    def __init__(self, user_features, item_features, cvr_params, counterfactual_cvr_params, ctr_params):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.tower_dims = len(user_features) * user_features[0].embed_dim + len(item_features) * item_features[0].embed_dim
        self.tower_cvr = MLP(self.tower_dims, **cvr_params)
        self.tower_counterfactual_cvr = MLP(self.tower_dims, **counterfactual_cvr_params)
        self.tower_ctr = MLP(self.tower_dims, **ctr_params)

    def forward(self, x):
        # Here we concat all the features instead of field-wise pooling them
        # [batch_size, num_features, embed_dim] --> [batch_size, num_features * embed_dim]
        _batch_size = self.embedding(x, self.user_features, squeeze_dim=False).shape[0]
        embed_user_features = self.embedding(x, self.user_features, squeeze_dim=False).reshape(_batch_size, -1)
        embed_item_features = self.embedding(x, self.item_features, squeeze_dim=False).reshape(_batch_size, -1)

        input_tower = torch.cat((embed_user_features, embed_item_features), dim=1)

        cvr_logit = self.tower_cvr(input_tower)
        counterfactual_cvr_logit = self.tower_counterfactual_cvr(input_tower)
        ctr_logit = self.tower_ctr(input_tower)

        cvr_pred = torch.sigmoid(cvr_logit)
        counterfactual_cvr_pred = torch.sigmoid(counterfactual_cvr_logit)
        ctr_pred = torch.sigmoid(ctr_logit)
        ctcvr_pred = torch.mul(ctr_pred, cvr_pred)

        clipped_cvr_pred = torch.clamp(cvr_pred, min=clip_min, max=clip_max)
        clipped_counterfactual_cvr_pred = torch.clamp(counterfactual_cvr_pred, min=clip_min, max=clip_max)
        clipped_ctr_pred = torch.clamp(ctr_pred, min=clip_min, max=clip_max)

        ys = [cvr_pred, counterfactual_cvr_pred, ctr_pred, ctcvr_pred]
        clipped_ys = [clipped_cvr_pred, clipped_counterfactual_cvr_pred, clipped_ctr_pred, ctcvr_pred]

        # return torch.cat(ys, dim=1)
        return torch.cat(clipped_ys, dim=1)

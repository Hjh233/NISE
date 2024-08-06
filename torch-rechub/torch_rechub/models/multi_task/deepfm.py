import torch
import torch.nn as nn

from ...basic.layers import DeepFM4CVR, EmbeddingLayer

clip_min = 1e-15
clip_max = 1-1e-15

class DeepFM4ESMM(nn.Module):

    def __init__(self, 
                 user_features, 
                 item_features, 
                 cvr_mlp_params, 
                 ctr_mlp_params
                 ):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.tower_dims = len(user_features) * user_features[0].embed_dim + len(item_features) * item_features[0].embed_dim

        self.tower_cvr = DeepFM4CVR(feature_dims = self.tower_dims,
                                    mlp_params = cvr_mlp_params)
        self.tower_ctr = DeepFM4CVR(feature_dims = self.tower_dims,
                                    mlp_params = ctr_mlp_params)

    def forward(self, x):
        # [batch_size, num_features, embed_dim] --> [batch_size, num_features * embed_dim]
        _batch_size = self.embedding(x, self.user_features, squeeze_dim=False).shape[0]
        embed_user_features = self.embedding(x, self.user_features, squeeze_dim=False).reshape(_batch_size, -1)
        embed_item_features = self.embedding(x, self.item_features, squeeze_dim=False).reshape(_batch_size, -1)
        input_tower = torch.cat((embed_user_features, embed_item_features), dim=1)

        embed_user_features_fm = self.embedding(x, self.user_features, squeeze_dim=False)
        embed_item_features_fm = self.embedding(x, self.item_features, squeeze_dim=False)
        input_tower_fm = torch.cat((embed_user_features_fm, embed_item_features_fm), dim=1)

        cvr_pred = self.tower_cvr(input_tower, input_tower_fm)
        ctr_pred = self.tower_ctr(input_tower, input_tower_fm)
        ctcvr_pred = torch.mul(cvr_pred, ctr_pred)

        clipped_cvr_pred = torch.clamp(cvr_pred, min=clip_min, max=clip_max)
        clipped_ctr_pred = torch.clamp(ctr_pred, min=clip_min, max=clip_max)
        clipped_ctcvr_pred = torch.clamp(ctcvr_pred, min=clip_min, max=clip_max)

        clipped_ys = [clipped_cvr_pred, clipped_ctr_pred, clipped_ctcvr_pred]

        return torch.cat(clipped_ys, dim=1)


class DeepFM4DCMT(nn.Module):

    def __init__(self, 
                 user_features, 
                 item_features, 
                 cvr_mlp_params, 
                 counterfactual_cvr_mlp_params,
                 ctr_mlp_params
                 ):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.tower_dims = len(user_features) * user_features[0].embed_dim + len(item_features) * item_features[0].embed_dim

        self.tower_cvr = DeepFM4CVR(feature_dims = self.tower_dims,
                                    mlp_params = cvr_mlp_params)
        self.tower_counterfactual_cvr = DeepFM4CVR(feature_dims = self.tower_dims,
                                                   mlp_params = counterfactual_cvr_mlp_params)
        self.tower_ctr = DeepFM4CVR(feature_dims = self.tower_dims,
                                    mlp_params = ctr_mlp_params)

    def forward(self, x):
        # [batch_size, num_features, embed_dim] --> [batch_size, num_features * embed_dim]
        _batch_size = self.embedding(x, self.user_features, squeeze_dim=False).shape[0]
        embed_user_features = self.embedding(x, self.user_features, squeeze_dim=False).reshape(_batch_size, -1)
        embed_item_features = self.embedding(x, self.item_features, squeeze_dim=False).reshape(_batch_size, -1)
        input_tower = torch.cat((embed_user_features, embed_item_features), dim=1)

        embed_user_features_fm = self.embedding(x, self.user_features, squeeze_dim=False)
        embed_item_features_fm = self.embedding(x, self.item_features, squeeze_dim=False)
        input_tower_fm = torch.cat((embed_user_features_fm, embed_item_features_fm), dim=1)

        cvr_pred = self.tower_cvr(input_tower, input_tower_fm)
        counterfactual_cvr_pred = self.tower_counterfactual_cvr(input_tower, input_tower_fm)
        ctr_pred = self.tower_ctr(input_tower, input_tower_fm)
        ctcvr_pred = torch.mul(cvr_pred, ctr_pred)

        clipped_cvr_pred = torch.clamp(cvr_pred, min=clip_min, max=clip_max)
        clipped_counterfactual_cvr_pred = torch.clamp(counterfactual_cvr_pred, min=clip_min, max=clip_max)
        clipped_ctr_pred = torch.clamp(ctr_pred, min=clip_min, max=clip_max)
        clipped_ctcvr_pred = torch.clamp(ctcvr_pred, min=clip_min, max=clip_max)

        clipped_ys = [clipped_cvr_pred, clipped_counterfactual_cvr_pred, clipped_ctr_pred, clipped_ctcvr_pred]

        return torch.cat(clipped_ys, dim=1)
    

class DeepFM4DR(nn.Module):

    def __init__(self, 
                 user_features, 
                 item_features, 
                 cvr_mlp_params, 
                 ctr_mlp_params,
                 imputation_mlp_params
                 ):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.tower_dims = len(user_features) * user_features[0].embed_dim + len(item_features) * item_features[0].embed_dim

        self.tower_cvr = DeepFM4CVR(feature_dims = self.tower_dims,
                                    mlp_params = cvr_mlp_params)
        self.tower_ctr = DeepFM4CVR(feature_dims = self.tower_dims,
                                    mlp_params = ctr_mlp_params)
        self.tower_imputation = DeepFM4CVR(feature_dims = self.tower_dims,
                                           mlp_params = imputation_mlp_params)
        
    def forward(self, x):
        # [batch_size, num_features, embed_dim] --> [batch_size, num_features * embed_dim]
        _batch_size = self.embedding(x, self.user_features, squeeze_dim=False).shape[0]
        embed_user_features = self.embedding(x, self.user_features, squeeze_dim=False).reshape(_batch_size, -1)
        embed_item_features = self.embedding(x, self.item_features, squeeze_dim=False).reshape(_batch_size, -1)
        input_tower = torch.cat((embed_user_features, embed_item_features), dim=1)

        embed_user_features_fm = self.embedding(x, self.user_features, squeeze_dim=False)
        embed_item_features_fm = self.embedding(x, self.item_features, squeeze_dim=False)
        input_tower_fm = torch.cat((embed_user_features_fm, embed_item_features_fm), dim=1)

        cvr_pred = self.tower_cvr(input_tower, input_tower_fm)
        ctr_pred = self.tower_ctr(input_tower, input_tower_fm)
        ctcvr_pred = torch.mul(cvr_pred, ctr_pred)
        imputation_pred = self.tower_imputation(input_tower, input_tower_fm)

        clipped_cvr_pred = torch.clamp(cvr_pred, min=clip_min, max=clip_max)
        clipped_ctr_pred = torch.clamp(ctr_pred, min=clip_min, max=clip_max)
        clipped_ctcvr_pred = torch.clamp(ctcvr_pred, min=clip_min, max=clip_max)
        clipped_imputation_pred = torch.clamp(imputation_pred, min=clip_min, max=clip_max)

        clipped_ys = [clipped_cvr_pred, clipped_ctr_pred, clipped_ctcvr_pred, clipped_imputation_pred]

        return torch.cat(clipped_ys, dim=1)

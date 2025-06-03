# NISE

This repository contains the PyTorch implementation for "NISE" as well as other baseline methods for RecSys 2024 paper entitled "Utilizing Non-click Samples via Semi-supervised Learning for Conversion Rate Prediction
". Our goal is to facilitate the reproduction of the results presented in the paper. Our implementations leverage the public recommendation library, torch-rechub. More details about torch-rechub can be found [here](https://github.com/datawhalechina/torch-rechub).


## Installation

Follow these steps to set up your environment:

```bash
# Clone the repository
git clone git@github.com:Hjh233/NISE.git
cd NISE/torch-rechub

# Install the required packages
pip install -r requirements.txt
pip install -e .
```

## Environment Requirement
The code has been tested under Python 3.9.18 with the following dependencies:
* torch==2.1.1
* torch-rechub 0.0.2: a recommendation library, modified for this project. Ensure to install it in development mode with the previously mentioned command: `pip install -e .` Note: Do not use `pip install torch-rechub` directly as it will not include the necessary modifications.
* numpy
* pandas
* scikit-learn
* tqdm


## Datasets
We use two public datasets: Ali-CCP and KuaiPure.
* Ali-CCP is a benchmark dataset for conversion rate prediction, collected from traffic logs in Taobao platform, please refer to [official page](https://tianchi.aliyun.com/dataset/408) for more details.
* KuaiRand-Pure is an unbiased sequential recommendation dataset collected from the recommendation logs of the video-sharing mobile app, Kuaishou. Please refer to [official page](https://github.com/chongminggao/KuaiRand) for more details.

## Run the code
### Command Line Arguments

Below is a brief explanation of the command line arguments used to configure the model training:

- **party** : default is `both`, using all available features.
- **seed**: default is `0`, sets the random seed for reproducibility.
- **weight**: default is `50`, hyper-parameter, stands for \beta, please refer to the original paper for more details.
- **ablation_weight**: default is `1`, hyper-parameter, stands for \alpha, please refer to the original paper for more details.
- **strategy**: specifies the training strategy. Possible values include:
  - `esmm` for ESMM
  - `mmoe` for MMOE
  - `ips` for ESCM^2-IPS
  - `dr` for ESCM^2-DR
  - `dcmt` for DCMT
  - `adaptive_ucvrlc` for the proposed NISE
- **device**: `cpu` or `cuda`.
- **frac**: default is `1.0`, defines the ratio of negatively sampled samples used during training.

**Note**: For strategies using DeepFM or DCNv2 as the backbone, the strategy argument should be formatted as `deepfm_xxx` or `dcn_xxx`.

### Main experiments
* For dataset Ali-CCP:

```bash
# MLP as the backbone model, use NISE
python mlp_model.py --party both --seed 0 --weight 50 --strategy adaptive_ucvrlc --ablation_weight 1 --device cuda:0 --frac 1.0

# DeepFM as the backbone model, use ESMM
python deepfm_model.py --party both --seed 0 --weight 50 --strategy deepfm_esmm --ablation_weight 1 --device cuda:0

# DCNv2 as the backbone model, use ESCM^2-IPS
python dcn_model.py --party both --seed 0 --weight 50 --strategy dcn_ips --ablation_weight 1 --device cuda:0
```

### Alternative strategies for dynamic weighting
We compare dynamic task prioritization (DTP) loss and dynamic weight average (DWA) strategy with our proposed dynamic weighting strategy. Specifically, we utilize MLP, DeepFM, DCNv2 as backbones and use Ali-CCP as the dataset. We compare the AUC, LogLoss and KS of CVR task between DTP, DWA and NISE(ours). 
We refer to https://github.com/lorenmt/auto-lambda/blob/main/trainer_dense.py to implement DWA and implement DTP on our own.
```bash
# DTP Ali-CCP
# MLP
python mlp_model.py --party both --seed 0 --weight 50 --strategy dtp --ablation_weight 1 --device cuda:0 --frac 1.0
# DeepFM
python deepfm_model.py --party both --seed 0 --weight 50 --strategy deepfm_dtp --ablation_weight 1 --device cuda:0
# DCN
python dcn_model.py --party both --seed 0 --weight 50 --strategy dcn_dtp --ablation_weight 1 --device cuda:0

# DWA Ali-CCP
# MLP
python mlp_model.py --party both --seed 0 --weight 50 --strategy dwa --ablation_weight 1 --device cuda:0 --frac 1.0
# DeepFM
python deepfm_model.py --party both --seed 0 --weight 50 --strategy deepfm_dwa --ablation_weight 1 --device cuda:0
# DCN
python dcn_model.py --party both --seed 0 --weight 50 --strategy dcn_dwa --ablation_weight 1 --device cuda:0
```

### Ablation Study
We use the Ali-CCP dataset and use three backbone models: MLP, DeepFM and DCNv2
```bash
# MLP as the backbone model, test NISE-1
python mlp_model.py --party both --seed 0 --weight 50 --strategy ablation_1 --ablation_weight 1 --device cuda:0 --frac 1.0

# DeepFM as the backbone model, use NISE-2
python deepfm_model.py --party both --seed 0 --weight 50 --strategy ablation_2 --ablation_weight 1 --device cuda:0

# DCNv2 as the backbone model, use NISE-3
python dcn_model.py --party both --seed 0 --weight 50 --strategy ablation_3 --ablation_weight 1 --device cuda:0
```


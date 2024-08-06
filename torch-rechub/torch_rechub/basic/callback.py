import copy


class EarlyStopper(object):
    """Early stops the training if validation loss doesn't improve after a given patience.
        
    Args:
        patience (int): How long to wait after last time validation auc improved.
    """

    def __init__(self, patience, epoch=10):
        self.patience = patience
        self.epoch = epoch
        self.trial_counter = 0
        self.best_auc = 0
        self.best_weights = None

    def stop_training(self, val_auc, weights, cur_epoch):
        """whether to stop training.

        Args:
            val_auc (float): auc score in val data.
            weights (tensor): the weights of model
        """
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(weights)

        if cur_epoch == self.epoch or self.trial_counter >= self.patience:
            return True
        else:
            self.trial_counter += 1
            return False

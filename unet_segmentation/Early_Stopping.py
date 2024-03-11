import numpy as np

class EarlyStopping:
    def __init__(self, patience=None, verbose=True, delta=0,  trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score1 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.max_score = 0
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss,val_metric):

        score = -val_loss
        score1=-val_metric

        if (self.best_score is None) and (self.best_score1 is None):
            self.best_score = score
            self.best_score1 = score1
            self.verbose_(val_loss,val_metric)
        elif (score < self.best_score + self.delta) or (score1 > self.best_score1 + self.delta):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score1 = score1
            self.verbose_(val_loss,val_metric)
            self.counter = 0

    def verbose_(self, val_loss,val_metric):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.trace_func(f'Validation metric increased ({self.max_score:.6f} --> {val_metric:.6f}).')
        self.val_loss_min = val_loss
        self.max_score = val_metric
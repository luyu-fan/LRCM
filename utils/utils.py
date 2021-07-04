from torch.optim.lr_scheduler import _LRScheduler
import torch

class WarmupWrapper(_LRScheduler):
    
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    ref: https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    
    """

    def __init__(self,optimizer,warm_start_lr,warmup_total_epochs,scheduler = None):

        self.warm_start_lr = warm_start_lr
        self.total_warmup_epochs = warmup_total_epochs
        self.scheduler = scheduler

        self.warmup_finish = False

        super(WarmupWrapper,self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_warmup_epochs:
            if self.scheduler is not None:
                if not self.warmup_finish:
                    self.warmup_finish = True
                    self.scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                return self.scheduler.get_last_lr()
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [self.warm_start_lr + (base_lr - self.warm_start_lr) * (self.last_epoch / self.total_warmup_epochs) for base_lr in self.base_lrs]

    def step(self,epoch = None):
        if self.warmup_finish and self.scheduler is not None:
            if epoch == None:
                self.scheduler.step(epoch)
            else:
                self.scheduler.step(epoch - self.total_warmup_epochs)
            self._last_lr = self.scheduler.get_last_lr()
        else:
            return super(WarmupWrapper, self).step(epoch)

def plot(lr_lists):

    import matplotlib.pyplot as plt
    import seaborn
    plt.figure(dpi = 200)
    for lr in lr_lists:
        seaborn.lineplot(x = list(range(len(lr))),y = lr)
    plt.show()

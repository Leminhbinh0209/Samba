import os
import numpy as np
import random
import torch
import random

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
   
def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)

def numel_2(parameters, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=[1]):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def save_model(save_dir, epoch, model, optimizer,  best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filename = str('_'.join([save_dir, 'current.pth']))
    torch.save(state, filename)

    if best:
        filename = str('_'.join([save_dir, 'best.pth']))
        torch.save(state, filename)

def stratified_train_val_split(set_labels, val_size=0.2):
    # print('Splitting to TRAIN and VAL set. TOTAL Videos: %d' % (len(set_labels)))
    # Declare vaiables
    indices_train, indices_val = list(), list()

    # Get the indices of the Appropriate videos
    indices_appropriate = [i for i, x in enumerate(set_labels) if x == 0.0]

    # Get the indices of the Disturbing videos
    indices_disturbing = [i for i, x in enumerate(set_labels) if x == 1.0]

    # APPROPRIATE
    total_appropriate_train = int(len(indices_appropriate) * (1 - val_size))
    indices_train = indices_appropriate[0:total_appropriate_train]
    indices_val = indices_appropriate[total_appropriate_train:len(indices_appropriate)]

    # DISTURBING
    total_disturbing_train = int(len(indices_disturbing) * (1 - val_size))
    indices_train += indices_disturbing[0:total_disturbing_train]
    indices_val += indices_disturbing[total_disturbing_train:len(indices_disturbing)]


    print('--- [TRAIN_VAL_SPLIT] TOTAL VIDEOS: %d | TOTAL TRAIN: %d, TOTAL VAL: %d' % (len(set_labels), len(indices_train), len(indices_val)))
    return indices_train, indices_val

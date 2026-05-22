import numpy as np
import torch
import random
import cv2

def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tensor2cv(input: torch.Tensor):
    input = input.clone().detach().cpu().squeeze()
    input = input.mul_(255).add_(0.5).clamp_(0,255).permute(1, 2, 0).type(torch.uint8).numpy()
    input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
    return input

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def judge_and_remove_module_dict(load_state_dict, remove_key='module.'):
    new_dict = {}
    for old_key, value in load_state_dict.items():
        if old_key.startswith(remove_key):
            new_key = old_key.replace(remove_key, '')
            new_dict[new_key] = value
        else:
            new_dict[old_key] = value
    return new_dict
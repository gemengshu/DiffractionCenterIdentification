import numpy as np

import torch
import torch.nn as nn
import random

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from train_mstrans import train_mstrans

if __name__ == '__main__':
    train_mstrans()
'''
## Code for transformer architecture (from scratch)
from https://nlp.seas.harvard.edu/annotated-transformer/
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
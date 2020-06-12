import torch
from torch import nn

EPS=1e-12

def bpr_loss(prediction_i, prediction_j):
#     sig = nn.Sigmoid()
    prob = torch.clamp((prediction_i - prediction_j).sigmoid(), min=EPS)
    bpr = - prob.log().sum()
    return bpr


def calc_map(topk, positive):
    ap = 0
    count = 0
    for i, item in enumerate(topk):
        if item in positive:
            count += 1
            ap += count/(i+1)
    ap /= (count + 1e-8)
    
    return ap
        
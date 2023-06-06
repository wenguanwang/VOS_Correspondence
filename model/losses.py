import torch
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_util import compute_tensor_iu
import math
from collections import defaultdict

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.eps = 1e-8

    def forward(self, logits, labels):
        num_class = logits.size(1)
        logits = F.softmax(logits, dim=1)
        labels_one_hot = F.one_hot(labels, num_class).permute(0,3,1,2).contiguous()

        dice_loss = 0.0
        for i in range(num_class):
            dice_loss_i = dice_loss_helper(logits[:, i], labels_one_hot[:, i], self.smooth, self.eps)
            dice_loss += dice_loss_i
        dice_loss = dice_loss / num_class

        return dice_loss


def dice_loss_helper(logit, label, smooth, eps):
    assert logit.shape == label.shape, \
        "The shape of logit and label should be the same"
    logit = logit.flatten(1)
    label = label.flatten(1)
    intersection = torch.sum(logit * label, dim=1)
    cardinality = torch.sum(logit + label, dim=1)
    dice_loss = 1 - (2 * intersection + smooth) / (cardinality + smooth + eps)
    dice_loss = dice_loss.mean()
    return dice_loss



def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)

def get_sec_iou_hook(values):
    return 'iou/sec_iou', (values['hide_iou/sec_i']+1)/(values['hide_iou/sec_u']+1)

iou_hooks_so = [
    get_iou_hook,
]

iou_hooks_mo = [
    get_iou_hook,
    get_sec_iou_hook,
]


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class LossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.bce = BootstrappedCE()
        self.dice = DiceLoss()

    def compute(self, data, it):
        losses = defaultdict(int)

        b, s, _, _, _ = data['gt'].shape
        selector = data.get('selector', None)

        for i in range(1, s):
            # Have to do it in a for-loop like this since not every entry has the second object
            # Well it's not a lot of iterations anyway
            for j in range(b):
                if selector is not None and selector[j][1] > 0.5:
                    loss, p = self.bce(data['logits_%d'%i][j:j+1], data['cls_gt'][j:j+1,i], it)
                    loss += 0.4*self.dice(data['logits_%d'%i][j:j+1], data['cls_gt'][j:j+1,i])
                else:
                    loss, p = self.bce(data['logits_%d'%i][j:j+1,:2], data['cls_gt'][j:j+1,i], it) 
                    loss += 0.4*self.dice(data['logits_%d'%i][j:j+1,:2], data['cls_gt'][j:j+1,i])

                losses['loss_%d'%i] += loss / b
                losses['p'] += p / b / (s-1)

            losses['total_loss'] += losses['loss_%d'%i]

            new_total_i, new_total_u = compute_tensor_iu(data['mask_%d'%i]>0.5, data['gt'][:,i]>0.5)
            losses['hide_iou/i'] += new_total_i
            losses['hide_iou/u'] += new_total_u

            if selector is not None:
                new_total_i, new_total_u = compute_tensor_iu(data['sec_mask_%d'%i]>0.5, data['sec_gt'][:,i]>0.5)
                losses['hide_iou/sec_i'] += new_total_i
                losses['hide_iou/sec_u'] += new_total_u
        losses['cvloss'] = data['cross_key']
        losses['stloss'] = data['temp']
        
#         if it<20000:
#             factor=0
#         elif it<80000:
#             factor = float(it-20000)/60000.0
#         else:
#             factor = 1.0
#         losses['total_loss'] = losses['total_loss']  + 0.04*factor*losses['cvloss'] + 0.20*factor*losses['stloss']
        
        if it<30000:
            factor=0
        elif it<90000:
            factor = float(it-30000)/60000.0
#         elif it<125000:
#             factor = 1.0
        else:
            factor = 1.0
        losses['total_loss'] = losses['total_loss']  + 0.04*factor*losses['cvloss'] + 0.20*factor*losses['stloss']

        return losses
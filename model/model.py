"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.network import STCN
from model.losses import LossComputer, iou_hooks_mo, iou_hooks_so
from util.log_integrator import Integrator
from util.image_saver import pool_pairs
import math

class STCNModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        self.para = para
        self.single_object = para['single_object']
        self.local_rank = local_rank

        self.STCN = nn.parallel.DistributedDataParallel(
            STCN(self.single_object).cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        if self.single_object:
            self.train_integrator.add_hook(iou_hooks_so)
        else:
            self.train_integrator.add_hook(iou_hooks_mo)
        self.loss_computer = LossComputer(para)
        self.train()
        
        def get_parameter_groups(model):
            groups = ([], [], [])
            for name, value in model.named_parameters():
                if 'key_encoder' in name or 'value_encoder' in name:
                    if 'layer2' in name or 'layer3' in name:
                        groups[1].append(value)
                    else:
                        groups[0].append(value)
                else:
                    groups[2].append(value)
            return groups
        param_groups = get_parameter_groups(self.STCN)
        self.optimizer = optim.AdamW([{'params': param_groups[0], 'lr': 0.2*para['lr']},
            {'params': param_groups[1], 'lr': 0.4*para['lr']},
            {'params': param_groups[2]}], lr=para['lr'], weight_decay=1e-7)
        
#         self.optimizer = optim.Adam(filter(
#             lambda p: p.requires_grad, self.STCN.parameters()), lr=para['lr'], weight_decay=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])
        if para['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 800
        self.save_model_interval = 50000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1
        self.GRID_SIZE = 8
        self.TEMP = 0.05
        self.eye = None
        self.GRID_SIZE_REF = 4


    def get_aff(self, mk, qk):
        CK, BTHW = mk.shape
        CK, BNN = qk.shape
        if self.para['amp']:
            a_sq = mk.float().pow(2).sum(0).unsqueeze(1) #BTHW, 1
            b_sq = qk.float().pow(2).sum(0).unsqueeze(0)  #1, BNN
            ab = mk.transpose(0,1) @ qk # BTHW, BNN  
            affinity = (2*ab-a_sq-b_sq) / math.sqrt(CK)   # BTHW, BNN 应该对BTHW维度做softmax
            return affinity.half()
        else:
            a_sq = mk.pow(2).sum(0).unsqueeze(1) #BTHW, 1
            b_sq = qk.pow(2).sum(0).unsqueeze(0)  #1, BNN
            ab = mk.transpose(0,1) @ qk # BTHW, BNN  
            affinity = (2*ab-a_sq-b_sq) / math.sqrt(CK)   # BTHW, BNN 应该对BTHW维度做softmax
            return affinity

    def _align(self, x, t):
        if self.para['amp']:
            tf = F.affine_grid(t, size=x.size(), align_corners=False).half()
        else:
            tf = F.affine_grid(t, size=x.size(), align_corners=False)
            x = x.float()
            tf = tf.float()
        return F.grid_sample(x, tf, align_corners=False, mode="nearest")

    def _sample_index(self, x, T, N):

        BT,K,H,W = x.shape
        B = x.view(-1,T,K,H*W).shape[0]

        # sample indices from a uniform grid
        xs, ys = W // N, H // N
        x_sample = torch.arange(0, W, xs).view(1, 1, N)
        y_sample = torch.arange(0, H, ys).view(1, N, 1)

        # Random offsets
        # [B x 1 x N]
        x_sample = x_sample + torch.randint(0, xs, (B, 1, 1))
        # [B x N x 1]
        y_sample = y_sample + torch.randint(0, ys, (B, 1, 1))

        # batch index
        # [B x N x N]
        hw_index = torch.LongTensor(x_sample + y_sample * W)

        return hw_index

    def _sample_from(self, x, index, T, N):

        BT,K,H,W = x.shape
        x = x.view(-1,T,K,H*W)
        B = x.shape[0]

        # > [B,T,K,HW] > [B,T,HW,K] > [B,THW,K]
        x = x.permute([0,1,3,2]).reshape(B,-1,K)

        # every video clip will have the same samples
        # on the grid
        # [B x N x N] -> [B x N*N x 1] -> [B x N*N x K]
        index = index.view(B,-1,1).expand(-1,-1,K)

        # selecting from the uniform grid
        y = x.gather(1, index.to(x.device))

        # [BNN,K]
        return y.flatten(0,1)


    def _pseudo_mask(self, logits, T):
        BT,K,h,w = logits.shape
        assert BT % T == 0, "Batch not divisible by sequence length"
        B = BT // T

        # N = [G x G]
        N = self.GRID_SIZE ** 2

        # generating a pseudo label
        # first we need to mask out the affinities across the batch
        if self.eye is None or self.eye.shape[0] != B*T \
                            or self.eye.shape[1] != B*N:
            eye = torch.eye(B)[:,:,None].expand(-1,-1,N).reshape(B,-1)
            eye = eye.unsqueeze(1).expand(-1,T,-1).reshape(B*T, B*N, 1, 1)
            self.eye = eye.to(logits.device)

        maxes = torch.max(logits, dim=1, keepdim=True)[0]    #对BNN维度做softmax
        x_exp = torch.exp(logits - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        probs = x_exp / x_exp_sum 
        return probs * self.eye

    def _cluster_grid(self, k1, k2, aff1, aff2, T, index=None):

        BT,K,H,W = k1.shape

        assert BT % T == 0, "Batch not divisible by sequence length"
        B = BT // T

        # N = [G x G]
        N = self.GRID_SIZE ** 2

        # [BT,K,H,W] -> [BTHW,K]
        flatten = lambda x: x.flatten(2,3).permute([0,2,1]).flatten(0,1)

        # [BTHW,BNN] -> [BT,BNN,H,W]
        def unflatten(x, aff=None):
            x = x.view(BT,H*W,-1).permute([0,2,1]).view(BT,-1,H,W)  #[BT, BNN, H, W]
            if aff is None:
                return x
            return self._align(x, aff)

        index = self._sample_index(k1, T, N = self.GRID_SIZE)  
        query1 = self._sample_from(k1, index, T, N = self.GRID_SIZE)       #[BNN,K]

        """Computing the distances and pseudo labels"""

        k1_ = flatten(k1)  # [BTHW,K]
        k2_ = flatten(k2)  # [BTHW,K]

        # [BTHW,BN] -> [BTHW,BN] -> [BT,BN,H,W], query1:[BNN, K] , k1_:[BTHW, K]
        vals_soft = unflatten(self.get_aff(k1_.transpose(0,1), query1.transpose(0,1)), aff1)       #_key_val return: [BTHW,BNN]
        vals_pseudo = unflatten(self.get_aff(k2_.transpose(0,1), query1.transpose(0,1)), aff2)   
        # [BT,BNN,H,W]
        probs_pseudo = self._pseudo_mask(vals_pseudo, T)  #input:[BT, BNN, H, W], 
        probs_pseudo2 = self._pseudo_mask(vals_soft, T)

        pseudo = probs_pseudo.argmax(1)
        pseudo2 = probs_pseudo2.argmax(1)

        return vals_soft, pseudo, index
    
    def stcn_softmax(self, affinity):
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum 
        return affinity
    
    def _ref_loss(self, x, y, N = 4, eps=1e-8):
        B,_,h,w = x.shape  #[B,K,H,W]
        
        index = self._sample_index(x, T=1, N=N)  #[B,NN,K]
        x1 = self._sample_from(x, index, T=1, N=N)   #[BNN,K]
        y1 = self._sample_from(y, index, T=1, N=N) 
        #print('x1:', x1.shape)
        x1 = x1.transpose(0,1) #[64,128]
        y1 = y1.transpose(0,1) #[64,128]
        #logits = torch.mm(x1, y1.t()) / self.TEMP  
        logits = self.get_aff(x1, y1)  

        labels = torch.arange(logits.size(1)).to(logits.device)
        #拆分成softmax+nll时，对y1_hw维度做softmax,也就是说y是reference
        logits_softmax = self.stcn_softmax(logits)
        return F.nll_loss(torch.log(logits_softmax+eps), labels)

    def _ce_loss(self, x, pseudo_map, T, eps=1e-8):
        x_softmax = self.stcn_softmax(x)
        error_map = F.nll_loss(torch.log(x_softmax+eps), pseudo_map, reduction="none", ignore_index=-1)

        BT,h,w = error_map.shape
        errors = error_map.view(-1,T,h,w)
        error_ref, error_t = errors[:,0], errors[:,1:]

        return error_ref.mean(), error_t.mean(), error_map

    def fetch_first(self, x1, x2, T):
        assert x1.shape[1:] == x2.shape[1:]
        c,h,w = x1.shape[1:]

        x1 = x1.view(-1,T+1,c,h,w)
        x2 = x2.view(-1,T-1,c,h,w)

        x2 = torch.cat((x1[:,-1:], x2), 1)
        x1 = x1[:,:-1]

        return x1.flatten(0,1), x2.flatten(0,1)

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['rgb_view1']
        B,T,C,H,W = Fs.shape
        Ms = data['gt']
        Fs2 = data['rgb_view2']
        view1_aff = data['view2_inv_aff']
        view2_aff = data['view1_aff']
        view1_aff = view1_aff.flatten(0,1).cuda()
        view2_aff = view2_aff.flatten(0,1).cuda()
        images1 = torch.cat((Fs, Fs2[:, ::T]), 1)
        images1 = images1.cuda()
        images2 = Fs2[:, 1:].cuda()

        with torch.cuda.amp.autocast(enabled=self.para['amp']):
            key1, k16, kf16, kf8, kf4, T1 = self.STCN('encode_key', images1)
            with torch.no_grad():
                key2, _, _, _, _, _ = self.STCN('encode_key', images2)

            key1, key2 = self.fetch_first(key1, key2, T)
            vals, pseudo, index = self._cluster_grid(key1, key2, view1_aff, view2_aff, T)

            key1_aligned = self._align(key1, view1_aff) 
            key2_aligned = self._align(key2, view2_aff)
            n_ref = self.GRID_SIZE_REF

            out["cross_key"] = self._ref_loss(key1_aligned[::T], key2_aligned[::T], N = n_ref)
            _, out["temp"], out["error_map"] = self._ce_loss(vals, pseudo, T)
  
            if self.single_object:
                ref_v = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0])

                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN('segment', 
                        k16[:,:,1], kf16[:,1], kf8[:,1], kf4[:,1], 
                        k16[:,:,0:1], ref_v)
                prev_v = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask)

                values = torch.cat([ref_v, prev_v], 2)

                del ref_v

                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask = self.STCN('segment', 
                        k16[:,:,2], kf16[:,2], kf8[:,2], kf4[:,2], 
                        k16[:,:,0:2], values)

                out['mask_1'] = prev_mask
                out['mask_2'] = this_mask
                out['logits_1'] = prev_logits
                out['logits_2'] = this_logits
            else:
                sec_Ms = data['sec_gt']
                selector = data['selector']

                ref_v1 = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0], sec_Ms[:,0])
                ref_v2 = self.STCN('encode_value', Fs[:,0], kf16[:,0], sec_Ms[:,0], Ms[:,0])
                ref_v = torch.stack([ref_v1, ref_v2], 1)

                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN('segment', 
                        k16[:,:,1], kf16[:,1], kf8[:,1], kf4[:,1], 
                        k16[:,:,0:1], ref_v, selector)
                
                prev_v1 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,0:1], prev_mask[:,1:2])
                prev_v2 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,1:2], prev_mask[:,0:1])
                prev_v = torch.stack([prev_v1, prev_v2], 1)
                values = torch.cat([ref_v, prev_v], 3)

                del ref_v

                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask = self.STCN('segment', 
                        k16[:,:,2], kf16[:,2], kf8[:,2], kf4[:,2], 
                        k16[:,:,0:2], values, selector)

                out['mask_1'] = prev_mask[:,0:1]
                out['mask_2'] = this_mask[:,0:1]
                out['sec_mask_1'] = prev_mask[:,1:2]
                out['sec_mask_2'] = this_mask[:,1:2]

                out['logits_1'] = prev_logits
                out['logits_2'] = this_logits

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.save_im_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)

            if self._is_train:
                if (it) % self.report_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_model_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save(it)

            # Backward pass
            # This should be done outside autocast
            # but I trained it like this and it worked fine
            # so I am keeping it this way for reference
            self.optimizer.zero_grad(set_to_none=True)
            if self.para['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward() 
                self.optimizer.step()
            self.scheduler.step()
            ##################################################################################################################
#             self.ema_STCN.update()
    def model_state_gain(self):
        return self.STCN.module.state_dict()

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.STCN.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)
        ##################################################################################################################
#         model_path_ema = self.save_path + ('_%s_ema.pth' % it)
#         torch.save(self.ema_STCN.ema_model.module.state_dict(), model_path_ema)
#         print('EMA Model saved to %s.' % model_path_ema)

#         self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + ('_%s_checkpoint.pth' % it)
        checkpoint = { 
            'it': it,
            'network': self.STCN.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.STCN.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

#         Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)
                    
#         for k in list(src_dict.keys()):
#             if k == 'value_encoder.conv1.weight':
#                 if src_dict[k].shape[1] == 5:
#                     src_dict[k] = src_dict[k][:,:4]

        self.STCN.module.load_state_dict(src_dict, strict=False)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Shall be in eval() mode to freeze BN parameters
        self.STCN.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.STCN.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.STCN.eval()
        return self


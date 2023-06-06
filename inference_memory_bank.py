import math
import torch


def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    # The types should be the same already
    # some people report an error here so an additional guard is added
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

    return x


class MemoryBank:
    def __init__(self, k, top_k=20):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None

        self.num_objects = k
        self.max_dis = 7
        self.padded_local_mask = None
        self.local_mask = None
        self.last_size_2d = None
        
    def compute_mask(self, height, width, device=None):
        pad_height = height + 2 * self.max_dis
        pad_width = width + 2 * self.max_dis

        if self.padded_local_mask is not None and (height,
                                                   width) == self.last_size_2d:
            padded_local_mask = self.padded_local_mask
            local_mask = self.local_mask

        else:
            ky, kx = torch.meshgrid([
                torch.arange(0, pad_height, device=device),
                torch.arange(0, pad_width, device=device)
            ])
            qy, qx = torch.meshgrid([
                torch.arange(0, height, device=device),
                torch.arange(0, width, device=device)
            ])

            qy = qy.reshape(-1, 1)
            qx = qx.reshape(-1, 1)
            offset_y = qy - ky.reshape(1, -1) + self.max_dis
            offset_x = qx - kx.reshape(1, -1) + self.max_dis
            padded_local_mask = (offset_y.abs() <= self.max_dis) & (
                offset_x.abs() <= self.max_dis)
            padded_local_mask = padded_local_mask.view(1, 1, height * width,
                                                       pad_height, pad_width)
            local_mask = padded_local_mask[:, :, :, self.max_dis:-self.max_dis,
                                           self.max_dis:-self.max_dis]
#             pad_pixel = self.max_dis * self.dilation
#             local_mask = F.pad(local_mask.float(),
#                                (pad_pixel, pad_pixel, pad_pixel, pad_pixel),
#                                mode='constant',
#                                value=0).view(1, 1, height * width, pad_height,
#                                              pad_width)
            self.padded_local_mask = padded_local_mask
            self.local_mask = local_mask

        return padded_local_mask, local_mask
    
        
    def _local_matching(self, mk, qk, H, W):
        B, CK, NE = mk.shape
        T = NE//H//W

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, NE, HW
        
        _, local_mask = self.compute_mask(H,W,affinity.device) # 1,1,H*W, H,W
        local_mask = local_mask[0].flatten(2).transpose(1,2).repeat(B,T,1).float()
        affinity = affinity * local_mask
        
        # softmax operation; aligned the evaluation style
        affinity = softmax_w_top(affinity, top=self.top_k)  # B, NE, HW

        return affinity
    

    def _global_matching(self, mk, qk):
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, NE, HW
        affinity = softmax_w_top(affinity, top=self.top_k)  # B, NE, HW

        return affinity

    def _readout(self, affinity, mv):
        return torch.bmm(mv, affinity)

    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)
        
        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v

        affinity = self._global_matching(mk, qk)

        # One affinity for all
        readout_mem = self._readout(affinity.expand(k,-1,-1), mv)

        return readout_mem.view(k, self.CV, h, w)

    def add_memory(self, key, value, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)


# import math
# import torch
# import numpy as np
# import torch.nn.functional as F

# def ToCuda(xs):
#     if torch.cuda.is_available():
#         if isinstance(xs, list) or isinstance(xs, tuple):
#             return [x.cuda(non_blocking=True) for x in xs]
#         else:
#             return xs.cuda(non_blocking=True) 
#     else:
#         return 
    
# def softmax_w_top(x, top):
#     values, indices = torch.topk(x, k=top, dim=1)
#     x_exp = values.exp_()

#     x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
#     # The types should be the same already
#     # some people report an error here so an additional guard is added
#     x.zero_().scatter_(1, indices, x_exp.type(x.dtype)) # B * THW * HW

#     return x


# class MemoryBank:
#     def __init__(self, k, top_k=20):
#         self.top_k = top_k

#         self.CK = None
#         self.CV = None

#         self.mem_k = None
#         self.mem_v = None

#         self.num_objects = k
        
#         self.gaussian_kernel = 3
#         self.gaussian_kernel_flow_window = 7
#         if self.gaussian_kernel != -1:
#             self.feature_H = -1
#             self.feature_W = -1
#             if self.gaussian_kernel_flow_window != -1:
#                 self.H_flow = -1
#                 self.W_flow = -1
#                 self.T_flow = 1e+7
#                 self.B_flow = -1

#     def apply_gaussian_kernel(self, corr, h, w, sigma_factor=1.):
#         b, hwt, hw = corr.size()

#         idx = corr.max(dim=2)[1] # b x h2 x w2
#         idx_y = (idx // w).view(b, hwt, 1, 1).float()
#         idx_x = (idx % w).view(b, hwt, 1, 1).float()
        
#         if h != self.feature_H:
#             self.feature_H = h
#             y_tmp = np.linspace(0,h-1,h)
#             self.y = ToCuda(torch.FloatTensor(y_tmp))
#         y = self.y.view(1,1,h,1).expand(b, hwt, h, 1)

#         if w != self.feature_W:
#             self.feature_W = w
#             x_tmp = np.linspace(0,w-1,w)
#             self.x = ToCuda(torch.FloatTensor(x_tmp))
#         x = self.x.view(1,1,1,w).expand(b, hwt, 1, w)
                
#         gauss_kernel = torch.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * (self.gaussian_kernel*sigma_factor)**2))
#         gauss_kernel = gauss_kernel.view(b, hwt, hw)

#         return gauss_kernel, idx
    
#     def _global_matching(self, mk, qk, H, W):
#         # NE means number of elements -- typically T*H*W
#         B, CK, NE = mk.shape
        
#         T = NE//H//W

#         # See supplementary material
#         a_sq = mk.pow(2).sum(1).unsqueeze(2)
#         ab = mk.transpose(1, 2) @ qk

#         affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, NE, HW
# #         affinity = softmax_w_top(affinity, top=self.top_k)  # B, NE, HW

#         if self.gaussian_kernel != -1:
#             if self.gaussian_kernel_flow_window != -1:
#                 affinity_tmp = affinity[:,int(-H*W):].clone()
#                 if (self.B_flow != B) or (self.T_flow != T) or (self.H_flow != H) or (self.W_flow != W):
#                     hide_non_local_qk_map_tmp = torch.ones(B,1,H,W,H,W).bool()
#                     window_size_half = (self.gaussian_kernel_flow_window-1) // 2
#                     for h_idx1 in range(H):
#                         for w_idx1 in range(W):
#                             h_left = max(h_idx1-window_size_half, 0)
#                             h_right = h_idx1+window_size_half+1
#                             w_left = max(w_idx1-window_size_half, 0)
#                             w_right = w_idx1+window_size_half+1
#                             hide_non_local_qk_map_tmp[:,0,h_idx1,w_idx1,h_left:h_right,w_left:w_right] = False
#                     hide_non_local_qk_map_tmp = hide_non_local_qk_map_tmp.view(B,H*W,H*W)
#                     self.hide_non_local_qk_map_flow = ToCuda(hide_non_local_qk_map_tmp)
#                 if (self.B_flow != B) or (self.T_flow > T) or (T==1) or (self.H_flow != H) or (self.W_flow != W):
#                     self.max_idx_stacked = None
#                 affinity_tmp.masked_fill_(self.hide_non_local_qk_map_flow, float('-inf'))
#                 gauss_kernel_map, max_idx = self.apply_gaussian_kernel(affinity_tmp, h=H, w=W)
#                 if self.max_idx_stacked is None:
#                     self.max_idx_stacked = max_idx
#                 else:
#                     if self.T_flow == T:
#                         self.max_idx_stacked = self.max_idx_stacked[:,:int(-H*W)]
#                     self.max_idx_stacked = torch.gather(max_idx, dim=1, index=self.max_idx_stacked)
#                     for t_ in range(1, T):
#                         gauss_kernel_map_tmp, _ = self.apply_gaussian_kernel(affinity_tmp, h=H, w=W, sigma_factor=(t_*0.5)+1)
#                         gauss_kernel_map_tmp = torch.gather(gauss_kernel_map_tmp, dim=1, index=self.max_idx_stacked[:,int((T-t_-1)*H*W):int((T-t_)*H*W)].unsqueeze(-1).expand(-1,-1,int(H*W)))
#                         gauss_kernel_map = torch.cat((gauss_kernel_map_tmp, gauss_kernel_map), dim=1)
#                     self.max_idx_stacked = torch.cat((self.max_idx_stacked, max_idx), dim=1)
#                 self.T_flow = T
#                 self.H_flow = H
#                 self.W_flow = W
#                 self.B_flow = B
#             else:
#                 gauss_kernel_map, _ = self.apply_gaussian_kernel(affinity, h=H, w=W)
                
#         affinity = F.softmax(affinity, dim=1) # b, THW, HW

#         if self.gaussian_kernel != -1:
#             affinity.mul_(gauss_kernel_map)
#             affinity.div_(affinity.sum(dim=1, keepdim=True))
                

#         return affinity

#     def _readout(self, affinity, mv):
#         return torch.bmm(mv, affinity)

#     def match_memory(self, qk):
#         k = self.num_objects
#         _, _, h, w = qk.shape

#         qk = qk.flatten(start_dim=2)
        
#         if self.temp_k is not None:
#             mk = torch.cat([self.mem_k, self.temp_k], 2)
#             mv = torch.cat([self.mem_v, self.temp_v], 2)
#         else:
#             mk = self.mem_k
#             mv = self.mem_v

#         affinity = self._global_matching(mk, qk, h, w)

#         # One affinity for all
#         readout_mem = self._readout(affinity.expand(k,-1,-1), mv)

#         return readout_mem.view(k, self.CV, h, w)

#     def add_memory(self, key, value, is_temp=False):
#         # Temp is for "last frame"
#         # Not always used
#         # But can always be flushed
#         self.temp_k = None
#         self.temp_v = None
#         key = key.flatten(start_dim=2)
#         value = value.flatten(start_dim=2)

#         if self.mem_k is None:
#             # First frame, just shove it in
#             self.mem_k = key
#             self.mem_v = value
#             self.CK = key.shape[1]
#             self.CV = value.shape[1]
#         else:
#             if is_temp:
#                 self.temp_k = key
#                 self.temp_v = value
#             else:
#                 self.mem_k = torch.cat([self.mem_k, key], 2)
#                 self.mem_v = torch.cat([self.mem_v, value], 2)
                
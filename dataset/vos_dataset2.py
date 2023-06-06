import os
from os import path

import random
import math
import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed
import dataset.daugm_video as tf

class VOSDataset2(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl, subset=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < 3:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
#         self.pair_im_lone_transform = transforms.Compose([
#             transforms.ColorJitter(0.01, 0.01, 0.01, 0),
#         ])

#         self.pair_im_dual_transform = transforms.Compose([
#             transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
#         ])

#         self.pair_gt_dual_transform = transforms.Compose([
#             transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
#         ])

#         # These transform are the same for all pairs in the sampled sequence
#         self.all_im_lone_transform = transforms.Compose([
#             transforms.ColorJitter(0.1, 0.03, 0.03, 0),
#             transforms.RandomGrayscale(0.05),
#         ])

        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
            ])
        else:
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
            ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.CROP_SIZE = [384,384]
        self._init_augm()

    def _init_augm(self):

        # general (unguided) affine transformations
        tfs_pre = [tf.CreateMask()]
        self.tf_pre = tf.Compose(tfs_pre)

        # photometric noise
        tfs_affine = []

        # guided affine transformations
        tfs_augm = []

        # 1.
        # general affine transformations
        #
        SMALLEST_RANGE = [384, 450]
        RND_CROP = True
        CROP_SIZE = [384,384]
        RND_HFLIP = True
        GUIDED_HFLIP=True
        RND_ZOOM=True
        RND_ZOOM_RANGE = [.5, 1.]
        

        #tfs_pre.append(tf.MaskScaleSmallest(SMALLEST_RANGE))
        
        #if RND_CROP:
        #    tfs_pre.append(tf.MaskRandCrop(CROP_SIZE, pad_if_needed=True))
        #else:
        #    tfs_pre.append(tf.MaskCenterCrop(CROP_SIZE))

        #if RND_HFLIP:
        #    tfs_pre.append(tf.MaskRandHFlip())
        # 2.
        # Guided affine transformation
        #
        if GUIDED_HFLIP:
            tfs_affine.append(tf.GuidedRandHFlip())

        # this will add affine transformation
        if RND_ZOOM:
            tfs_affine.append(tf.MaskRandScaleCrop(*RND_ZOOM_RANGE))

        self.tf_affine = tf.Compose(tfs_affine)
        self.tf_affine2 = tf.Compose([tf.AffineIdentity()])

        tfs_post = [tf.ToTensorMask(),
                    tf.Normalize(mean=self.MEAN, std=self.STD),
                    tf.ApplyMask(-1)]

        # image to the teacher will have no noise
        self.tf_post = tf.Compose(tfs_post)

    def _get_affine(self, params):

        N = len(params)

        # construct affine operator
        affine = torch.zeros(N, 2, 3)

        aspect_ratio = float(self.CROP_SIZE[0]) / \
                            float(self.CROP_SIZE[1])

        for i, (dy,dx,alpha,scale,flip) in enumerate(params):

            # R inverse
            sin = math.sin(alpha * math.pi / 180.)
            cos = math.cos(alpha * math.pi / 180.)

            # inverse, note how flipping is incorporated
            affine[i,0,0], affine[i,0,1] = flip * cos, sin * aspect_ratio
            affine[i,1,0], affine[i,1,1] = -sin / aspect_ratio, cos

            # T inverse Rinv * t == R^T * t
            affine[i,0,2] = -1. * (cos * dx + sin * dy)
            affine[i,1,2] = -1. * (-sin * dx + cos * dy)

            # T
            affine[i,0,2] /= float(self.CROP_SIZE[1] // 2)
            affine[i,1,2] /= float(self.CROP_SIZE[0] // 2)

            # scaling
            affine[i] *= scale

        return affine

    def _get_affine_inv(self, affine, params):

        aspect_ratio = float(self.CROP_SIZE[0]) / \
                            float(self.CROP_SIZE[1])

        affine_inv = affine.clone()
        affine_inv[:,0,1] = affine[:,1,0] * aspect_ratio**2
        affine_inv[:,1,0] = affine[:,0,1] / aspect_ratio**2
        affine_inv[:,0,2] = -1 * (affine_inv[:,0,0] * affine[:,0,2] + affine_inv[:,0,1] * affine[:,1,2])
        affine_inv[:,1,2] = -1 * (affine_inv[:,1,0] * affine[:,0,2] + affine_inv[:,1,1] * affine[:,1,2])

        # scaling
        affine_inv /= torch.Tensor(params)[:,3].view(-1,1,1)**2

        return affine_inv

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            # Don't want to bias towards beginning/end
            this_max_jump = min(len(frames), self.max_jump)
            start_idx = np.random.randint(len(frames)-this_max_jump+1)
            f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1
            f1_idx = min(f1_idx, len(frames)-this_max_jump, len(frames)-1)

            f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
            f2_idx = min(f2_idx, len(frames)-this_max_jump//2, len(frames)-1)

            frames_idx = [start_idx, f1_idx, f2_idx]
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                #print('jpg_name:', jpg_name)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                #print('png_name:', png_name)
                this_im = self.all_im_dual_transform(this_im)
#                 this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)

                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

#                 pairwise_seed = np.random.randint(2147483647)
#                 reseed(pairwise_seed)
#                 this_im = self.pair_im_dual_transform(this_im)
#                 this_im = self.pair_im_lone_transform(this_im)
#                 reseed(pairwise_seed)
#                 this_gt = self.pair_gt_dual_transform(this_gt)

#                 this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)
            #################################################
            frames_origin, valid_origin = self.tf_pre(images)

            # 1.1 creating two sequences in forward/backward order
            frames1, valid1 = frames_origin[:], valid_origin[:]
            # second copy

            frames2 = [f.copy() for f in frames_origin]
            valid2 = [v.copy() for v in valid_origin]
            # 2. guided affine transforms, tf_affine, tf_affine2: identity get affine param
            frames1, valid1, affine_params1 = self.tf_affine(frames1, valid1)
            frames2, valid2, affine_params2 = self.tf_affine2(frames2, valid2)

            # convert to tensor, zero out the values
            frames1 = self.tf_post(frames1, valid1)
            frames2 = self.tf_post(frames2, valid2)
            # converting the affine transforms
            aff_reg = self._get_affine(affine_params1)
            aff_main = self._get_affine(affine_params2)
            
            aff_reg_inv = self._get_affine_inv(aff_reg, affine_params1)
            aff_reg = aff_main # identity affine2_inv
            aff_main = aff_reg_inv #inverse_aff1
            frames1 = torch.stack(frames1, 0)
            frames2 = torch.stack(frames2, 0)
            assert frames1.shape == frames2.shape

            ##################################################
            #images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]
            
            if len(labels) == 0:
                target_object = -1 # all black if no objects
                has_second_object = False
                trials += 1
            else:
                #labels:[1]/[1,2]
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:
                    labels = labels[labels!=target_object]
                    second_object = np.random.choice(labels)
                break
        
        masks = np.stack(masks, 0)
        tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:]
        if has_second_object:
            sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, 384, 384), dtype=np.int)
        cls_gt[tar_masks[:,0] > 0.5] = 1
        cls_gt[sec_masks[:,0] > 0.5] = 2
        data = {
            #'rgb': images,
            'rgb_view1': frames2, 
            'rgb_view2': frames1,
            'view1_aff': aff_reg,
            'view2_inv_aff': aff_main,
            'gt': tar_masks,
            'cls_gt': cls_gt,
            'sec_gt': sec_masks,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)
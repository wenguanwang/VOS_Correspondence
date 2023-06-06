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
from dataset.tps import random_tps_warp
from dataset.reseed import reseed
import dataset.daugm_video as tf


class StaticTransformDataset(Dataset):
    """
    Generate pseudo VOS data by applying random transforms on static images.
    Single-object only.

    Method 0 - FSS style (class/1.jpg class/1.png)
    Method 1 - Others style (XXX.jpg XXX.png)
    """
    def __init__(self, root, method=0):
        self.root = root
        self.method = method

        if method == 0:
            # Get images
            self.im_list = []
            classes = os.listdir(self.root)
            for c in classes:
                imgs = os.listdir(path.join(root, c))
                jpg_list = [im for im in imgs if 'jpg' in im[-3:].lower()]

                joint_list = [path.join(root, c, im) for im in jpg_list]
                self.im_list.extend(joint_list)

        elif method == 1:
            self.im_list = [path.join(self.root, im) for im in os.listdir(self.root) if '.jpg' in im]

        print('%d images found in %s' % (len(self.im_list), root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0), # No hue change here as that's not realistic
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
            transforms.Resize(384, InterpolationMode.BICUBIC),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=0),
            transforms.Resize(384, InterpolationMode.NEAREST),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0.05),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=0),
            transforms.RandomHorizontalFlip(),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
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
        im = Image.open(self.im_list[idx]).convert('RGB')

        if self.method == 0:
            gt = Image.open(self.im_list[idx][:-3]+'png').convert('L')
        else:
            gt = Image.open(self.im_list[idx].replace('.jpg','.png')).convert('L')

        sequence_seed = np.random.randint(2147483647)

        images = []
        masks = []
        for _ in range(3):
            reseed(sequence_seed)
            this_im = self.all_im_dual_transform(im)
            this_im = self.all_im_lone_transform(this_im)
            reseed(sequence_seed)
            this_gt = self.all_gt_dual_transform(gt)

            pairwise_seed = np.random.randint(2147483647)
            reseed(pairwise_seed)
            this_im = self.pair_im_dual_transform(this_im)
            this_im = self.pair_im_lone_transform(this_im)
            reseed(pairwise_seed)
            this_gt = self.pair_gt_dual_transform(this_gt)

            # Use TPS only some of the times
            # Not because TPS is bad -- just that it is too slow and I need to speed up data loading
            if np.random.rand() < 0.33:
                this_im, this_gt = random_tps_warp(this_im, this_gt, scale=0.02)

            #this_im = self.final_im_transform(this_im)
            this_gt = self.final_gt_transform(this_gt)

            images.append(this_im)
            masks.append(this_gt)
        ###########################################################
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

        ###########################################################
        masks = torch.stack(masks, 0)

        info = {}
        info['name'] = self.im_list[idx]

        cls_gt = np.zeros((3, 384, 384), dtype=np.int)
        cls_gt[masks[:,0] > 0.5] = 1

        data = {
            #'rgb': images,
            'rgb_view1': frames2, 
            'rgb_view2': frames1,
            'view1_aff': aff_reg,
            'view2_inv_aff': aff_main,
            'gt': masks,
            'cls_gt': cls_gt,
            'info': info,
        }

        return data


    def __len__(self):
        return len(self.im_list)
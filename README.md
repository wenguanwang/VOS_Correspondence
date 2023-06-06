# Boosting Video Object Segmentation via Space-time Correspondence Learning
[[`arXiv`](https://arxiv.org/abs/2304.06211)] [[`BibTeX`](#CitingSemiVOS)]

## Updates
* Code, ckpt and training log have been released [here](https://github.com/wenguanwang/VOS_Correspondence/releases/tag/CKPT). Readme will be updated. 
* Our new project [Segment and Track Anything (SAM-Track)](https://github.com/z-x-yang/Segment-and-Track-Anything) which focuses on the segmentation and tracking of any objects in videos, utilizing both automatic and interactive methods has been released.
* This repo will release an official *Pytorch** implementation.

## Abstract
Current top-leading solutions for video object segmentation (VOS) typically follow a matching-based regime: for each query frame, the segmentation mask is inferred according to its correspondence to previously processed and the first annotated frames. They simply exploit the supervisory signals from the groundtruth masks for learning mask prediction only, without posing any constraint on the space-time correspondence matching, which, however, is the fundamental building block of such regime. To alleviate this crucial yet commonly ignored issue, we devise a correspondence-aware training framework, which boosts matching-based VOS solutions by explicitly encouraging robust correspondence matching during network learning. Through comprehensively exploring the intrinsic coherence in videos on pixel and object levels, our algorithm reinforces the standard, fully supervised training of mask segmentation with label-free, contrastive correspondence learning. Without neither requiring extra annotation cost during training, nor causing speed delay during deployment, nor incurring architectural modification, our algorithm provides solid performance gains on four widely used benchmarks, i.e., DAVIS2016&2017, and YouTube-VOS2018&2019, on the top of famous matching-based VOS solutions.


## <a name="CitingSemiVOS"></a>Citing Correspondence VOS
```BibTeX
@inproceedings{zhang2023boosting,
	title={Boosting Video Object Segmentation via Space-time Correspondence Learning},
        author={Zhang, Yurong and Li, Liulei and Wang, Wenguan and Xie, Rong and Song, Li and Zhang, Wenjun},
	booktitle=CVPR,
	year={2023}
}
```

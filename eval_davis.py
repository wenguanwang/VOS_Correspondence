# import os
# from os import path
# import time
# from argparse import ArgumentParser

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import numpy as np
# from PIL import Image
# from model.eval_network import STCN
# from dataset.davis_test_dataset import DAVISTestDataset
# from util.tensor_util import unpad
# from inference_core import InferenceCore

# from progressbar import progressbar
# import pandas as pd
# from davis2017.evaluation import DAVISEvaluation
# import sys

# """
# Arguments loading
# """
# time_start = time.time()
# #../DAVIS/2017/trainvals
# parser = ArgumentParser()
# parser.add_argument('--model', default='stcn.pth')
# parser.add_argument('--davis_path', default='data/DAVIS/2017')
# parser.add_argument('--csv_path', default='output')
# parser.add_argument('--output', default = 'output')
# parser.add_argument('--split', help='val/testdev', default='val')
# parser.add_argument('--top', type=int, default=20)
# parser.add_argument('--amp', action='store_true')
# parser.add_argument('--mem_every', default=5, type=int)
# parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')

# parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
# parser.add_argument('--task', type=str, help='Task to evaluate the results', default='semi-supervised',
#                     choices=['semi-supervised', 'unsupervised'])

# args = parser.parse_args()
# csv_name_global = 'global_results-15000.csv'
# csv_name_per_sequence = 'per-sequence_results-15000.csv'
# if not os.path.exists(args.csv_path):
#     os.makedirs(args.csv_path)
# csv_name_global_path = os.path.join(args.csv_path, csv_name_global)
# csv_name_per_sequence_path = os.path.join(args.csv_path, csv_name_per_sequence)

# davis_path = args.davis_path
# davis_metric_path = davis_path + '/trainval'
# out_path = args.output

# # Simple setup
# os.makedirs(out_path, exist_ok=True)
# palette = Image.open(path.expanduser(davis_path + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()

# torch.autograd.set_grad_enabled(False)

# # Setup Dataset
# if args.split == 'val':
#     test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/val.txt')
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
# elif args.split == 'testdev':
#     test_dataset = DAVISTestDataset(davis_path+'/test-dev', imset='2017/test-dev.txt')
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
# else:
#     raise NotImplementedError

# # Load our checkpoint
# top_k = args.top
# prop_model = STCN().cuda().eval()

# # Performs input mapping such that stage 0 model can be loaded
# prop_saved = torch.load(args.model)
# for k in list(prop_saved.keys()):
#     if k == 'value_encoder.conv1.weight':
#         if prop_saved[k].shape[1] == 4:
#             pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
#             prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
# prop_model.load_state_dict(prop_saved, strict=False)

# total_process_time = 0
# total_frames = 0

# # Start eval
# for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

#     with torch.cuda.amp.autocast(enabled=args.amp):
#         rgb = data['rgb'].cuda()
#         msk = data['gt'][0].cuda()
#         info = data['info']
#         name = info['name'][0]
#         k = len(info['labels'][0])
#         size = info['size_480p']

#         torch.cuda.synchronize()
#         process_begin = time.time()

#         processor = InferenceCore(prop_model, rgb, k, top_k=top_k, 
#                         mem_every=args.mem_every, include_last=args.include_last)
#         processor.interact(msk[:,0], 0, rgb.shape[1])

#         # Do unpad -> upsample to original size 
#         out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
#         for ti in range(processor.t):
#             prob = unpad(processor.prob[:,ti], processor.pad)
#             prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
#             out_masks[ti] = torch.argmax(prob, dim=0)
        
#         out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

#         torch.cuda.synchronize()
#         total_process_time += time.time() - process_begin
#         total_frames += out_masks.shape[0]

#         # Save the results
#         this_out_path = path.join(out_path, name)
#         os.makedirs(this_out_path, exist_ok=True)
#         for f in range(out_masks.shape[0]):
#             img_E = Image.fromarray(out_masks[f])
#             img_E.putpalette(palette)
#             img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

#         del rgb
#         del msk
#         del processor

# print('Total processing time: ', total_process_time)
# print('Total processed frames: ', total_frames)
# print('FPS: ', total_frames / total_process_time)

# if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
#     print('Using precomputed results...')
#     table_g = pd.read_csv(csv_name_global_path)
#     table_seq = pd.read_csv(csv_name_per_sequence_path)
# else:
#     print(f'Evaluating sequences for the {args.task} task...')
#     # Create dataset and evaluate
#     dataset_eval = DAVISEvaluation(davis_root=davis_metric_path, task=args.task, gt_set=args.set)
#     metrics_res = dataset_eval.evaluate(out_path)
#     J, F = metrics_res['J'], metrics_res['F']

#     # Generate dataframe for the general results
#     g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
#     final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
#     g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
#                       np.mean(F["D"])])
#     g_res = np.reshape(g_res, [1, len(g_res)])
#     table_g = pd.DataFrame(data=g_res, columns=g_measures)
#     with open(csv_name_global_path, 'w') as f:
#         table_g.to_csv(f, index=False, float_format="%.3f")
#     print(f'Global results saved in {csv_name_global_path}')

#     # Generate a dataframe for the per sequence results
#     seq_names = list(J['M_per_object'].keys())
#     seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
#     J_per_object = [J['M_per_object'][x] for x in seq_names]
#     F_per_object = [F['M_per_object'][x] for x in seq_names]
#     table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
#     with open(csv_name_per_sequence_path, 'w') as f:
#         table_seq.to_csv(f, index=False, float_format="%.3f")
#     print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

# # Print the results
# sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
# print(table_g.to_string(index=False))
# sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
# print(table_seq.to_string(index=False))
# total_time = time.time() - time_start
# sys.stdout.write('\nTotal time:' + str(total_time))

import os
from os import path
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from dataset.davis_test_dataset import DAVISTestDataset
from util.tensor_util import unpad
from inference_core import InferenceCore

from progressbar import progressbar


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--davis_path', default='DAVIS')
parser.add_argument('--output', default='output')
parser.add_argument('--split', help='val/testdev', default='testdev')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

davis_path = args.davis_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(davis_path + '/Annotations/480p/aerobatics/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Setup Dataset
if args.split == 'val':
    test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/val.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
elif args.split == 'testdev':
    test_dataset = DAVISTestDataset(davis_path, imset='2017/test-dev.txt')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
else:
    raise NotImplementedError

# Load our checkpoint
top_k = args.top
prop_model = STCN().cuda().eval()

# Performs input mapping such that stage 0 model can be loaded
prop_saved = torch.load(args.model)
for k in list(prop_saved.keys()):
    if k == 'value_encoder.conv1.weight':
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
prop_model.load_state_dict(prop_saved, strict=False)

total_process_time = 0
total_frames = 0

# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data['rgb'].cuda()
        msk = data['gt'][0].cuda()
        info = data['info']
        name = info['name'][0]
        k = len(info['labels'][0])
        size = info['size_480p']

        torch.cuda.synchronize()
        process_begin = time.time()

        processor = InferenceCore(prop_model, rgb, k, top_k=top_k, 
                        mem_every=args.mem_every, include_last=args.include_last)
        processor.interact(msk[:,0], 0, rgb.shape[1])

        # Do unpad -> upsample to original size 
        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
        for ti in range(processor.t):
            prob = unpad(processor.prob[:,ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)
        
        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        torch.cuda.synchronize()
        total_process_time += time.time() - process_begin
        total_frames += out_masks.shape[0]

        # Save the results
        this_out_path = path.join(out_path, name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(out_masks.shape[0]):
            img_E = Image.fromarray(out_masks[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

        del rgb
        del msk
        del processor

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)
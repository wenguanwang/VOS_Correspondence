import datetime
import os
from os import path
import math
import sys
import random
import numpy as np
import torch
import torch.nn.functional as Functorch
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed

from model.model import STCNModel
from dataset.static_dataset import StaticTransformDataset
from dataset.vos_dataset import VOSDataset

from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters
from util.load_subset import load_sub_davis, load_sub_yv
import time

from model.eval_network import STCN
from dataset.davis_test_dataset import DAVISTestDataset
from util.tensor_util import unpad
from inference_core import InferenceCore

from progressbar import progressbar
import pandas as pd
from davis2017.evaluation import DAVISEvaluation

"""
Initial setup
"""
# Init distributed environment
distributed.init_process_group(backend="nccl")
# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

print('CUDA Device count: ', torch.cuda.device_count())

# Parse command line arguments
para = HyperParameters()
para.parse()

if para['benchmark']:
    torch.backends.cudnn.benchmark = True

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print('I am rank %d in this world of size %d!' % (local_rank, world_size))

"""
Model related
"""
if local_rank == 0:
    # Logging
    if para['id'].lower() != 'null':
        print('I will take the role of logging!')
        long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), para['id'])
    else:
        long_id = None
    logger = TensorboardLogger(para['id'], long_id)
    logger.log_string('hyperpara', str(para))

    # Construct the rank 0 model
    model = STCNModel(para, logger=logger, 
                    save_path=path.join(para['ckpt_output'], long_id, long_id) if long_id is not None else None, 
                    local_rank=local_rank, world_size=world_size).train()
else:
    # Construct model for other ranks
    model = STCNModel(para, local_rank=local_rank, world_size=world_size).train()

# Load pertrained model if needed
if para['load_model'] is not None:
    total_iter = model.load_model(para['load_model'])
    print('Previously trained model loaded!')
else:
    total_iter = 0

if para['load_network'] is not None:
    model.load_network(para['load_network'])
    print('Previously trained network loaded!')

"""
Dataloader related
"""
# To re-seed the randomness everytime we start a worker
def worker_init_fn(worker_id): 
    return np.random.seed(torch.initial_seed()%(2**31) + worker_id + local_rank*100)

def construct_loader(dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
    train_loader = DataLoader(dataset, para['batch_size'], sampler=train_sampler, num_workers=para['num_workers'],
                            worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)
    return train_sampler, train_loader

def renew_vos_loader(max_skip):
    //5 because we only have annotation for every five frames
    yv_dataset = VOSDataset(path.join(yv_root, 'JPEGImages'), 
                        path.join(yv_root, 'Annotations'), max_skip//5, is_bl=False, subset=load_sub_yv())
    #################################################################################################################
    davis_dataset = VOSDataset(path.join(davis_root, 'JPEGImages', '480p'), 
                        path.join(davis_root, 'Annotations', '480p'), max_skip, is_bl=False, subset=load_sub_davis())
    train_dataset = ConcatDataset([davis_dataset] + [yv_dataset])

    print('Concat dataset size: ', len(train_dataset))
    print('Renewed with skip: ', max_skip)

    return construct_loader(train_dataset)


def renew_bl_loader(max_skip):
    train_dataset = VOSDataset(path.join(bl_root, 'JPEGImages'), 
                        path.join(bl_root, 'Annotations'), max_skip, is_bl=True)

    print('Blender dataset size: ', len(train_dataset))
    print('Renewed with skip: ', max_skip)

    return construct_loader(train_dataset)

def eval_stcn_net(model2, iteration, para):
    time_start = time.time()
    set_para = para['set']
    csv_name_global = f'global_results-{set_para}-{iteration}.csv'
    csv_name_per_sequence = f'per-sequence_results-{set_para}-{iteration}.csv'
    if not os.path.exists(para['csv_path']):
        os.makedirs(para['csv_path'])
    csv_name_global_path = os.path.join(para['csv_path'], csv_name_global)
    csv_name_per_sequence_path = os.path.join(para['csv_path'], csv_name_per_sequence)

    davis_path = para['davis_path']
    davis_metric_path = davis_path + '/trainval'
    out_path = para['output']

    # Simple setup
    os.makedirs(out_path, exist_ok=True)
    palette = Image.open(path.expanduser(davis_path + '/trainval/Annotations/480p/blackswan/00000.png')).getpalette()

    torch.autograd.set_grad_enabled(False)

    # Setup Dataset
    if para['split'] == 'val':
        test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/val.txt')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    elif para['split'] == 'testdev':
        test_dataset = DAVISTestDataset(davis_path+'/test-dev', imset='2017/test-dev.txt')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    else:
        raise NotImplementedError

    # Load our checkpoint
    top_k = para['top']

    # Performs input mapping such that stage 0 model can be loaded
    #prop_saved = torch.load(para['model'])
    #################################################################################################################
    prop_saved = model.STCN.module.state_dict().copy()
#     prop_saved = model.ema_STCN.ema_model.module.state_dict().copy()
    for k in list(prop_saved.keys()):
        if k == 'value_encoder.conv1.weight':
            if prop_saved[k].shape[1] == 4:
                pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
                prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
    prop_model = STCN().cuda().eval()
    prop_model.load_state_dict(prop_saved,strict=False)
    torch.cuda.synchronize()

    total_process_time = 0
    total_frames = 0

    # Start eval
    for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

        with torch.cuda.amp.autocast(False):
            rgb = data['rgb'].cuda()
            msk = data['gt'][0].cuda()
            info = data['info']
            name = info['name'][0]
            k = len(info['labels'][0])
            size = info['size_480p']

            torch.cuda.synchronize()
            process_begin = time.time()

            processor = InferenceCore(prop_model, rgb, k, top_k=top_k, 
                            mem_every=para['mem_every'], include_last=para['include_last'])
            processor.interact(msk[:,0], 0, rgb.shape[1])

            # Do unpad -> upsample to original size 
            out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
            for ti in range(processor.t):
                prob = unpad(processor.prob[:,ti], processor.pad)
                prob = Functorch.interpolate(prob, size, mode='bilinear', align_corners=False)
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

    task_para = para['task']
    print(f'Evaluating sequences for the {task_para} task...')
    # Create dataset and evaluate
    dataset_eval = DAVISEvaluation(davis_root=davis_metric_path, task=para['task'], gt_set=para['set'])
    metrics_res = dataset_eval.evaluate(out_path)
    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                    np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

    # Print the results
    sys.stdout.write(f"--------------------------- Global results for {para['set']} ---------------------------\n")
    print(table_g.to_string(index=False))
    sys.stdout.write(f"\n---------- Per sequence results for {para['set']} ----------\n")
    print(table_seq.to_string(index=False))
    total_time = time.time() - time_start
    sys.stdout.write('\nTotal time:' + str(total_time))

"""
Dataset related
"""

"""
These define the training schedule of the distance between frames
We will switch to skip_values[i] once we pass the percentage specified by increase_skip_fraction[i]
Not effective for stage 0 training
"""
#################################################################################################################
# skip_values = [10, 15, 20, 25, 25]
skip_values = [10, 15, 5]

if para['stage'] == 0:
    static_root = path.expanduser(para['static_root'])
    fss_dataset = StaticTransformDataset(path.join(static_root, 'fss'), method=0)
    duts_tr_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TR'), method=1)
    duts_te_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TE'), method=1)
    ecssd_dataset = StaticTransformDataset(path.join(static_root, 'ecssd'), method=1)

    big_dataset = StaticTransformDataset(path.join(static_root, 'BIG_small'), method=1)
    hrsod_dataset = StaticTransformDataset(path.join(static_root, 'HRSOD_small'), method=1)
#     coco_dataset = StaticTransformDataset(path.join(static_root, 'COCO-2017'), method=2)
#     voc_dataset = StaticTransformDataset(path.join(static_root, 'PASCAL-VOC'), method=2)

    # BIG and HRSOD have higher quality, use them more
    train_dataset = ConcatDataset([fss_dataset, duts_tr_dataset, duts_te_dataset, ecssd_dataset]
             + [big_dataset, hrsod_dataset]*5)
    train_sampler, train_loader = construct_loader(train_dataset)

#     coco_dataset = StaticTransformDataset(path.join(static_root, 'COCO-2017'), method=2)
#     train_dataset = coco_dataset
#     train_sampler, train_loader = construct_loader(train_dataset)

    print('Static dataset size: ', len(train_dataset))
elif para['stage'] == 1:
    increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.8, 1.0]
    bl_root = path.join(path.expanduser(para['bl_root']))

    train_sampler, train_loader = renew_bl_loader(5)
    renew_loader = renew_bl_loader
else:
    # stage 2 or 3
    #################################################################################################################
#     increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.9, 1.0]
    increase_skip_fraction = [0.1, 0.3, 0.8, 1.0]
    # VOS dataset, 480p is used for both datasets
    yv_root = path.join(path.expanduser(para['yv_root']), 'train_480p')
    #../YouTube/train_480p/JPEGImages
    davis_root = path.join(path.expanduser(para['davis_root']), '2017', 'trainval')
    #../DAVIS/2017/trainval/JPEGImages/480p
    train_sampler, train_loader = renew_vos_loader(5)
    renew_loader = renew_vos_loader

"""
Determine current/max epoch
"""
total_epoch = math.ceil(para['iterations']/len(train_loader))
current_epoch = total_iter // len(train_loader)
print('Number of training epochs (the last epoch might not complete): ', total_epoch)
if para['stage'] != 0:
    increase_skip_epoch = [round(total_epoch*f) for f in increase_skip_fraction]
    # Skip will only change after an epoch, not in the middle
    print('The skip value will increase approximately at the following epochs: ', increase_skip_epoch[:-1])

"""
Starts training
"""
# Need this to select random bases in different workers
np.random.seed(np.random.randint(2**30-1) + local_rank*100)
try:
    for e in range(current_epoch, total_epoch): 
        print('Epoch %d/%d' % (e, total_epoch))
        if para['stage']!=0 and e!=total_epoch and e>=increase_skip_epoch[0]:
            while e >= increase_skip_epoch[0]:
                cur_skip = skip_values[0]
                skip_values = skip_values[1:]
                increase_skip_epoch = increase_skip_epoch[1:]
            print('Increasing skip to: ', cur_skip)
            train_sampler, train_loader = renew_loader(cur_skip)

        # Crucial for randomness! 
        train_sampler.set_epoch(e)

        # Train loop
        model.train()
        for data in train_loader:
            model.do_pass(data, total_iter)
            if total_iter % 5000== 0 and total_iter != 0 and local_rank==0:
                if total_iter>0:
                    model.save(total_iter)
                    eval_stcn_net(model, total_iter, para)

            total_iter += 1
            if total_iter >= para['iterations']:
                break
        
finally:
    if not para['debug'] and model.logger is not None and total_iter>5000:
        model.save(total_iter)
        eval_stcn_net(model, total_iter, para)
    # Clean up
    distributed.destroy_process_group()

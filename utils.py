# Copyright (c) Meta Platforms, Inc. All rights reserved.
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import os
import subprocess
import time
from datetime import datetime


def freeze_params_by_layers(model, num_enc_layers, num_frozen_layers=0):
    additional_frozen_params = ['model.shared.weight', 'model.encoder.embed_positions.weight',
                                'model.decoder.embed_positions.weight']
    for name, par in model.named_parameters():
        if name in additional_frozen_params:
            par.requires_grad = False
        elif not name.startswith('model'):
            print(f'{name} will update!')
        else:
            try:
                layer_idx = int(name.split('.')[3])
            except ValueError:
                par.requires_grad = False
                continue
            is_decoder = 'decoder' in name
            if is_decoder:
                layer_idx += num_enc_layers
            if layer_idx < num_frozen_layers:
                par.requires_grad = False


def freeze_params(model, except_para_l=()):
    for name, par in model.named_parameters():
        skip = False
        for except_para in except_para_l:
            if except_para in name:
                # print(f'{name} |skipped when alterning requires_grad')
                skip = True
                break
        if skip:
            continue
        par.requires_grad = False


def unfreeze_params(model, except_para=None):
    for name, par in model.named_parameters():
        if except_para is not None and except_para in name:
            par.requires_grad = False
        else:
            par.requires_grad = True


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_info = [eval(x) for x in result.strip().split('\n')]
    gpu_info = dict(zip(range(len(gpu_info)), gpu_info))
    sorted_gpu_info = sorted(gpu_info.items(), key=lambda kv: kv[1][0], reverse=True)
    sorted_gpu_info = sorted(sorted_gpu_info, key=lambda kv: kv[1][1])
    return sorted_gpu_info


def choose_gpu(n_gpus=1, min_gpu_memory=6000, retry=False, sleep_time=30):
    start_time = time.time()
    sorted_gpu_info = get_gpu_memory_map()
    try:
        gpustat = subprocess.check_output(
            [
                'gpustat'
            ], encoding='utf-8')
        print(gpustat)
    except Exception as e:
        print(e)
    print(f'gpu_id, (mem_left, util): {sorted_gpu_info}')
    while True:
        gpus = []
        for gpu_id, (mem_left, util) in sorted_gpu_info:
            if mem_left >= min_gpu_memory:
                gpus.append(gpu_id)
                print('use gpu:{} with {} MB left, util {}%'.format(gpu_id, mem_left, util))
            if len(gpus) == n_gpus:
                # print('max num of gpus reached.')
                break
        if len(gpus) == 0:
            if retry:
                print(f'[{datetime.now().strftime("%H:%M:%S")}'
                      f' waited {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}]'
                      f' no gpu has memory >= {min_gpu_memory} MB, sleep {sleep_time}s...', end='\r')
                time.sleep(sleep_time)
            else:
                print(f'no gpu has memory >= {min_gpu_memory} MB, exiting...')
                exit()
        else:
            break
        sorted_gpu_info = get_gpu_memory_map()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    visible_gpus = ','.join([str(gpu_id) for gpu_id in gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

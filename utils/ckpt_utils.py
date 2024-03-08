import os
import argparse
from math import ceil
import requests
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn

available_simclr_models = ['r50_1x_sk0', 'r50_1x_sk1', 'r50_2x_sk0', 'r50_2x_sk1',
                           'r101_1x_sk0', 'r101_1x_sk1', 'r101_2x_sk0', 'r101_2x_sk1',
                           'r152_1x_sk0', 'r152_1x_sk1', 'r152_2x_sk0', 'r152_2x_sk1', 'r152_3x_sk1']
simclr_base_url = 'https://storage.googleapis.com/simclr-checkpoints/simclrv2/{category}/{model}/'
files = ['checkpoint', 'graph.pbtxt', 'model.ckpt-{category}.data-00000-of-00001',
         'model.ckpt-{category}.index', 'model.ckpt-{category}.meta']
simclr_categories = {'finetuned_100pct': 37535, 'finetuned_10pct': 3754,
                     'finetuned_1pct': 751, 'pretrained': 250228, 'supervised': 28151}
chunk_size = 1024 * 8

def download(url, destination):
    if os.path.exists(destination):
        return
    response = requests.get(url, stream=True)
    save_response_content(response, destination)


def save_response_content(response, destination):
    if 'Content-length' in response.headers:
        total = int(ceil(int(response.headers['Content-length']) / chunk_size))
    else:
        total = None
    with open(destination, 'wb') as f:
        for data in tqdm(response.iter_content(chunk_size=chunk_size), leave=False, total=total):
            f.write(data)


def extract_simclr_weights(model, weight_path, model_save_path):
    """Convert tf weights to pytorch weights
    """
    prefix = 'ema_model/base_model/'
    head_prefix = 'ema_model/head_contrastive/'
    conv_prefix = 'ema_model/base_model/conv2d{key}/kernel'
    batchnorm_prefix = 'ema_model/base_model/batch_normalization{key}'
    ckpt_reader = tf.train.load_checkpoint(weight_path)
    resnet_backbone = {}
    projection_head = {}
    fc = {}
    for v in tf.train.list_variables(weight_path):
        if v[0].startswith(prefix) and not v[0].endswith('/Momentum'):
            resnet_backbone[v[0]] = ckpt_reader.get_tensor(v[0])
        elif v[0] in {'head_supervised/linear_layer/dense/bias', 'head_supervised/linear_layer/dense/kernel'}:
            fc[v[0]] = ckpt_reader.get_tensor(v[0])
        elif v[0].startswith(head_prefix) and not v[0].endswith('/Momentum'):
            projection_head[v[0]] =  ckpt_reader.get_tensor(v[0])

    # create index mapping from torch model to tf model
    n_bottleneck = [0, 3, 4, 6, 3]
    n_convs = [i * 3 + 1 for i in n_bottleneck]
    layer_conv_start_pos = np.cumsum(n_convs)
    indexes = np.arange(layer_conv_start_pos[-1])
    for pos in layer_conv_start_pos[:-1]:
        temp = indexes[pos]
        indexes[pos:pos+3] = indexes[pos+1:pos+4]
        indexes[pos+3] = temp

    conv_op = []
    bn_op = []
    for m in model.backbone.modules():
        if isinstance(m, nn.Conv2d):
            conv_op.append(m)
        elif isinstance(m, nn.BatchNorm2d):
            bn_op.append(m) 

    
    for i, index in enumerate(indexes):
        key = '' if index == 0 else f'_{index}'
        conv_key = conv_prefix.format(key=key)
        batchnorm_gamma_key = batchnorm_prefix.format(key=key) + '/gamma'
        batchnorm_beta_key = batchnorm_prefix.format(key=key) + '/beta'
        batchnorm_moving_mean_key = batchnorm_prefix.format(key=key) + '/moving_mean'
        batchnorm_moving_variance_key = batchnorm_prefix.format(key=key) + '/moving_variance'

        # copy conv weights
        torch_conv = conv_op[i]
        tf_conv = torch.from_numpy(resnet_backbone[conv_key]).permute(3, 2, 0, 1)
        assert tf_conv.shape == torch_conv.weight.shape, f'size mismatch {tf_conv.shape} <> {torch_conv.weight.shape}'
        torch_conv.weight.data = tf_conv

        # copy batchnorm weights
        torch_batchnorm = bn_op[i]
        tf_gamma = torch.from_numpy(resnet_backbone[batchnorm_gamma_key])
        assert torch_batchnorm.weight.shape == tf_gamma.shape, f'size mismatch {tf_gamma.shape} <> {torch_batchnorm.weight.shape}'
        torch_batchnorm.weight.data = tf_gamma
        torch_batchnorm.bias.data = torch.from_numpy(resnet_backbone[batchnorm_beta_key])
        torch_batchnorm.running_mean = torch.from_numpy(resnet_backbone[batchnorm_moving_mean_key])
        torch_batchnorm.running_var = torch.from_numpy(resnet_backbone[batchnorm_moving_variance_key])

    # copy fc weights
    w = torch.from_numpy(fc['head_supervised/linear_layer/dense/kernel']).t()
    assert model.fc.weight.shape == w.shape
    model.fc.weight.data = w
    b = torch.from_numpy(fc['head_supervised/linear_layer/dense/bias'])
    assert model.fc.bias.shape == b.shape
    model.fc.bias.data = b

    # copy contrastive head weights
    linear_op = []
    bn_op = []
    for m in model.projection_head.modules():
        if isinstance(m, nn.Linear):
            linear_op.append(m)
        elif isinstance(m, nn.BatchNorm1d):
            bn_op.append(m)
    for i, (l, m) in enumerate(zip(linear_op, bn_op)):
        l.weight.data = torch.from_numpy(projection_head[f'{head_prefix}nl_{i}/dense/kernel']).t()
        common_prefix = f'{head_prefix}nl_{i}/batch_normalization/'
        m.weight.data = torch.from_numpy(projection_head[f'{common_prefix}gamma'])
        if i != 2:
            m.bias.data = torch.from_numpy(projection_head[f'{common_prefix}beta'])
        m.running_mean = torch.from_numpy(projection_head[f'{common_prefix}moving_mean'])
        m.running_var = torch.from_numpy(projection_head[f'{common_prefix}moving_variance'])

    torch.save({'state_dict': model.state_dict()}, model_save_path)

def extract_classifier_weights(model, weight_path, model_save_path):
    """Convert tf weights to pytorch weights
    """
    prefix = 'base_model/'
    conv_prefix = 'base_model/conv2d{key}/kernel'
    batchnorm_prefix = 'base_model/batch_normalization{key}'
    ckpt_reader = tf.train.load_checkpoint(weight_path)
    resnet_backbone = {}
    projection_head = {}
    fc = {}
    for v in tf.train.list_variables(weight_path):
        if v[0].startswith(prefix) and not v[0].endswith('/Momentum'):
            resnet_backbone[v[0]] = ckpt_reader.get_tensor(v[0])
        elif v[0] in {'head_supervised/linear_layer/dense/bias', 'head_supervised/linear_layer/dense/kernel'}:
            fc[v[0]] = ckpt_reader.get_tensor(v[0])

    # create index mapping from torch model to tf model
    n_bottleneck = [0, 3, 4, 6, 3]
    n_convs = [i * 3 + 1 for i in n_bottleneck]
    layer_conv_start_pos = np.cumsum(n_convs)
    indexes = np.arange(layer_conv_start_pos[-1])
    for pos in layer_conv_start_pos[:-1]:
        temp = indexes[pos]
        indexes[pos:pos+3] = indexes[pos+1:pos+4]
        indexes[pos+3] = temp

    conv_op = []
    bn_op = []
    for m in model.backbone.modules():
        if isinstance(m, nn.Conv2d):
            conv_op.append(m)
        elif isinstance(m, nn.BatchNorm2d):
            bn_op.append(m) 
   
    for i, index in enumerate(indexes):
        key = '' if index == 0 else f'_{index}'
        conv_key = conv_prefix.format(key=key)
        batchnorm_gamma_key = batchnorm_prefix.format(key=key) + '/gamma'
        batchnorm_beta_key = batchnorm_prefix.format(key=key) + '/beta'
        batchnorm_moving_mean_key = batchnorm_prefix.format(key=key) + '/moving_mean'
        batchnorm_moving_variance_key = batchnorm_prefix.format(key=key) + '/moving_variance'

        # copy conv weights
        torch_conv = conv_op[i]
        tf_conv = torch.from_numpy(resnet_backbone[conv_key]).permute(3, 2, 0, 1)
        assert tf_conv.shape == torch_conv.weight.shape, f'size mismatch {tf_conv.shape} <> {torch_conv.weight.shape}'
        torch_conv.weight.data = tf_conv

        # copy batchnorm weights
        torch_batchnorm = bn_op[i]
        tf_gamma = torch.from_numpy(resnet_backbone[batchnorm_gamma_key])
        assert torch_batchnorm.weight.shape == tf_gamma.shape, f'size mismatch {tf_gamma.shape} <> {torch_batchnorm.weight.shape}'
        torch_batchnorm.weight.data = tf_gamma
        torch_batchnorm.bias.data = torch.from_numpy(resnet_backbone[batchnorm_beta_key])
        torch_batchnorm.running_mean = torch.from_numpy(resnet_backbone[batchnorm_moving_mean_key])
        torch_batchnorm.running_var = torch.from_numpy(resnet_backbone[batchnorm_moving_variance_key])

    # copy fc weights
    w = torch.from_numpy(fc['head_supervised/linear_layer/dense/kernel']).t()
    assert model.fc.weight.shape == w.shape
    model.fc.weight.data = w
    b = torch.from_numpy(fc['head_supervised/linear_layer/dense/bias'])
    assert model.fc.bias.shape == b.shape
    model.fc.bias.data = b

    torch.save({'state_dict': model.state_dict()}, model_save_path)

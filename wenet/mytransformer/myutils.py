#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Xiong Wang (Northwestern Polytechnical University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import yaml
import json
import torch
import argparse
import importlib

from distutils.util import strtobool as dist_strtobool

IGNORE_ID = -1

def assign_args_from_yaml(args, yaml_path, prefix_key=None):
    with open(yaml_path) as f:
        ydict = yaml.load(f, Loader=yaml.FullLoader)
    if prefix_key is not None:
        ydict = ydict[prefix_key]
    for k, v in ydict.items():
        k_args = k.replace('-', '_') 
        if hasattr(args, k_args):
            setattr(args, k_args, ydict[k])
    return args

def get_model_conf(model_path):
    model_conf = os.path.dirname(model_path) + '/model.json'
    with open(model_conf, "rb") as f:
        print('reading a config file from ' + model_conf)
        confs = json.load(f)
    # for asr, tts, mt
    idim, odim, args = confs
    return argparse.Namespace(**args)

def strtobool(x):
    return bool(dist_strtobool(x))

def dynamic_import(import_path, alias=dict()):
    """dynamic import module and class

    :param str import_path: syntax 'module_name:class_name'
        e.g., 'espnet.transform.add_deltas:AddDeltas'
    :param dict alias: shortcut for registered class
    :return: imported class
    """
    if import_path not in alias and ':' not in import_path:
        raise ValueError(
            'import_path should be one of {} or '
            'include ":", e.g. "espnet.transform.add_deltas:AddDeltas" : '
            '{}'.format(set(alias), import_path))
    if ':' not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(':')
    m = importlib.import_module(module_name)
    return getattr(m, objname)

def set_deterministic_pytorch(args):
    # seed setting
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


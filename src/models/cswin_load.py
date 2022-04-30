import os
import torch
import torch.nn as nn
from collections import OrderedDict

def load_checkpoint(model, use_ema=True):
    fine_22k =  False
    local_rank = 0
    checkpoint_path = '/home/student/min_JWLiao_2022/sam/Orchid_Classification/src/models/pretrain_weight/cswin_base_384.pth'

    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        model_key = 'model'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                if fine_22k and 'head' in k:
                    continue
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif model_key and model_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[model_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                if fine_22k and 'head' in k:
                    continue
                new_state_dict[name] = v
            state_dict = new_state_dict

        else:
            state_dict = checkpoint
        if local_rank == 0:
            print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
    else:
        if local_rank == 0:
            print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
    model_dict = model.state_dict()
    pretrained_dict = state_dict
    loaded_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    load_dic = [k for k, v in pretrained_dict.items() if k in model_dict]
    miss_dic = [k for k, v in pretrained_dict.items() if not (k in model_dict)]
    unexpect_dic = [k for k, v in model_dict.items() if not (k in pretrained_dict)]
    if local_rank == 0:
        print ("Miss Keys:", miss_dic)
        print ("Ubexpected Keys:", unexpect_dic)
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict, strict=True)
    model.head = nn.Linear(model.head.in_features, 219)

    return model

import sys
import os

# Stable Diffusion v1.5
input_path = './models/v1-5-pruned.ckpt'

# Output DiAD model
output_path = './models/diad.ckpt'

# Finetuned autoencoders
path_input = './models/mvtecad_fs.ckpt'

import torch
from share import *
from sgn.model import create_model, load_state_dict


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./models/diad.yaml')
input_state_dict = load_state_dict(path_input)

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    is_first_stage, name2 = get_node_name(k, 'first_stage_model')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if copy_k in pretrained_weights and not is_first_stage:
        target_dict[k] = pretrained_weights[copy_k].clone()
    elif is_first_stage:
        target_dict[k] = input_state_dict[name2[1:]].clone()
        print(f'These weights are newly added from first stage: {k}')
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

assert len(target_dict) == len(scratch_dict)
model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')

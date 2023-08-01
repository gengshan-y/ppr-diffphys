from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from torch.utils.data import DataLoader
from utils.io import config_to_dataloader

def eval_loader(opts_dict):
    num_workers = 0
   
    dataset = config_to_dataloader(opts_dict,is_eval=True)
    dataset = DataLoader(dataset,
         batch_size= 1, num_workers=num_workers, drop_last=False, pin_memory=True, shuffle=False)
    return dataset

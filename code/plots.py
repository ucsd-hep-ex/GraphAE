import pandas as pd
import numpy as np
import tqdm
import torch
import math
from torch_geometric.data import Data, DataListLoader, Batch
from torch.utils.data import random_split
import os.path as osp
import matplotlib.pyplot as plt
from graph_data import GraphDataset
import sys
from models import EdgeNet

device = 'cuda:0'
batch_size = 4

def collate(items):
    l = sum(items, [])
    return Batch.from_data_list(l)

def get_data():
    gdata = GraphDataset(root='/anomalyvol/data/gnn_node_global_merge', bb=0)
    fulllen = len(gdata)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    torch.manual_seed(0)
    train_dataset, valid_dataset, test_dataset = random_split(gdata, [fulllen-2*tv_num,tv_num,tv_num])

    train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    train_loader.collate_fn = collate
    valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    valid_loader.collate_fn = collate
    test_loader = DataListLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader.collate_fn = collate
    
    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    test_samples = len(test_dataset)

    return train_loader, valid_loader, test_loader, train_samples, valid_samples, test_samples

def get_model(model_name):
    # specific for gnn_geom for now
    model = EdgeNet().to(device)
    modpath = osp.join('/anomalyvol/models/',model_name+'.best.pth')
    try:
        model.load_state_dict(torch.load(modpath))
    except:
        sys.exit("Model not found at: " + modpath)
    
    return model

# helper for appending 3 lists
def in_out_diff_append(diff, output, inputs, i, ft_idx):
    diff.append(((output_x[i][:,ft_idx]-input_x[i][:,ft_idx])/input_x[i][:,ft_idx]).flatten())
    output.append(output_x[i][:,ft_idx].flatten())
    inputs.append(input_x[i][:,ft_idx].flatten())

def in_out_diff_concat(diff, output, inputs):
    diff = np.concatenate(diff)
    output = np.concatenate(output)
    inputs = np.concatenate(inputs)
    return [diff, output, inputs]

def make_hists(diff, output, inputs, bin1, feat):
    plt.figure(figsize=(1,1))
    plt.hist(inputs, bins=bin1,alpha=0.5)
    plt.hist(output, bins=bin1,alpha=0.5)
    plt.show()

    plt.figure()
    plt.hist(diff, bins=np.linspace(-5, 5, 101))
    plt.save('figures/' + feat + '.pdf')

def gen_plots(model_name):
    model = get_model(model_name)
    train_loader, valid_loader, test_loader, train_samples, valid_samples, test_samples = get_data()

    input_x = []
    output_x = []
    t = enumerate(test_loader)
    for i, data in t:
        data[0].to(device)
        input_x.append(data[0].x.cpu().numpy())
        output_x.append(model(data[0]).cpu().detach().numpy())

    diff_px = []
    output_px = []
    input_px = []
    diff_py = []
    output_py = []
    input_py = []
    diff_pz = []
    output_pz = []
    input_pz = []
    diff_e = []
    output_e = []
    input_e = []

    # get output in readable format
    for i in range(len(input_x)):
        # px
        in_out_diff_append(diff_px, output_px, input_px, i, 0)
        in_out_diff_append(diff_py, output_py, input_py, i, 1)
        in_out_diff_append(diff_pz, output_pz, input_pz, i, 2)
        in_out_diff_append(diff_e, output_e, input_e, i, 3)

    # remove extra brackets
    diff_px, output_px, input_px = in_out_diff_concat(diff_px, output_px, input_px)
    diff_py, output_py, input_py = in_out_diff_concat(diff_py, output_py, input_py)
    diff_pz, output_pz, input_pz = in_out_diff_concat(diff_pz, output_pz, input_pz)
    diff_e, output_e, input_e = in_out_diff_concat(diff_e, output_e, input_e)
    
    # make plots
    feat = 'px'
    bins = np.linspace(-20, 20, 101)
    make_hists(diff_px, output_px, input_px, bins, feat)

    feat = 'py'
    make_hists(diff_py, output_py, input_py, bins, feat)

    feat = 'pz'
    make_hists(diff_pz, output_pz, input_pz, bins, feat)

    feat = 'e'
    bins = np.linspace(-5, 35, 101)
    make_hists(diff_e, output_e, input_e, bins, feat)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="name of model file", required=True)
    args = parser.parse_args()
    
    gen_plots(args.model_name)
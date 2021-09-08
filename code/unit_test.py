import torch
import random
import numpy as np
import energyflow as ef
from tqdm import tqdm
from itertools import chain
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, DataLoader

import models.models as models
import models.emd_models as emd_models
from util.scaler import Standardizer
from datagen.graph_data_gae import GraphDataset
from util.train_util import get_model, forward_loss
from util.plot_util import *
from util.loss_util import LossFunction

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_standardization(data):
    scaler1 = Standardizer()
    scaler2 = StandardScaler()
    scaler1.fit(data)
    scaler2.fit(data.numpy())
    t1 = scaler1.transform(data)
    t2 = torch.tensor(scaler2.transform(data),dtype=torch.float32)
    assert torch.allclose(t1, t2, atol=1e-7), "Custom scaler not matching sklearn"
    assert not torch.allclose(t1, data, atol=1e-7), "Did not transform data"

    t3 = scaler1.inverse_transform(t1)
    t4 = torch.tensor(scaler2.inverse_transform(t2.numpy()),dtype=torch.float32)

    assert torch.allclose(t1, t2, atol=1e-7), "Inverse transformations do not match"
    return "standardization good"

def test_plot_jet_images():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod-path', type=str, help='model save file', required=True)
    parser.add_argument('--plot-dir', type=str, help='plot save dir', required=True)
    args = parser.parse_args()

    gdata = GraphDataset(root='/anomalyvol/data/bb_train_sets/test_rel/', bb=0)
    dataset = [data for data in chain.from_iterable(gdata)]
    data_x = torch.cat([d.x for d in dataset])

    scaler = Standardizer()
    scaler.fit(data_x)

    random.Random(0).shuffle(dataset)
    small_sample = dataset[:10]

    model = get_model('EdgeNet', input_dim=3, hidden_dim=2, big_dim=32, emd_modname=None)
    model.load_state_dict(torch.load(args.mod_path, map_location=device))
    model.to(device)
    model.eval()

    for i, batch in enumerate(small_sample):
        plot_jet_images(batch.x.numpy(), args.plot_dir, f'jet_input_{i}')
        batch.x[:,:] = scaler.transform(batch.x)
        batch.to(device)
        out = model(batch)
        out = scaler.inverse_transform(out.detach().cpu())
        plot_jet_images(out.numpy(), args.plot_dir, f'jet_output_{i}')

def test_reco_relative_diff():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod-path', type=str, help='model save file', required=True)
    parser.add_argument('--plot-dir', type=str, help='plot save dir', required=True)
    parser.add_argument('--model', type=str, help='name of model class', required=False, default='EdgeNet')
    args = parser.parse_args()

    gdata = GraphDataset(root='/anomalyvol/data/bb_train_sets/test_rel/', bb=0)
    dataset = [data for data in chain.from_iterable(gdata)]

    random.Random(0).shuffle(dataset)
    small_sample = dataset[:int(0.10 * len(dataset))]
    loader = DataLoader(small_sample, batch_size=256)

    data_x = torch.cat([d.x for d in small_sample])
    scaler = Standardizer()
    scaler.fit(data_x)

    model = get_model(args.model, input_dim=3, hidden_dim=2, big_dim=32, emd_modname=None)
    model.load_state_dict(torch.load(args.mod_path, map_location=device))
    model.to(device)
    model.eval()

    jet_in = []
    jet_out = []
    for batch in tqdm(loader):
        batch.x[:,:] = scaler.transform(batch.x)
        jet_in.append(batch.x.numpy())
        batch.to(device)
        out = model(batch)
        # out = scaler.inverse_transform(out.detach().cpu())
        jet_out.append(out.detach().cpu().numpy())

    jet_in = np.concatenate(jet_in)
    jet_out = np.concatenate(jet_out)
    save_dir = args.plot_dir
    save_name = 'reco_rel_difference'

    reco_relative_diff(jet_in, jet_out, save_dir, save_name)

def test_plot_emd_corr():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod-path', type=str, help='gae model save file', required=True)
    parser.add_argument('--plot-dir', type=str, help='plot save dir', required=True)
    args = parser.parse_args()

    gdata = GraphDataset(root='/anomalyvol/data/bb_train_sets/test_rel/', bb=0)
    dataset = [data for data in chain.from_iterable(gdata)]

    random.Random(0).shuffle(dataset)
    small_sample = dataset[:int(0.10 * len(dataset))]
    loader = DataLoader(small_sample, batch_size=1)

    data_x = torch.cat([d.x for d in small_sample])
    scaler = Standardizer()
    scaler.fit(data_x)

    model = get_model('EdgeNet', input_dim=3, hidden_dim=2, big_dim=32, emd_modname=None)
    model.load_state_dict(torch.load(args.mod_path, map_location=device))
    model.to(device)
    model.eval()

    lf = LossFunction('emd_loss')

    pred_emd = []
    true_emd = []
    for batch in tqdm(loader):
        batch.x[:,:] = scaler.transform(batch.x)
        batch.to(device)
        out = model(batch)
        try:
            out = scaler.inverse_transform(out.detach().cpu())
            jet_in = scaler.inverse_transform(batch.x.detach().cpu())
            emd_val = ef.emd.emd(jet_in.numpy(), out.numpy(), n_iter_max=500000)
        except RuntimeError as err:
            Path('debug').mkdir(exist_ok=True)
            np.save('debug/emd_jet_in', jet_in.numpy())
            np.save('debug/emd_jet_out', out.numpy())
            raise RuntimeError(err)

        true_emd.append(emd_val)

        batch.x[:,:] = scaler.inverse_transform(batch.x.detach().cpu()).to(device)
        out = scaler.inverse_transform(out.detach().cpu()).to(device)
        emd_loss = lf.loss_ftn(batch.x, out, batch.batch, mean=False)
        pred_emd.append(emd_loss)

    pred_emd = torch.cat(pred_emd).numpy()
    save_dir = args.plot_dir
    save_name = 'emd_correlation'

    plot_emd_corr(true_emd, pred_emd, save_dir, save_name)

if __name__ == '__main__':
    # test_plot_emd_corr()
    test_reco_relative_diff()

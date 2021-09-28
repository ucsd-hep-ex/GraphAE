"""
Generate graphs for bump hunting on invariant mass.

example:
python bump_hunt.py --model-name EdgeNet_emd_1k --model EdgeNet --loss emd_loss --box-num 1
"""
import sys
import tqdm
import math
import torch
import random
import inspect
import numpy as np
import pandas as pd
import mplhep as hep
import os.path as osp
import pyBumpHunter as BH
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from sklearn import metrics
from scipy.optimize import curve_fit
from torch_geometric.loader import DataListLoader

import models.models as models
from util.loss_util import LossFunction
from datagen.graph_data_gae import GraphDataset, collate
from util.plot_util import loss_distr, plot_reco_difference

plt.style.use(hep.style.CMS)
random.seed(0)
np.random.seed(seed=0)
torch.manual_seed(0)

def invariant_mass(jet1_e, jet1_px, jet1_py, jet1_pz, jet2_e, jet2_px, jet2_py, jet2_pz):
    """
        Calculates the invariant mass between 2 jets. Based on the formula:
        m_12 = sqrt((E_1 + E_2)^2 - (p_x1 + p_x2)^2 - (p_y1 + p_y2)^2 - (p_z1 + p_z2)^2)

        Args:
            jet1_(e, px, py, pz) (torch.float): 4 momentum of first jet of dijet
            jet2_(e, px, py, pz) (torch.float): 4 momentum of second jet of dijet

        Returns:
            torch.float dijet invariant mass.
    """
    return torch.sqrt(torch.square(jet1_e + jet2_e) - torch.square(jet1_px + jet2_px)
                      - torch.square(jet1_py + jet2_py) - torch.square(jet1_pz + jet2_pz))

def make_bump_graph(nonoutlier_mass, outlier_mass, x_lab, save_name, bins):
    """
    Create matplotlib graphs, overlaying histograms of invariant mass for outliers and all events.

    Args:
        nonoutlier_mass (tensor): inv mass of jets from nonoutlier event
        outlier_mass (tensor): inv mass from outlier events
        x_lab (str): x axis label for graph
        save_name (str): what name to save graph pdf as
        bins (np.linspace): the bins for the histogram
    """
    # plot mjj bump histograms
    plt.figure(figsize=(6,4.4))
    weights = np.ones_like(outlier_mass) / len(outlier_mass)
    plt.hist(outlier_mass, alpha = 0.5, bins=bins, weights=weights, label='Outlier events')
    weights = np.ones_like(nonoutlier_mass) / len(nonoutlier_mass)
    plt.hist(nonoutlier_mass, alpha = 0.5, bins=bins, weights=weights, label='Nonoutlier events')
    plt.legend()
    plt.xlabel(x_lab, fontsize=16)
    plt.ylabel('Normalized events [a. u.]', fontsize=16)
    plt.tight_layout()
    plt.savefig(osp.join(save_name+'.pdf'))
    plt.close()

def bump_hunter(nonoutlier_mass, outlier_mass, save_name):
    """
        Create pyBumpHunter graphs that scan for signal bumps.

        nonoutlier_mass (np.array) - background data
        outlier_mass (np.array) - data flagged as anomalous
        save_name (str) - name to save bump hunt graphs
    """
    signal_mjj = np.array([1, 3, 6, 10, 16, 23, 31, 40, 50, 61, 74, 88, 103, 119, 137, 156, 
              176, 197, 220, 244, 270, 296, 325, 354, 386, 419, 453, 489, 526, 
              565, 606, 649, 693, 740, 788, 838, 890, 944, 1000, 1058, 1118, 
              1181, 1246, 1313, 1383, 1455, 1530, 1607, 1687, 1770, 1856, 
              1945, 2037, 2132, 2231, 2332, 2438, 2546, 2659, 2775, 2895, 
              3019, 3147, 3279, 3416, 3558, 3704, 3854, 4010, 4171, 4337, 
              4509, 4686, 4869, 5058, 5253, 5455, 5663, 5877, 6099, 6328, 
              6564, 6808, 7060, 7320, 7589, 7866, 8152, 8447, 8752, 9067, 
              9391, 9726, 10072, 10430, 10798, 11179, 11571, 11977, 12395, 
              12827, 13272, 13732, 14000])

    # first perform a fit to improve background prediciton
    bins = signal_mjj[(signal_mjj >= 2659) * (signal_mjj <= 6099)]

    xmin = bins[0]
    xmax = bins[-1]
    
    # define fit function.
    def fit_function(x, p0, p1, p2, p3, p4):
        xnorm = (x-xmin)/(xmax-xmin)
        return p0*(1-xnorm)**4 + 4*p1*xnorm*(1-xnorm)**3 + 6*p2*xnorm**2*(1-xnorm)**2 + 4*p3*xnorm**3*(1-xnorm) + p4*xnorm**4

    # do the fit
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    
    outlier_hist, _ = np.histogram(outlier_mass, bins=bins)
    nonoutlier_hist, _ = np.histogram(nonoutlier_mass, bins=bins)
    ratio = outlier_hist/nonoutlier_hist
    mask = (~np.isnan(ratio)) * (~np.isinf(ratio))
    popt, pcov = curve_fit(fit_function, 
                           xdata=binscenters[mask], 
                           ydata=ratio[mask], 
                           p0=[0.1]*5,
                           bounds=(0, [np.inf, np.inf, np.inf, np.inf, np.inf]))
    
    # save fit reesult in plot
    f, axs = plt.subplots(1,2, figsize=(16, 6))
    axs[0].hist(nonoutlier_mass,bins=bins,alpha=0.5, label='Nonoutliers')
    axs[0].hist(outlier_mass,bins=bins,alpha=0.5, label='Outliers')
    axs[0].set_xlim(xmin, xmax)
    #axs[0].set_ylim(1, 1e4)
    axs[0].set_xlabel(r'$m_{jj}$ [GeV]')
    axs[0].set_ylabel(r'Events')
    axs[0].legend(title='Prefit')
    axs[0].semilogy()

    axs[1].hist(nonoutlier_mass,bins=bins,weights=fit_function(nonoutlier_mass, *popt),alpha=0.5, label='Nonoutliers')
    axs[1].hist(outlier_mass,bins=bins,alpha=0.5, label='Outliers')
    axs[1].set_xlabel(r'$m_{jj}$ [GeV]')
    axs[1].set_ylabel(r'Events')
    axs[1].set_xlim(xmin, xmax)
    #axs[1].set_ylim(1,1e4)
    axs[1].legend(title='Postfit')
    axs[1].semilogy()
    f.savefig(save_name+'_postfit.pdf')
    
    # Plot the histogram and the fitted function
    # Generate enough x values to make the curves look smooth.
    xspace = np.linspace(xmin, xmax, 100000)
    nonoutlier_hist_weighted, _ = np.histogram(nonoutlier_mass, bins=bins, weights=fit_function(nonoutlier_mass, *popt))
    weighted_ratio = outlier_hist/nonoutlier_hist_weighted
    weighted_mask = (~np.isnan(weighted_ratio)) * (~np.isinf(weighted_ratio))
    
    f, axs = plt.subplots(1,2, figsize=(16, 3))
    axs[0].plot(binscenters[mask], ratio[mask], color='navy', label=r'Prefit', marker='o',linestyle='')
    axs[0].plot(xspace, fit_function(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')
    axs[0].set_xlabel(r'$m_{jj}$ [GeV]')
    axs[0].set_xlim(xmin, xmax)
    #axs[0].set_ylim(0, 2)
    axs[0].set_ylabel(r'Ratio')
    axs[0].legend()
    axs[1].plot(binscenters[weighted_mask], weighted_ratio[weighted_mask], color='navy', label=r'Postfit', marker='o',linestyle='')
    axs[1].plot(xspace, np.ones_like(xspace), color='gray', linewidth=2.5)
    axs[1].set_xlabel(r'$m_{jj}$ [GeV]')
    axs[1].set_ylabel(r'Ratio')
    axs[1].set_xlim(xmin, xmax)
    #axs[1].set_ylim(0, 2)
    axs[1].legend(loc='upper right')
    f.savefig(save_name+'_ratio.pdf')
    
    # bump hunter
    # now reweight the background prediction to make it more accurate
    weights = fit_function(nonoutlier_mass, *popt)
    bh = BH.BumpHunter1D(rang=[xmin, xmax],
                         bins=bins,
                         weights=weights,
                         width_min=2,
                         width_max=5,
                         npe=10000,
                         seed=42)
    bh.bump_scan(outlier_mass, nonoutlier_mass)
    sys.stdout = open(save_name+'.txt', "w")
    bh.print_bump_true(outlier_mass, nonoutlier_mass)
    sys.stdout = sys.__stdout__
    bh.plot_bump(data=outlier_mass, bkg=nonoutlier_mass, filename=save_name+'.pdf', x_label=r'$m_{jj}$ [GeV]')
    bh.plot_stat(show_Pval=True, filename=save_name+'_stat.pdf')

def process(data_loader, model_path, model, loss_ftn_obj, latent_dim, features):
    """
    Use the specified model to determine the reconstruction loss of each sample.
    Also calculate the invariant mass of the jets.

    Args:
        data_loader (torch.data.DataLoader): pytorch dataloader for loading in black boxes
        model_path (str): path to saved model
        model (str): name of model class
        loss_ftn_obj (LossFunction): see loss_util.py
        latent_dim (int): latent dimension of the model

    Returns: torch.tensor of size (num_events, 5).
             column-wise: [jet1_loss, jet2_loss, dijet_invariant_mass, jet1_mass, jet2_mass, rnd_truth_bit]
             Row-wise: dijet of event
        
    """

    # load corresponding model
    if model == 'MetaLayerGAE':
        model = models.GNNAutoEncoder()
    else:
        input_dim = 3
        model = getattr(models, model)(input_dim=input_dim, hidden_dim=latent_dim)
    modpath = glob(osp.join(model_path,'*.best.pth'))[0]
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(modpath, map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(modpath, map_location=torch.device('cpu')))
    model.eval()

    # Store the return values
    jets_proc_data = []
    input_fts = []
    reco_fts = []

    event = 0
    # for each event in the dataset calculate the loss and inv mass for the leading 2 jets
    with torch.no_grad():
        for k, data_batch in tqdm.tqdm(enumerate(data_loader),total=len(data_loader)):
            # select appropriate features based on what model was trained on
            if features == 'xyz':
                data_batch.x = data_batch.x[:,:3]
            else:
                data_batch.x = data_batch.x[:,4:7]

            jets_x = data_batch.x
            batch = data_batch.batch
            jets_u = data_batch.u
            jets0_u = jets_u[::2]
            jets1_u = jets_u[1::2]
            # run inference on all jets
            if loss_ftn_obj.name == 'vae_loss':
                jets_rec, mu, log_var = model(data_batch)
            else:
                jets_rec = model(data_batch)
            
            # calculate invariant mass (data.u format: p[event_idx, n_particles, jet.mass, jet.px, jet.py, jet.pz, jet.e]])
            dijet_mass = invariant_mass(jets0_u[:,6], jets0_u[:,3], jets0_u[:,4], jets0_u[:,5],
                                        jets1_u[:,6], jets1_u[:,3], jets1_u[:,4], jets1_u[:,5])
            njets = len(torch.unique(batch))
            losses = torch.zeros((njets), dtype=torch.float32)
            # calculate loss per each batch (jet)
            for ib in torch.unique(batch):
                if loss_ftn_obj.name == 'vae_loss':
                    losses[ib] = loss_ftn_obj.loss_ftn(jets_rec[batch==ib], jets_x[batch==ib], mu, log_var)
                elif loss_ftn_obj.name == 'emd_loss' or loss_ftn_obj.name == 'chamfer_loss' or loss_ftn_obj.name == 'hungarian_loss':
                    losses[ib] = loss_ftn_obj.loss_ftn(jets_rec[batch==ib], jets_x[batch==ib], torch.tensor(0).repeat(jets_rec[batch==ib].shape[0]))
                else:
                    losses[ib] = loss_ftn_obj.loss_ftn(jets_rec[batch==ib], jets_x[batch==ib])

            loss0 = losses[::2]
            loss1 = losses[1::2]
            jets_info = torch.stack([loss0,
                                     loss1,
                                     dijet_mass,              # mass of dijet
                                     jets0_u[:,2],            # mass of jet 1
                                     jets1_u[:,2],            # mass of jet 2
                                     jets1_u[:,-1]],          # if this event was an anomaly
                                    dim=1)
            jets_proc_data.append(jets_info)
            input_fts.append(jets_x[::2])
            input_fts.append(jets_x[1::2])
            reco_fts.append(jets_rec[::2])
            reco_fts.append(jets_rec[1::2])
            event += njets/2
    # return pytorch tensors
    return torch.cat(jets_proc_data), torch.cat(input_fts), torch.cat(reco_fts)

def bump_hunt(df, cuts, loss_name, bb, save_path):
    """
    Loops and makes multiple cuts on the jet losses, and generates a graph for each cut by
    delegating to make_bump_graph().

    Args:
        df (pd.DataFrame): output of process() transformed into datafram; has loss and mass of jets per event
        cuts (list of floats): all the percentages to perform a cut on the loss
        loss_name (str): name of loss function
        bb (str): which black box the bump hunt is being performed on (e.g. 'bb1')
    """
    losses = np.concatenate([df['loss1'], df['loss2']])
    loss_distr(losses,  osp.join(save_path,'loss_distribution'))

    # generate a graph for different cuts
    for cut in cuts:
        # name for graph files when saved
        dijet_graph_name = '{save_path}/dijet_bump_{cut:.2f}'.format(save_path=save_path, cut=cut)
        mj1_graph_name = '{save_path}/mj1_bump_{cut:.2f}'.format(save_path=save_path, cut=cut)
        mj2_graph_name = '{save_path}/mj2_bump_{cut:.2f}'.format(save_path=save_path, cut=cut)

        loss_thresh = np.quantile(losses, cut)
        # classify dijet as outlier if both jets are outliers
        df['outlier'] = (np.minimum(df['loss1'], df['loss2']) > loss_thresh)
        outliers = df[df['outlier']]
        # otherwise, classify dijet as nonoutlier
        nonoutliers = df[~df['outlier']]
        # alternative definition:
        #df['nonoutlier'] = (np.maximum(df['loss1'], df['loss2']) < loss_thresh)
        #nonoutliers = df[df['nonoutlier']]

        # make dijet bump hunt graph
        all_dijet_mass = df['dijet_mass']
        nonoutlier_dijet_mass = nonoutliers['dijet_mass']
        outlier_dijet_mass = outliers['dijet_mass'] # get the mass of only outliers

        x_lab = '$m_{jj}$ [GeV]'
        bins = np.linspace(1000, 6000, 51)
        make_bump_graph(nonoutlier_dijet_mass, outlier_dijet_mass, x_lab, dijet_graph_name, bins)
        bump_hunter(nonoutlier_dijet_mass, outlier_dijet_mass, dijet_graph_name + '_bumphunter_ver')

        # make graph for mj1
        all_m1_mass = df['mass1']
        nonoutlier_m1_mass = nonoutliers['mass1']
        outlier_m1_mass = outliers['mass1']
        x_lab = '$m_{j1}$ [GeV]'
        bins = np.linspace(0, 1800, 51)
        make_bump_graph(nonoutlier_m1_mass, outlier_m1_mass, x_lab, mj1_graph_name, bins)

        # make graph for mj2
        all_m2_mass = df['mass2']
        nonoutlier_m2_mass = nonoutliers['mass2']
        outlier_m2_mass = outliers['mass2']
        x_lab = '$m_{j2}$ [GeV]'
        bins = np.linspace(0, 1800, 51)
        make_bump_graph(nonoutlier_m2_mass, outlier_m2_mass, x_lab, mj2_graph_name, bins)

    if 'rnd' in bb:  # plot roc for rnd set
        df['loss_sum'] = df['loss1']+df['loss2']
        df['loss_min'] = np.minimum(df['loss1'],df['loss2'])
        df['loss_max'] = np.maximum(df['loss1'],df['loss2'])

        if loss_name == 'chamfer_loss':
            loss_type = '$D^{NN}$'
        elif loss_name == 'MSE':
            loss_type = 'MSE'
        elif loss_name == 'hungarian_loss':
            loss_type = 'Hung.'
        elif loss_name == 'emd_loss':
            loss_type = 'emd-nn'

        plt.figure(figsize=(8,6))
        plt.style.use(hep.style.CMS)
        lw = 2            
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

        for var in ['min']:
            fpr, tpr, thresholds = metrics.roc_curve(df['truth_bit'],df['loss_'+var])
            auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, 
                     lw=lw, label='ROC curve (AUC = %0.2f)' % (auc))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.legend(title='R&D dataset, ' + loss_type,loc="lower right")
        plt.tight_layout()
        plt.savefig(osp.join(save_path,'roc.pdf'))
        plt.close()

def main(args):
    # validate arguments
    if args.num_events <= 0 or args.num_events > 1000000:
        exit("--num_events must be in range (0, 1000000]")
    if args.latent_dim <= 0:
        exit("--latent_dim must be greater than 0")
    if args.box_num not in [0, 1, 2, 4]:
        exit("--box_num invalid; must be 0, 1, 2, or 4")

    loss_ftn_obj = LossFunction(args.loss)
    model_path = args.model_path
    model_fname = osp.basename(glob(osp.join(model_path,'*.best.pth'))[0]).split('.')[0]
    model = args.model
    num_events = args.num_events
    latent_dim = args.latent_dim
    output_dir = args.output_dir
    overwrite = args.overwrite
    box_num = args.box_num
    features = args.features
    cuts = np.arange(0.5, 1.0, 0.1)

    def get_df(proc_jets):
        d = {'loss1': proc_jets[:,0],
             'loss2': proc_jets[:,1],
             'dijet_mass': proc_jets[:,2],
             'mass1': proc_jets[:,3],
             'mass2': proc_jets[:,4],
             'truth_bit': proc_jets[:,5]}
        df = pd.DataFrame(d)
        return df

    # read in dataset
    bb_name = ["bb0_xyz_pyg2", "bb1_xyz_pyg2", "bb2_xyz_pyg2", "bb3_xyz_pyg2", "rnd_xyz_pyg2"][box_num]
    print("Plotting %s"%bb_name)

    save_path = osp.join(output_dir,model_fname,'bump_hunt',bb_name)
    Path(save_path).mkdir(parents=True,exist_ok=True) # make a subfolder

    if not osp.isfile(osp.join(save_path,'df.pkl')) or overwrite:
        print("Processing jet losses")
#         gdata = GraphDataset('/anomalyvol/data/lead_2/tiny', n_events=num_events, bb=box_num, features=features)
        gdata = GraphDataset('/anomalyvol/data/lead_2/%s/'%bb_name, n_events=num_events, bb=box_num, features=features)
        bb_loader = DataListLoader(gdata, batch_size=1, pin_memory=True, shuffle=False)
        bb_loader.collate_fn = collate
        proc_jets, input_fts, reco_fts = process(bb_loader, model_path, model, loss_ftn_obj, latent_dim, features)
        df = get_df(proc_jets)
        df.to_pickle(osp.join(save_path,'df.pkl'))
        torch.save(input_fts, osp.join(save_path,'input_fts.pt'))
        torch.save(reco_fts, osp.join(save_path,'reco_fts.pt'))
    else:
        print("Using preprocessed dictionary")
        df = pd.read_pickle(osp.join(save_path,'df.pkl'))
        input_fts = torch.load(osp.join(save_path,'input_fts.pt'))
        reco_fts = torch.load(osp.join(save_path,'reco_fts.pt'))
    plot_reco_difference(input_fts, reco_fts, model_fname, save_path)
    bump_hunt(df, cuts, args.loss, bb_name, save_path)

if __name__ == "__main__":

    # process arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Directory to saved model discluding the model file", required=True)
    parser.add_argument("--output-dir", type=str, help="Output directory for files.", required=False, default='/anomalyvol/experiments/')
    parser.add_argument("--model", choices=[m[0] for m in inspect.getmembers(models, inspect.isclass) if m[1].__module__ == 'models.models'], help="model selection", required=True)
    parser.add_argument("--overwrite", action='store_true', help="Toggle overwrite of pkl. Default False.", default=False, required=False)
    parser.add_argument("--num-events", type=int, help="How many events to process (multiple of 100). Default 1mil", default=1000000, required=False)
    parser.add_argument("--latent-dim", type=int, help="How many units for the latent space (def=2)", default=2, required=False)
    parser.add_argument('--loss', choices=[m for m in dir(LossFunction) if not m.startswith('__')], 
                        help='loss function', required=True)
    parser.add_argument("--box-num", type=int, help="0=QCD-background; 1=bb1; 2=bb2; 4=rnd", required=True)
    parser.add_argument("--features", choices=['xyz','relptetaphi'], help="Generate (px,py,pz) or relative (pt,eta,phi)", required=True)
    args = parser.parse_args()

    main(args)

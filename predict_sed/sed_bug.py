import pandas as pd
from sedpy.observate import load_filters
from prospect.models import priors, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    "savefig.facecolor": "w",
    "figure.facecolor" : 'w',
    "figure.figsize" : (10,8),
    "text.color": "k",
    "legend.fontsize" : 20,
    "font.size" : 30,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "axes.linewidth": 3,
    "xtick.color": "k",
    "ytick.color": "k",
    "xtick.labelsize" : 25,
    "ytick.labelsize" : 25,
    "ytick.major.size" : 12,
    "xtick.major.size" : 12,
    "ytick.major.width" : 2,
    "xtick.major.width" : 2,
    "font.family": 'STIXGeneral',
    "mathtext.fontset" : "cm"
})

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import FastStepBasis
    sps = FastStepBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps
sps = build_sps()


def build_model_with_true_params(galaxy):
    from prospect.models import priors, sedmodel
    import scipy.stats
    model_params = []
    print('building model')
    model_params.append({'name': "lumdist", "N": 1, "isfree": False,"init": 1.0e-5,"units": "Mpc"})
    model_params.append({'name': 'imf_type', 'N': 1,'isfree': False,'init': 2})
    model_params.append({'name': 'logmass', 'N': 1,'isfree': False,'init': props['true_mass'][galaxy],'prior': None})
    model_params.append({'name': 'logzsol', 'N': 1,'isfree': False,'init': props['true_logzsol'][galaxy],'prior': None})
    
    print(f"log stellar mass: {props['true_mass'][galaxy]}")
    
    i = galaxy
    ids = np.array(sfh_data['id'])
    massform = np.array(sfh_data['massform'], dtype=object)[ids==i][0]
    tform = np.array(sfh_data['tform'],dtype=object)[ids==i][0]
    t_H = 13.87 # age of universe @ z=2
    binwidth = 0.1 # put SFH in 100 Myr bins
    bins = np.arange(0, t_H, binwidth) 
    sfr, bins, binnumber = scipy.stats.binned_statistic(tform, massform, statistic=get_sfr, bins=bins)
    bincenters = 0.5*(bins[:-1]+bins[1:])
    
    masses_formed = np.trapz(sfr, bincenters*1e9)

    print(f'masses formed: {np.log10(masses_formed)}')
    model_params.append({'name': "sfh", "N": 1, "isfree": False, "init": 3})
    model_params.append({'name': "true_sfh", 'N': len(sfr), 'isfree': False, 'init': sfr})
    model_params.append({'name': "sfh_time", 'N': len(sfr), 'isfree': False,'init': bincenters})
    model_params.append({'name': 'mass', 'N': 1, 'isfree': False,'init': masses_formed})
    #turn everything else off
    model_params.append({'name': 'dust_type', 'N': 1,'isfree': False,'init': 4,'prior': None})
    model_params.append({'name': 'dust2', 'N': 1,'isfree': False, 'init': 0.0,'prior': None})
    model_params.append({'name': 'add_dust_emission', 'N': 1,'isfree': False,'init': 0})
    model_params.append({'name': 'add_agb_dust_model', 'N': 1,'isfree': False,'init': 0})
                        
    model = sedmodel.SedModel(model_params)
    return model
def get_sfr(massform):
    binwidth = 0.1
    return np.sum(massform)/(binwidth*1e9)

galaxies = [221,898,58]
galaxy = galaxies[1]
props = pd.read_pickle('galaxy_props.pkl')
obs = pd.read_pickle(f'galaxy_{galaxy}_obs.pkl')
sfh_data = pd.read_pickle('sfh_m25n512_z0.pickle') #mformed as a function of tform, later binned to get SFH

mod =build_model_with_true_params(galaxy)

spec, phot, x = mod.predict([], obs, sps)

plt.loglog(obs['true_wavelength'], obs['true_spectrum'], label='True SED')
plt.scatter([x.wave_mean for x in obs['filters']], obs['maggies'], zorder=10, alpha=0.5, s=100)

plt.plot(sps.wavelengths, spec, color='gray', label='Prospector SED')

#plt.ylim([1e6, 1e10])
plt.xlim([8e2, 5e4])

plt.ylabel('Flux')
plt.xlabel('Wavelength')
plt.savefig(f'sed_{galaxy}.png')


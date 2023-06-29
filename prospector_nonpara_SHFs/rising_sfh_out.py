import numpy as np
import astropy.units as u
from tqdm.auto import tqdm
import pandas as pd
import sys
import prospect.io.read_results as pread
from corner import quantile
from prospect.models import transforms

res, _ , _ = pread.results_from(f'CF_galaxy6.h5')
sps = pread.get_sps(res)


def get_sfr10_beta(res, mod):
    #agebins = mod.params['agebins']
    agebins = transforms.zred_to_agebins_pbeta(np.array([7.2]))
    thetas = mod.theta_labels()
    agebins_yrs = 10**agebins.T
    bin_edges = np.unique(agebins_yrs)
    dt = agebins_yrs[1, :] - agebins_yrs[0, :]
    epsilon = 1e-4 #fudge factor used to define the fraction time separation of adjacent points at the bin edges
    t = np.concatenate((bin_edges * (1.-epsilon), bin_edges * (1+epsilon)))
    t.sort()
    t = t[1:-1]
    weights = res.get('weights',None)
    idx = np.argsort(weights)[-3000:]
    sfh_bin_idx = [i for i, s in enumerate(thetas) if 'nzsfh' in s]
    sfr_chain = []
    for i in idx:
        sfh_bin_chain = res['chain'][i,3:sfh_bin_idx[-1]+1]
        total_mass_chain = res['chain'][i,1]
        sfr = transforms.logsfr_ratios_to_sfrs(total_mass_chain, sfh_bin_chain, agebins)
        sfrout = np.zeros_like(t)
        #sfrout[::2] = sfr
        #sfrout[1::2] = sfr
        sfr_chain.append(sfr[0])
    return sfr_chain


mass_quan = []
sfr_quan = []
metal_quan = []
dmass_quan = []
galaxy_list = []
spec50=[]
spec16=[]
spec84=[]
pd_phot, pd_sed=[],[]
phot_wave, pd_wave=[],[]


for galaxy in [int(sys.argv[1])]:
    try:
        res, obs, mod = pread.results_from(f'CF_galaxy{galaxy}.h5')
    except:
        continue
    galaxy_list.append(galaxy)
    spec, phot, mass_frac, dmass = [], [], [], []
    weights = res.get('weights',None)
    idx = np.argsort(weights)[-3000:]
    for i in tqdm(idx):
        sspec, pphot, mmass_frac = mod.predict(res['chain'][i], obs, sps)
        spec.append(sspec)
        phot.append(pphot)
        mass_frac.append(mmass_frac)
        #dmass.append(sps.ssp.dust_mass)
    raw_masses = res['chain'][idx,1]
    pred_masses = 10**raw_masses * mass_frac
    mass_quantiles = quantile(np.log10(pred_masses), [.16, .5, .84], weights=res['weights'][idx])
    mass_quan.append(mass_quantiles)
    sfr_chain = get_sfr10_beta(res, mod)
    sfr_quan.append(quantile(sfr_chain, [.16, .5, .84], weights=res['weights'][idx]))
    metal_quan.append(quantile(res['chain'][idx,2], [0.16, 0.5, 0.84], weights=res['weights'][idx]))
    #dmass_quan.append(quantile(np.log10(dmass), [0.16, 0.5, 0.84], weights=res['weights'][::3]))
    
    pd_phot.append(obs['maggies'])
    phot_wave.append([x.wave_mean/1e4 for x in obs['filters']])
    
    pd_sed.append(obs['pd_sed'])
    pd_wave.append(obs['pd_wav'])
    
    spec_50, spec_16, spec_84 = [], [], []
    for i in range(len(spec[0])):
        quantiles = quantile([item[i] for item in spec], [.16, .5, .84], weights=res['weights'][idx])
        spec_50.append(quantiles[1])
        spec_16.append(quantiles[0])
        spec_84.append(quantiles[2])
    spec50.append(spec_50)
    spec16.append(spec_16)
    spec84.append(spec_84)
    
    
spec_wave = []
for i in range(len(galaxy_list)):
    spec_wave.append(sps.wavelengths*(1+7.2)/1e4)

data_props = {'galaxy': galaxy_list,'log_smass_quantiles': mass_quan, 'sfr_quantiles': sfr_quan,
        'logZsol': metal_quan}

data_sed = {'galaxy': galaxy_list, 'powderday_sed': pd_sed, 'powderday_wave': [item/1e4 for item in pd_wave],
        'spec_wave':spec_wave,
       'spec_50': spec50, 'spec_16': spec16, 'spec_84': spec84, 'phot': pd_phot, 'phot_wave': phot_wave}


df_props = pd.DataFrame(data_props)
df_sed = pd.DataFrame(data_sed)


df_sed.to_pickle(f'temp/sed_galaxy{galaxy_list[0]}.pkl')
df_props.to_pickle(f'temp/props_galaxy{galaxy_list[0]}.pkl')

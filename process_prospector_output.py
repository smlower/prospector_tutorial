import numpy as np
import astropy.units as u
from tqdm.auto import tqdm
import pandas as pd
import sys
import prospect.io.read_results as pread
from corner import quantile
from prospect.models import transforms

res, _ , _ = pread.results_from('galaxy_100_nonpara_fit.h5')
sps = pread.get_sps(res)


def get_sfh(res, mod):
    agebins = mod.params['agebins']
    thetas = mod.theta_labels()
    agebins_yrs = 10**agebins.T
    bin_edges = np.unique(agebins_yrs)
    dt = agebins_yrs[1, :] - agebins_yrs[0, :]
    epsilon = 1e-4 #fudge factor used to define the fraction time separation of adjacent points at the bin edges
    t = np.concatenate((bin_edges * (1.-epsilon), bin_edges * (1+epsilon)))
    t.sort()
    t = t[1:-1]
    zfrac_idx = [i for i, s in enumerate(thetas) if 'z_fraction' in s]
    zfrac_chain = res['chain'][:,zfrac_idx[0]:zfrac_idx[-1]+1]
    try:
        total_mass_chain = res['chain'][:,thetas.index('massmet_1')]
    except:
        total_mass_chain = res['chain'][:,thetas.index('logmass')]
    sfr_chain = []
    weights = res.get('weights',None)
    idx = np.argsort(weights)[-3000:]
    for i in idx:
        masses_chain = transforms.zfrac_to_masses(10**total_mass_chain[i], zfrac_chain[i], agebins)
        sfr = masses_chain / dt
        sfrout = np.zeros_like(t)
        sfrout[::2] = sfr
        sfrout[1::2] = sfr
        sfr_chain.append(sfrout)
    return (t[-1] - t[::-1])/1e9, sfr_chain


mass_quan = []
sfr_quan = []
metal_quan = []
dmass_quan = []
pd_phot, pd_sed=[],[]
phot_wave, pd_wave=[],[]

res, obs, mod = pread.results_from(f'dust1_fixed_galaxy{galaxy}.h5')
model_params = model.theta_labels()
spec, phot, mass_frac, dmass = [], [], [], []

#Here I'm only calculating the quantiles of the 5000 most likely fit results

weights = res.get('weights',None)
idx = np.argsort(weights)[-5000:]


#if you want to process the entire posterior (which could really slow this calculation down, uncomment the next line
#idx = np.arange(len(res['chain']))

for i in tqdm(idx):
    sspec, pphot, mmass_frac = mod.predict(res['chain'][i], obs, sps)
    spec.append(sspec)
    phot.append(pphot)
    mass_frac.append(mmass_frac)
    dust_mass.append(sps.ssp.dust_mass)
raw_masses = res['chain'][idx,model_params.index('logmass')]
pred_masses = 10**raw_masses * mass_frac
mass_quan = quantile(np.log10(pred_masses), [.16, .5, .84], weights=res['weights'][idx])
dmass_quan = quantile(np.log10(dust_mass), [.16, .5, .84], weights=res['weights'][idx])
metal_quan = quantile(res['chain'][idx,1], [0.16, 0.5, 0.84], weights=res['weights'][idx])

sfh_time, sfr_chain = get_sfh(res, mod)
sfh_50, sfh_16, sfh_84 = [], [], []
for i in range(len(sfh_time)):
    sfh_quans = quantile([item[i] for item in sfr_chain], [.16, .5, .84], weights=res['weights'][idx])
    sfh_50.append(sfh_quans[1])
    sfh_16.append(sfh_quans[0])
    sfh_84.append(sfh_quans[2])


pd_phot.append(obs['maggies'])
phot_wave.append([x.wave_mean for x in obs['filters']])

pd_sed.append(obs['pd_sed'])
pd_wave.append(obs['pd_wav'])

spec_50, spec_16, spec_84 = [], [], []
for i in range(len(spec[0])):
    quantiles = quantile([item[i] for item in spec], [.16, .5, .84], weights=res['weights'][idx])
    spec_50.append(quantiles[1])
    spec_16.append(quantiles[0])
    spec_84.append(quantiles[2])

"""
Now we save results to two pickle files, one with the SED info and one with the estimates for 
stellar mass, SFH, metallicity, and dust mass

The SED file contains the original Powderday SED, the photometry sampled from the Powderday SED (phot, phot_wave)
and the model Prospector SED

The masses and metallicities are saved as the median and the 16th-84th quantiles. same for the SFH and SED

"""


data_props = {'log_smass_quantiles': mass_quan, 'sfh_time': sfh_time, 'sfh_16':sfh_16, 'sfh_50': sfh_50, 'sfh_84':sfh_84,
              'logZsol': metal_quan, 'log_dmass_quantiles': dmass_quan}

data_sed = {'powderday_sed': pd_sed, 'powderday_wave': [item for item in pd_wave],
        'spec_wave':sps.wavelengths,
       'spec_50': spec_50, 'spec_16': spec_16, 'spec_84': spec_84, 'phot': pd_phot, 'phot_wave': phot_wave}


df_props = pd.DataFrame(data_props)
df_sed = pd.DataFrame(data_sed)


df_sed.to_pickle(f'galaxy100_SEDs.pkl')
df_props.to_pickle(f'galaxy100_props.pkl')

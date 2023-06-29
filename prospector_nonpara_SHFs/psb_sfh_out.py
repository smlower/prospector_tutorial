import numpy as np
import astropy.units as u
from tqdm.auto import tqdm
import pandas as pd
import sys
import prospect.io.read_results as pread
from corner import quantile
from prospect.models import transforms

res, _ , _ = pread.results_from('CF_galaxy6.h5')
sps = pread.get_sps(res)


def SL_logsfr_ratios_to_masses_psb(logmass=None, logsfr_ratios=None,
                                 logsfr_ratio_young=None, logsfr_ratio_old=None,
                                 tlast=None, tflex=None, nflex=None, nfixed=None,
                                 agebins=None, **extras):
    """This is a modified version of logsfr_ratios_to_masses_flex above. This now
    assumes that there are nfixed fixed-edge timebins at the beginning of
    the universe, followed by nflex flexible timebins that each form an equal
    stellar mass. The final bin has variable width and variable SFR; the width
    of the bin is set by the parameter tlast.
    The major difference between this and the transform above is that
    logsfr_ratio_old is a vector.
    """

    # clip for numerical stability
    logsfr_ratio_young = np.clip(logsfr_ratio_young[0], -7, 7)
    logsfr_ratio_old = np.clip(logsfr_ratio_old, -7, 7)
    syoung, sold = 10**logsfr_ratio_young, 10**logsfr_ratio_old
    sratios = 10.**np.clip(logsfr_ratios, -7, 7) # numerical issues...

    # get agebins
    abins = SL_psb_logsfr_ratios_to_agebins(logsfr_ratios=logsfr_ratios,
            agebins=agebins, tlast=tlast, tflex=tflex, nflex=nflex, nfixed=nfixed, **extras)

    # get find mass in each bin
    dtyoung, dt1 = (10**abins[:2, 1] - 10**abins[:2, 0])
    dtold = 10**abins[-nfixed-1:, 1] - 10**abins[-nfixed-1:, 0]
    old_factor = np.zeros(nfixed)
    for i in range(nfixed):
        old_factor[i] = (1. / np.prod(sold[:i+1]) * np.prod(dtold[1:i+2]) / np.prod(dtold[:i+1]))
    mbin = 10**logmass / (syoung*dtyoung/dt1 + np.sum(old_factor) + nflex)
    myoung = syoung * mbin * dtyoung / dt1
    mold = mbin * old_factor
    n_masses = np.full(nflex, mbin)

    return np.array([myoung] + n_masses.tolist() + mold.tolist())


def SL_psb_logsfr_ratios_to_agebins(logsfr_ratios=None, agebins=None,
                                 tlast=None, tflex=None, nflex=None, nfixed=None, **extras):
    """This is a modified version of logsfr_ratios_to_agebins above. This now
    assumes that there are nfixed fixed-edge timebins at the beginning of
    the universe, followed by nflex flexible timebins that each form an equal
    stellar mass. The final bin has variable width and variable SFR; the width
    of the bin is set by the parameter tlast.
    For the flexible bins, we again use the equation:
        delta(t1) = tuniv  / (1 + SUM(n=1 to n=nbins-1) PROD(j=1 to j=n) Sn)
        where Sn = SFR(n) / SFR(n+1) and delta(t1) is width of youngest bin
    """


    # numerical stability
    logsfr_ratios = np.clip(logsfr_ratios, -7, 7)

    # flexible time is t_flex - youngest bin (= tlast, which we fit for)
    # this is also equal to tuniv - upper_time - lower_time
    tf = (tflex - tlast) * 1e9

    # figure out other bin sizes
    n_ratio = logsfr_ratios.shape[0]
    sfr_ratios = 10**logsfr_ratios
    dt1 = tf / (1 + np.sum([np.prod(sfr_ratios[:(i+1)]) for i in range(n_ratio)]))

    # translate into agelims vector (time bin edges)
    agelims = [1, (tlast*1e9), dt1+(tlast*1e9)]
    for i in range(n_ratio):
        agelims += [dt1*np.prod(sfr_ratios[:(i+1)]) + agelims[-1]]
    agelims += list(10**agebins[-nfixed:,1])
    abins = np.log10([agelims[:-1], agelims[1:]]).T

    return abins




def get_sfh_psb(res, mod):
    logmass=res['chain'][:,0]
    logsfr_ratios_chain = res['chain'][:,7:11] 
    logsfr_ratios_young_chain = res['chain'][:,3]
    logsfr_ratios_old_chain = res['chain'][:,4:7]
    tlast_chain = res['chain'][:,2]
    tflex = 0.37112653
    nflex=5
    nfixed=3
    agebins = mod.params['agebins']
    sfr_chain = []
    time_chain = []
    
    weights = res.get('weights',None)
    idx = np.argsort(weights)[-3000:]
    
    for i in idx:
        #print(logsfr_ratios_young_chain[i])
        masses = SL_logsfr_ratios_to_masses_psb(logmass=logmass[i], logsfr_ratios=logsfr_ratios_chain[i],
                                 logsfr_ratio_young=[logsfr_ratios_young_chain[i]], 
                                logsfr_ratio_old=logsfr_ratios_old_chain[i],
                                 tlast=tlast_chain[i], tflex=tflex, nflex=nflex, nfixed=nfixed,
                                 agebins=agebins)

        
        fit_agebins = SL_psb_logsfr_ratios_to_agebins(logsfr_ratios=logsfr_ratios_chain[i], agebins=agebins,
                                 tlast=tlast_chain[i], tflex=tflex, nflex=nflex, nfixed=nfixed)
    
        #print(10**fit_agebins)
        #dt = (10**fit_agebins[:, 1] - 10**fit_agebins[:, 0])
        agebins_yrs = 10**fit_agebins.T
        bin_edges = np.unique(agebins_yrs)
        dt = agebins_yrs[1, :] - agebins_yrs[0, :]
        epsilon = 1e-4 #fudge factor used to define the fraction time separation of adjacent points at the bin edges
        t = np.concatenate((bin_edges * (1.-epsilon), bin_edges * (1+epsilon)))
        t.sort()
        t = t[1:-1]
        #print(f'age bin years {agebins_yrs}')
        #print(f'bin edges {bin_edges}')
        #print(f't {t}')
        #print(f'dt {dt}')
        #print(f'masses {masses}')
        sfr = masses/dt
        sfrout = np.zeros_like(t)
        sfrout[::2] = sfr
        sfrout[1::2] = sfr
        sfr_chain.append(sfrout)
        time_chain.append((t[-1] - t)/1e6)
    return [item[0] for item in sfr_chain]


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
    raw_masses = res['chain'][idx,0]
    pred_masses = 10**raw_masses * mass_frac
    mass_quantiles = quantile(np.log10(pred_masses), [.16, .5, .84], weights=res['weights'][idx])
    mass_quan.append(mass_quantiles)
    sfr_chain = get_sfh_psb(res, mod)
    sfr_quan.append(quantile(sfr_chain, [.16, .5, .84], weights=res['weights'][idx]))
    metal_quan.append(quantile(res['chain'][idx,1], [0.16, 0.5, 0.84], weights=res['weights'][idx]))
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


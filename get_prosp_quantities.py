from corner import quantile
import prospect.io.read_results as pread
from prospect.models import transforms
import numpy as np
import sys
import glob, os
import pandas as pd
from tqdm.auto import tqdm


#------------------------------------------------- 

def get_best(res, **kwargs):
    imax = np.argmax(res['lnprobability'])
    theta_best = res['chain'][imax, :].copy()
    return theta_best
def nonpara_massweighted_age_old(sfr_in_each_bin, time):
    top = 0.0
    bottom = 0.0
    for bin_ in range(len(sfr_in_each_bin)):
        top +=  time[bin_] * sfr_in_each_bin[bin_]
    return top / np.sum(sfr_in_each_bin)
def find_nearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx
def get_sfr100(res, mod):
    agebins = mod.params['agebins']
    thetas = mod.theta_labels()
    #print(thetas)
    agebins_yrs = 10**agebins.T
    dt = agebins_yrs[1, :] - agebins_yrs[0, :]
        
    zfrac_idx = [i for i, s in enumerate(thetas) if 'z_fraction' in s]
    zfrac_chain = res['chain'][:,zfrac_idx[0]:zfrac_idx[-1]+1]
    try:
        total_mass_chain = res['chain'][:,thetas.index('massmet_1')]
    except:
        total_mass_chain = res['chain'][:,thetas.index('logmass')]
    sfr_chain = []
    for i in range(len(zfrac_chain)):
        masses_chain = transforms.zfrac_to_masses(10**total_mass_chain[i], zfrac_chain[i], agebins)
        sfr = masses_chain / dt
        #sfrout = np.zeros_like(t)
        #sfrout[::2] = sfr
        #sfrout[1::2] = sfr
        sfr_chain.append(sfr[0])
    #print(dt)
    return sfr_chain
def get_bestfit_sfr100(best, mod):
    agebins = mod.params['agebins']
    thetas = mod.theta_labels()
    #best = 
    
    zfrac_idx = [i for i, s in enumerate(thetas) if 'z_fraction' in s]
    zfrac_best = best[zfrac_idx[0]:zfrac_idx[-1]+1]
    try:
        mass_best = best[thetas.index('massmet_1')]
    except:
        mass_best = best[thetas.index('logmass')]
    agebins_yrs = 10**agebins.T
    dt = agebins_yrs[1, :] - agebins_yrs[0, :]
    
    masses_fracs = transforms.zfrac_to_masses(10**mass_best, zfrac_best, agebins)
    sfr = masses_fracs / dt
    sfrout = sfr
    return sfrout


def get_bestfit_sfr100_cont(best, mod):
    agebins = mod.params['agebins']
    thetas = mod.theta_labels()
    #best =                                                                                                                                                                       

    logsfr_ratios_idx = [i for i, s in enumerate(thetas) if 'logsfr_ratios' in s]
    logsfr_ratios_best = best[logsfr_ratios_idx[0]:logsfr_ratios_idx[-1]+1]
    try:
        mass_best = best[thetas.index('massmet_1')]
    except:
        mass_best = best[thetas.index('logmass')]
    agebins_yrs = 10**agebins.T
    dt = agebins_yrs[1, :] - agebins_yrs[0, :]

    masses_fracs = transforms.logsfr_ratios_to_masses(mass_best, logsfr_ratios_best, agebins)
    sfr = masses_fracs / dt
    sfrout = np.log10(sfr[0])
    return sfrout


def get_sfr100_cont(res, mod):
    agebins = mod.params['agebins']
    thetas = mod.theta_labels()
    agebins_yrs = 10**agebins.T
    dt = agebins_yrs[1, :] - agebins_yrs[0, :]

    zfrac_idx = [i for i, s in enumerate(thetas) if 'logsfr_ratios' in s]
    zfrac_chain = res['chain'][:,zfrac_idx[0]:zfrac_idx[-1]+1]
    try:
        total_mass_chain = res['chain'][:,thetas.index('massmet_1')]
    except:
        total_mass_chain = res['chain'][:,thetas.index('logmass')]
    sfr_chain = []
    for i in range(len(zfrac_chain)):
        masses_chain = transforms.logsfr_ratios_to_masses(total_mass_chain[i], zfrac_chain[i], agebins)
        sfr = masses_chain / dt
        sfr_chain.append(sfr[0])
    return sfr_chain



#-------------------------------------------------



galaxy = sys.argv[1]
#galaxy = np.load('/orange/narayanan/s.lower/prospector/simba_runs/simba_galaxy_SFRcut.npz')['ID'][int(galaxy_idx)]
prosp_dir = sys.argv[2]

model = 'dirichlet'
#prosp_dir = '/orange/narayanan/s.lower/prospector/attenuation_tests/fiducial_models/'+model+'/calzetti/'


print('importing model and sps')
sys.path.append(prosp_dir)
from run_prosp import build_model, build_sps



sfr_50 = [] 
sfr_16 = []
sfr_84 = []

print('now reading files')

infile = prosp_dir+'/galaxy_'+str(int(galaxy))+'.h5'

for prosp_output in [infile]:
    print(prosp_output)
    res, obs, mod = pread.results_from(prosp_output)

print('building sps and model')

sps = build_sps()
mod = build_model(redshift=5.024386)
thetas = mod.theta_labels()


weights = res['weights']


thetas_50 = []
thetas_16 = []
thetas_84 = []
print('quantiles for all thetas')
for theta in thetas:
    idx = thetas.index(theta)
    chain = [item[idx] for item in res['chain']]
    quan = quantile(chain, [.16, .5, .84], weights)
    thetas_50.append(quan[1])
    thetas_16.append(quan[0])
    thetas_84.append(quan[2])




#mod_50 = mod.mean_model(thetas_50, obs, sps)
#spec = mod_50[0]
print('median quantities')
print('    mass and Z')

try:
    mass = thetas_50[thetas.index('massmet_1')]
    mass_50 = thetas_50[thetas.index('massmet_1')]
    mass_16 = thetas_16[thetas.index('massmet_1')]
    mass_84 = thetas_84[thetas.index('massmet_1')]
    Z_50 = thetas_50[thetas.index('massmet_2')]
    Z_16 = thetas_16[thetas.index('massmet_2')]
    Z_84 = thetas_84[thetas.index('massmet_2')]

except:
    mass = thetas_50[thetas.index('logmass')]
    mass_50 = thetas_50[thetas.index('logmass')]
    mass_16 = thetas_16[thetas.index('logmass')]
    mass_84 = thetas_84[thetas.index('logmass')]
    Z_50 = thetas_50[thetas.index('logzsol')]
    Z_16 = thetas_16[thetas.index('logzsol')]
    Z_84 = thetas_84[thetas.index('logzsol')]

if model == 'dirichlet':
    print('    dirichlet sfr')



    sfr_chain = get_sfr100(res, mod)


    sfr_quan = []
    sfr_quan = quantile(sfr_chain, [0.16, 0.5, 0.84])
    sfr_50 = sfr_quan[1]
    sfr_16 = sfr_quan[0]
    sfr_84 = sfr_quan[2]
    
if model == 'continuity':
    print('    continuity sfr')

    sfr_chain = get_sfr100_cont(res, mod)


    sfr_quan = []
    sfr_quan = quantile(sfr_chain, [0.16, 0.5, 0.84])
    sfr_50 = sfr_quan[1]
    sfr_16 = sfr_quan[0]
    sfr_84 = sfr_quan[2]



print('maximum liklihood thetas')
theta_max = get_best(res)

sfr_best = get_bestfit_sfr100(theta_max, mod)

#does run include free dust emission?
run_name = prosp_dir
"""if 'IR' in run_name:
    print('this run has no FIR emission, will infer Mdust from default duste params')
    print('    dust mass')
    try:
        total_mass = 10**theta_max[thetas.index('logmass')]
    except:
        total_mass = 10**theta_max[thetas.index('massmet_1')]
    time_bins_log = next(item for item in res['model_params'] if item["name"] == "agebins")['init']
    z_frac = theta_max[thetas.index('z_fraction_1'):thetas.index('z_fraction_5')+1]
    masses = transforms.zfrac_to_masses(total_mass, z_frac, time_bins_log)
    converted_sfh = sps.convert_sfh(time_bins_log, masses)
    model_sp = sps.ssp
    model_sp.params['sfh'] = 3
    model_sp.params['imf_type'] = 2
    model_sp.params['zred'] = 2.0
    model_sp.set_tabular_sfh(converted_sfh[0], converted_sfh[1])
    model_sp.params['add_dust_emission'] = True
    model_sp.params['dust_type'] = 5
    try:
        model_sp.params['dust2'] = theta_max[thetas.index('dust2')]#[thetas.index('av_delta_1')]                                                             
        model_sp.params['dust_index'] = theta_max[thetas.index('dust_index')]#[thetas.index('av_delta_2')]                                                   
    except:
        model_sp.params['dust2'] = theta_max[thetas.index('av_delta_1')]
        model_sp.params['dust_index'] = theta_max[thetas.index('av_delta_2')]
    try:
        model_sp.params['logzsol'] = theta_max[thetas.index('massmet_2')]
    except:
        model_sp.params['logzsol'] = theta_max[thetas.index('logzsol')]
    model_sp.params['duste_gamma'] = 0.01
    model_sp.params['duste_umin'] = 1.0
    model_sp.params['duste_qpah'] = 5.86
    dust_mass = model_sp.dust_mass
else:

    print('    dust mass')
    try:
        total_mass = 10**theta_max[thetas.index('massmet_1')]
    except:
        total_mass = 10**theta_max[thetas.index('logmass')]
    time_bins_log = next(item for item in res['model_params'] if item["name"] == "agebins")['init']
    z_frac = theta_max[thetas.index('z_fraction_1'):thetas.index('z_fraction_5')+1]
    masses = transforms.zfrac_to_masses(total_mass, z_frac, time_bins_log)
    converted_sfh = sps.convert_sfh(time_bins_log, masses)
    model_sp = sps.ssp
    model_sp.params['sfh'] = 3
    #model_sp.params['zcontinuous'] = 1
    model_sp.params['imf_type'] = 2
    model_sp.params['zred'] = 2.0
    model_sp.set_tabular_sfh(converted_sfh[0], converted_sfh[1])
    model_sp.params['add_dust_emission'] = True
    model_sp.params['dust_type'] = 5
    if 'dust2' in mod.fixed_params:
        model_sp.params['dust2'] = next(item for item in res['model_params'] if item["name"] == "dust2")['init']
    else:
        try:
            model_sp.params['dust2'] = theta_max[thetas.index('dust2')]#[thetas.index('av_delta_1')]
            model_sp.params['dust_index'] = theta_max[thetas.index('dust_index')]#[thetas.index('av_delta_2')]
        except: 
            model_sp.params['dust2'] = theta_max[thetas.index('av_delta_1')]
            model_sp.params['dust_index'] = theta_max[thetas.index('av_delta_2')]
    
    try:
        model_sp.params['logzsol'] = theta_max[thetas.index('massmet_2')]
    except:
        model_sp.params['logzsol'] = theta_max[thetas.index('logzsol')]
    model_sp.params['duste_gamma'] = theta_max[thetas.index('duste_gamma')]
    model_sp.params['duste_umin'] = theta_max[thetas.index('duste_umin')]
    model_sp.params['duste_qpah'] = 5.86
    dust_mass = model_sp.dust_mass"""



print('done. saving')
data = {'Galaxy' : galaxy, 'Mass_50' : mass_50, 'SFR_50': sfr_50, 
        'SFR_84': sfr_84, 'SFR_16': sfr_16, 'SFR_best': sfr_best,
        'Mass_16' : mass_16, 'Mass_84' : mass_84, 'Z_50' : Z_50, 'Z_16' : Z_16, 'max_liklihood_thetas': theta_max, 
        'Z_84' : Z_84, 'theta_labels': thetas}#, 'dust_mass': dust_mass}


#data = {'Galaxy' : galaxy, 'Mass_50' : mass_50, 
#        'Mass_16' : mass_16, 'Mass_84' : mass_84, 'Z_50' : Z_50, 'Z_16' : Z_16, 'max_liklihood_thetas': theta_max,
#        'Z_84' : Z_84, 'Spec': spec, 'theta_labels': thetas, 'dust_mass': dust_mass}  


np.savez(prosp_dir+'/galaxy_'+str(galaxy)+'_prosp_quan.npz', data=data)

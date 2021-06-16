import numpy as np
from sedpy.observate import load_filters
import h5py
import prospect.io.read_results as pread
from prospect.models import priors, transforms
from scipy.stats import truncnorm
from prospect.io import write_results as writer
from prospect.fitting import fit_model
import sys, os





#------------------------
# Convienence Functions
#------------------------

def get_best(res, **kwargs):
    imax = np.argmax(res['lnprobability'])
    theta_best = res['chain'][imax, :].copy()

    return theta_best
def find_nearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx


def zfrac_to_masses_log(logmass=None, z_fraction=None, agebins=None, **extras):
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    mass_fraction = sfr_fraction * np.array(time_per_bin)
    mass_fraction /= mass_fraction.sum()

    masses = 10**logmass * mass_fraction
    return masses

#----------------------
# SSP and noise functions
#-----------------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import FastStepBasis
    sps = FastStepBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps



#--------------------
# Model Setup
#--------------------


priors.Uniform = priors.TopHat

model_params = []

#basics                                                                            
model_params.append({'name': "lumdist", "N": 1, "isfree": False,"init": 1.0e-5,"units": "Mpc"})
model_params.append({'name': 'pmetals', 'N': 1,'isfree': False,'init': -99,'prior': None})
model_params.append({'name': 'imf_type', 'N': 1,'isfree': False,'init': 2})
#M-Z
model_params.append({'name': 'logmass', 'N': 1,'isfree': True,'init': 10.0,'prior': priors.TopHat(mini=7, maxi=12)})
model_params.append({'name': 'logzsol', 'N': 1,'isfree': True,'init': -0.5,'prior': priors.TopHat(mini=-1.6, maxi=0.2)})
#SFH
model_params.append({'name': "sfh", "N": 1, "isfree": False, "init": 3})
model_params.append({'name': "mass", 'N': 3, 'isfree': False, 'init': 1., 'depends_on':zfrac_to_masses_log})
model_params.append({'name': "agebins", 'N': 1, 'isfree': False,'init': []})
model_params.append({'name': "z_fraction", "N": 2, 'isfree': True, 'init': [0, 0],'prior': priors.Beta(alpha=1.0, beta=1.0, mini=0.0, maxi=1.0)})
#Dust attenuation                                                                                                                                            
model_params.append({'name': 'dust_type', 'N': 1,'isfree': False,'init': 5,'prior': None})
model_params.append({'name': 'dust2', 'N': 1,'isfree': True, 'init': 0.1,'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=0.0, sigma=0.3)})
model_params.append({'name': 'dust_index', 'N': 1,'isfree': True,'init': -0.5, 'prior': priors.Uniform(mini=-1.8, maxi=0.3)})
#Dust Emission                                                                                                                                             
model_params.append({'name': 'add_dust_emission', 'N': 1,'isfree': False,'init': 1})
model_params.append({'name': 'duste_gamma', 'N': 1,'isfree': True,'init': 0.01,'prior': priors.Uniform(mini=0.0, maxi=1.0)})
model_params.append({'name': 'duste_umin', 'N': 1,'isfree': True,'init': 1.0,'prior': priors.Uniform(mini=0.1, maxi=20.0)})
model_params.append({'name': 'duste_qpah', 'N': 1,'isfree': False,'init': 5.86,'prior': priors.Uniform(mini=0.0, maxi=7.0)})

#Misc
model_params.append({'name': 'add_agb_dust_model', 'N': 1,'isfree': False,'init': 0})



#-------------------
# Build Model
#-------------------

def build_model(**kwargs):
    from prospect.models import priors, sedmodel
    print('building model')

    #editing our SFH model to have 6 components 
    n = [p['name'] for p in model_params]
    tuniv = 14.0
    nbins=6
    tbinmax = (tuniv * 0.85) * 1e9
    lim1, lim2 = 8.0, 8.52 #100 Myr and 330 Myr                                                                                                             
    #setting the edges of the SFH time bins
    agelims = [0,lim1] + np.linspace(lim2,np.log10(tbinmax),nbins-2).tolist() + [np.log10(tuniv*1e9)]
    agebins = np.array([agelims[:-1], agelims[1:]])
    ncomp = nbins
    alpha_sfh = 0.7  # desired Dirichlet concentration                                                                                                       
    #now just re-adjusting the priors, etc. to reflect the fact that we have 6 SFH components 
    alpha = np.repeat(alpha_sfh,nbins-1)
    tilde_alpha = np.array([alpha[i-1:].sum() for i in range(1,ncomp)])
    zinit = np.array([(i-1)/float(i) for i in range(ncomp, 1, -1)])
    zprior = priors.Beta(alpha=tilde_alpha, beta=np.ones_like(alpha), mini=0.0, maxi=1.0)

    model_params[n.index('mass')]['N'] = ncomp
    model_params[n.index('agebins')]['N'] = ncomp
    model_params[n.index('agebins')]['init'] = agebins.T
    model_params[n.index('z_fraction')]['N'] = len(zinit)
    model_params[n.index('z_fraction')]['init'] = zinit
    model_params[n.index('z_fraction')]['prior'] = zprior

    
    model = sedmodel.SedModel(model_params)


    return model

#---------------------
# Setup Observations
#---------------------

galex = ['galex_FUV', 'galex_NUV']
hst_wfc3_uv  = ['wfc3_uvis_f275w', 'wfc3_uvis_f336w', 'wfc3_uvis_f475w','wfc3_uvis_f555w', 'wfc3_uvis_f606w', 'wfc3_uvis_f814w']
hst_wfc3_ir = ['wfc3_ir_f105w', 'wfc3_ir_f125w', 'wfc3_ir_f140w', 'wfc3_ir_f160w']
spitzer_mips = ['spitzer_mips_24']
herschel_pacs = ['herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160']
herschel_spire = ['herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500']
filternames = (galex + hst_wfc3_uv +  hst_wfc3_ir + spitzer_mips + herschel_pacs + herschel_spire)

#------------------
# Build Observations
#-------------------


def build_obs(galaxy,**kwargs):
    print('loading obs')
    from hyperion.model import ModelOutput
    from astropy import units as u
    from astropy import constants
    #here, I am taking the powderday RT SEDs and extracting our photometry
    pd_dir = '/orange/narayanan/s.lower/simba/pd_runs/snap305/snap305.galaxy'+str(galaxy)+'.rtout.sed'
    m = ModelOutput(pd_dir)
    wav,flux = m.get_sed(inclination=0,aperture=-1)
    wav  = np.asarray(wav)*u.micron #wav is in micron                                                  
    wav = wav.to(u.AA)
    flux = np.asarray(flux)*u.erg/u.s
    dl = (10. * u.pc).to(u.cm)
    flux /= (4.*3.14*dl**2.)
    nu = constants.c.cgs/(wav.to(u.cm))
    nu = nu.to(u.Hz)
    flux /= nu
    flux = flux.to(u.Jy)
    maggies = flux / 3631.

    filters_unsorted = load_filters(filternames)
    waves_unsorted = [x.wave_mean for x in filters_unsorted]
    filters = [x for _,x in sorted(zip(waves_unsorted,filters_unsorted))]
    flx = []
    flxe = []
    for i in range(len(filters)):
        flux_range = []
        wav_range = []
        for j in filters[i].wavelength:
            flux_range.append(maggies[find_nearest(wav.value,j)].value)
            wav_range.append(wav[find_nearest(wav.value,j)].value)
        a = np.trapz(wav_range * filters[i].transmission* flux_range, wav_range, axis=-1)
        b = np.trapz(wav_range * filters[i].transmission, wav_range)
        flx.append(a/b)
        flxe.append(0.03* flx[i])
    flx = np.asarray(flx)
    flxe = np.asarray(flxe)
    flux_mag = flx
    unc_mag = flxe

    obs = {}
    obs['filters'] = filters
    obs['maggies'] = flux_mag
    obs['maggies_unc'] = unc_mag
    obs['phot_mask'] = np.isfinite(flux_mag)
    obs['wavelength'] = None
    obs['spectrum'] = None

    return obs



#-------------------
# Put it all together
#-------------------


def build_all(galaxy, **kwargs):

    return (build_obs(galaxy, **kwargs), build_model(**kwargs),
            build_sps(**kwargs))


run_params = {'verbose':False,
              'debug':False,
              'output_pickles': False,
              'nested_bound': 'multi', # bounding method                                                                                                     
              'nested_sample': 'auto', # sampling method                                                                                                     
              'nested_nlive_init': 400,
              'nested_nlive_batch': 200,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              }


#------------------- 
# Run It
#-------------------

if __name__ == '__main__':

    galaxy_idx = sys.argv[1]
    print('Fitting galaxy ',str(galaxy))
    obs, model, sps, noise = build_all(galaxy=galaxy, **run_params)
    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__
    hfile = f'galaxy_{galaxy_idx}.h5'
    print('Running fits')
    output = fit_model(obs, model, sps, noise, **run_params)
    print('Done. Writing now')
    writer.write_hdf5(hfile, run_params, model, obs,
              output["sampling"][0], output["optimization"][0],
              tsample=output["sampling"][1],
              toptimize=output["optimization"][1])



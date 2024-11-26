import numpy as np
import pandas as pd
from sedpy.observate import load_filters
import h5py
import prospect.io.read_results as pread
from prospect.models import priors, transforms
from prospect.models import priors_beta
from scipy.stats import truncnorm
from prospect.io import write_results as writer
from prospect.fitting import fit_model
import sys, os

#------------------------
# Convienence Functions
#------------------------
def find_nearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx

#----------------------
# SSP and noise functions
#-----------------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import FastStepBasis
    sps = FastStepBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps


def build_noise(**extras):
    return None, None

#-------------------
# Build Model
#-------------------

def build_model(**kwargs):
    from prospect.models import priors, sedmodel
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    print('building model')
    priors.Uniform = priors.TopHat

    model_params = []

    #basics
    model_params.append({'name': "zred", "N": 1, "isfree": False,"init": 7.2,"depends_on":transforms.nzsfh_to_zred})
    model_params.append({'name': 'pmetals', 'N': 1,'isfree': False,'init': -99,'prior': None})
    model_params.append({'name': 'imf_type', 'N': 1,'isfree': False,'init': 2})
    #prospector_beta
    model_params.append({"name":'nzsfh','N': 9, 'isfree': True, 'init': np.array([7.2,8,0.0, 0,0,0,0,0,0]),
                         'prior': priors_beta.NzSFH(zred_mini=7.1, zred_maxi=7.3,
                                                    mass_mini=7.0, mass_maxi=12.5,
                                                    z_mini=-1.98, z_maxi=0.19,
                                                    logsfr_ratio_mini=-5.0, logsfr_ratio_maxi=5.0,
                                                    const_phi=True)})
    #M-Z
    model_params.append({'name': 'logmass', 'N': 1,'isfree': False,'init': 10.0,'depends_on': transforms.nzsfh_to_logmass})
    model_params.append({'name': 'logzsol', 'N': 1,'isfree': False,'init': -0.5,'depends_on': transforms.nzsfh_to_logzsol})
    #SFH
    model_params.append({'name': "sfh", "N": 1, "isfree": False, "init": 3})
    model_params.append({'name': "mass", 'N': 3, 'isfree': False, 'init': 1., 'depends_on':transforms.logsfr_ratios_to_masses})
    model_params.append({"name": "agebins", 'N': 7, 'isfree': False, 'init': transforms.zred_to_agebins_pbeta(np.atleast_1d(0.5)),
                           'depends_on': transforms.zred_to_agebins_pbeta})
    model_params.append({'name': 'logsfr_ratios', 'N': 6, 'isfree': False, 'init': 0.0,
                                 'depends_on': transforms.nzsfh_to_logsfr_ratios})
    #Dust attenuation   
    model_params.append({'name': 'dust_type', 'N': 1,'isfree': False,'init': 0,'prior': None})
    model_params.append({'name': 'dust1', 'N': 1,'isfree': True, 'init': 1.0,'prior': priors.Uniform(mini=0., maxi=2.0)})
    model_params.append({'name': 'dust2', 'N': 1,'isfree': True, 'init': 1.0,'prior': priors.Uniform(mini=0.0, maxi=3.0)})
    model_params.append({'name': 'dust_index', 'N': 1,'isfree': True,'init': -0.9, 'prior': priors.Uniform(mini=-1.2, maxi=0.3)})
    #model_params.append({'name': 'frac_nodust', 'N': 1,'isfree': True, 'init': 0.1,'prior': priors.Uniform(mini=0.0, maxi=1.0)})
    #Dust Emission      
    model_params.append({'name': 'add_dust_emission', 'N': 1,'isfree': False,'init': 1})
    model_params.append({'name': 'duste_gamma', 'N': 1,'isfree': False,'init': 0.01,'prior': priors.Uniform(mini=0.0, maxi=1.0)})
    model_params.append({'name': 'duste_umin', 'N': 1,'isfree': False,'init': 10.0,'prior': priors.Uniform(mini=0.1, maxi=30.0)})
    model_params.append({'name': 'duste_qpah', 'N': 1,'isfree': False,'init': 1.,'prior': priors.Uniform(mini=0.0, maxi=10.0)})
    #Misc               
    model_params.append({'name': 'add_agb_dust_model', 'N': 1,'isfree': False,'init': 0})
    
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
#F115W, F150W, F200W, F277W, F356W, F444W, F560W, F770W, F1000W, F1280W, F1500W, F1800W, F2100W
jwst_nircam = ['jwst_f115w', 'jwst_f150w', 'jwst_f200w', 'jwst_f277w', 'jwst_f356w', 'jwst_f444w']
jwst_miri = ['jwst_f560w', 'jwst_f770w', 'jwst_f1000w', 'jwst_f1280w', 'jwst_f1500w', 'jwst_f1800w', 'jwst_f2100w']


filternames = jwst_nircam

#------------------
# Build Observations
#-------------------


def build_obs(pd_dir,**kwargs):
    print('loading obs')
    import sedpy
    from astropy import units as u
    from astropy import constants
    from astropy.cosmology import FlatLambdaCDM    
    from hyperion.model import ModelOutput
    cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)
    m = ModelOutput(pd_dir)
    wav,flux = m.get_sed(inclination=0,aperture=-1)
    wav  = np.asarray(wav)*u.micron #wav is in micron                                                                                                                             
    wav = wav.to(u.AA)
    flux = np.asarray(flux)*u.erg/u.s
    dl = cosmo.luminosity_distance(7.2).to('cm')
    flux /= (4.*3.14*dl**2.)
    flux *= (1+7.2)
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
    redshifted_wav = wav*(1.+7.2)
    for i in range(len(filters)):
        flux_range = []
        wav_range = []
        for j in filters[i].wavelength:
            flux_range.append(maggies[find_nearest(redshifted_wav.value,j)].value)
            wav_range.append(redshifted_wav[find_nearest(redshifted_wav.value,j)].value)
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
    obs['pd_sed'] = maggies
    obs['pd_wav'] = redshifted_wav

    return obs



#-------------------
# Put it all together
#-------------------


def build_all(pd_dir,**kwargs):

    return (build_obs(pd_dir,**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))




run_params = {'verbose':False,
              'debug':False,
              'output_pickles': True,
              'nested_bound': 'multi', # bounding method                                                                                      
              'nested_sample': 'auto', # sampling method                                                                                      
              'nested_nlive_init': 400,
              'nested_nlive_batch': 200,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              }



if __name__ == '__main__':

    parent_dir = "/orange/narayanan/desika.narayanan/pd_runs/simba/ultrabright_highz/m25n512/snap059_cf/"
    galaxy = int(sys.argv[1])
    pds = f'/snap059.galaxy{galaxy}.rtout.sed'
    pd_dir = parent_dir+pds
    obs, model, sps, noise = build_all(pd_dir,**run_params)
    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__
    hfile = f'/orange/narayanan/s.lower/prospector/early_massive_jwst_galaxies_labbe/prospector_beta/simba/nircam/CF_galaxy{galaxy}.h5'
    print('Running fits')
    output = fit_model(obs, model, sps, noise, **run_params)
    print('Done. Writing now')
    writer.write_hdf5(hfile, run_params, model, obs,
              output["sampling"][0], output["optimization"][0],
              tsample=output["sampling"][1],
              toptimize=output["optimization"][1])



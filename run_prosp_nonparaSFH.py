import numpy as np
from prospect.io import write_results as writer
from prospect.fitting import fit_model
import sys, os

#------------------------
# Convienence Functions
#------------------------

def find_nearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx

def zfrac_to_masses_log(logmass=None, z_fraction=None, agebins=None, **extras):
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])
    # convert to mass fractions
    time_per_bin = np.diff(10**agebins, axis=-1)[:, 0]
    mass_fraction = sfr_fraction * np.array(time_per_bin)
    mass_fraction /= mass_fraction.sum()

    if (mass_fraction < 0).any():
        idx = mass_fraction < 0
        if np.isclose(mass_fraction[idx],0,rtol=1e-8):
            mass_fraction[idx] = 0.0
        else:
            raise ValueError('The input z_fractions are returning negative masses!')

    masses = 10**logmass * mass_fraction
    return masses

#----------------------
# SSP function
#-----------------------

def build_sps(**kwargs):
    """
    This is our stellar population model which generates the spectra for stars of a given age and mass. 
    Because we are using a non parametric SFH model, we do have to use a different SPS model than before 
    """
    from prospect.sources import FastStepBasis
    sps = FastStepBasis(zcontinuous=1)
    return sps


#--------------------
# Model Setup
#--------------------

def build_model(**kwargs):
    from prospect.models import priors, sedmodel
    print('building model')
    model_params = []
    #basics
    model_params.append({'name': "lumdist", "N": 1, "isfree": False,"init": 1e-5,"units": "Mpc"})
    model_params.append({'name': 'imf_type', 'N': 1,'isfree': False,'init': 2})
    model_params.append({'name': 'dust_type', 'N': 1,'isfree': False,'init': 2,'prior': None})
    model_params.append({'name': 'dust2', 'N': 1,'isfree': True, 'init': 0.1,'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=0.0, sigma=0.3)})
    model_params.append({'name': 'add_dust_emission', 'N': 1,'isfree': False,'init': 1,'prior': None})
    model_params.append({'name': 'duste_gamma', 'N': 1,'isfree': True,'init': 0.01,'prior': priors.TopHat(mini=0.0, maxi=1.0)})
    model_params.append({'name': 'duste_umin', 'N': 1,'isfree': True,'init': 1.0,'prior': priors.TopHat(mini=0.1, maxi=20.0)})
    model_params.append({'name': 'duste_qpah', 'N': 1,'isfree': True,'init': 3.0,'prior': priors.TopHat(mini=0.0, maxi=6.0)})                                                          
    model_params.append({'name': 'add_agb_dust_model', 'N': 1,'isfree': False,'init': 0})
    
    #M-Z
    model_params.append({'name': 'logmass', 'N': 1,'isfree': True,'init': 10.0,'prior': priors.Uniform(mini=9., maxi=12.)})
    model_params.append({'name': 'logzsol', 'N': 1,'isfree': True,'init': -0.5,'prior': priors.Uniform(mini=-1.5, maxi=0.2)})
    

    #SFH 
    #here, we tell fsps (via Prospector) that we will be using a special SFH (so init=3, which corresponds to a
    #'custom' SFH). Of note is that the "mass" parameter no long refers to the total stellar mass. Instead,
    #this is related to the stellar mass formed in each piece-wise time bin. However, the model doesn't actually
    #sample the mass posteriors. Instead, it uses a proxy variable "z_fraction" that is related to the choice of
    #prior (Dirichlet distribution). If you want to learn more, I'd highly recommend reading Joel Leja's 2019
    #paper introducing the Prospector non parametric SFH models
    model_params.append({'name': "sfh", "N": 1, "isfree": False, "init": 3})
    #Now, mass refers to the stellar mass formed *in each time bin* while the logmass parameter above 
    #sets the overall normalization 
    model_params.append({'name': "mass", 'N': 3, 'isfree': False, 'init': 1., 'depends_on':zfrac_to_masses_log})
    #agebins are the limits for each piece-wise bin of star formation. these are set below
    model_params.append({'name': "agebins", 'N': 1, 'isfree': False,'init': []})
    #proxy parameter for SFR in each age bin
    model_params.append({'name': "z_fraction", "N": 2, 'isfree': True, 'init': [0, 0],'prior': priors.Beta(alpha=1.0, beta=1.0, mini=0.0, maxi=1.0)})                                                                                                                                                                                                                           

    #here we set the number and location of the timebins, and edit the other SFH parameters to match in size
    n = [p['name'] for p in model_params]
    tuniv = 14. #Gyr, age at z=0                                                                                                                                                                                                         
    nbins=10
    tbinmax = (tuniv * 0.8) * 1e9 #earliest time bin goes from age = 0 to age = 2.8 Gyr
    lim1, lim2 = 7.47, 8.0 #most recent time bins at 30 Myr and 100 Myr ago                                                                                                                                                                                                 
    agelims = [0,lim1] + np.linspace(lim2,np.log10(tbinmax),nbins-2).tolist() + [np.log10(tuniv*1e9)]
    agebins = np.array([agelims[:-1], agelims[1:]])

    zinit = np.array([(i-1)/float(i) for i in range(nbins, 1, -1)])
    # Set up the prior in `z` variables that corresponds to a dirichlet in sfr
    # fraction. 
    alpha = np.arange(nbins-1, 0, -1)
    zprior = priors.Beta(alpha=alpha, beta=np.ones_like(alpha), mini=0.0, maxi=1.0)

    model_params[n.index('mass')]['N'] = nbins
    model_params[n.index('agebins')]['N'] = nbins
    model_params[n.index('agebins')]['init'] = agebins.T
    model_params[n.index('z_fraction')]['N'] = nbins-1
    model_params[n.index('z_fraction')]['init'] = zinit
    model_params[n.index('z_fraction')]['prior'] = zprior

    model = sedmodel.SedModel(model_params)
    

    return model


#------------------
# Build Observations
#-------------------

def build_obs(pd_dir, **kwargs):
    
    from sedpy.observate import load_filters
    from astropy import units as u
    from astropy import constants
    from astropy.cosmology import FlatLambdaCDM
    from hyperion.model import ModelOutput
    
    cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)
    m = ModelOutput(pd_dir)
    wav, lum = m.get_sed(inclination=0,aperture=-1)
    wav  = np.asarray(wav)*u.micron                                                                                                                                    
    wav = wav.to(u.AA)
    lum = np.asarray(lum)*u.erg/u.s
    dl = (10*u.pc).to(u.cm) #setting luminosity distance to 10pc since we're at z=0
    flux = lum/(4.*3.14*dl**2.) #*(1+z) this is where you would scale the flux density by 1+z
    nu = constants.c.cgs/(wav.to(u.cm))
    nu = nu.to(u.Hz)
    flux /= nu
    flux = flux.to(u.Jy)
    maggies = flux / 3631. 
    #don't ask me why, but Prospector expects your data to be in units of maggies
    #which thankfully is proportional to Janksys
    
    #OK, above is the raw Powderday SED. But what we want to hand Prospector is fake broadband or narrow band
    #photometry (unless we want to pretend the Powderday SED is a spectrum, which is possible). So below, we 
    #will sample photometry from the Powderday SED at a few filters from different instruments
    
    # these filter names / transmission data come from sedpy
    # it's super easy to add new filters to the database but for now we'll just rely on what sedpy already has
    jwst_nircam = ['jwst_f070w', 'jwst_f090w', 'jwst_f115w', 'jwst_f150w', 'jwst_f200w', 
                   'jwst_f277w', 'jwst_f356w', 'jwst_f444w']
    herschel_pacs = ['herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160']
    herschel_spire = ['herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500']
    filternames = (jwst_nircam + herschel_pacs + herschel_spire)
    
    filters_unsorted = load_filters(filternames)
    waves_unsorted = [x.wave_mean for x in filters_unsorted]
    filters = [x for _,x in sorted(zip(waves_unsorted,filters_unsorted))]
    
    flx = []
    flxe = []
    #redshifted_wav = wav*(1.+7.2) ---- if you are fitting a z>0 SED from Powderday, this is where
    #you would redshift those wavelengths. 
    for i in range(len(filters)):
        flux_range = []
        wav_range = []
        for j in filters[i].wavelength:
            flux_range.append(maggies[find_nearest(wav.value,j)].value)
            wav_range.append(wav[find_nearest(wav.value,j)].value)
        #convolving the Powderday SED with each filter transmission curve
        a = np.trapz(wav_range * filters[i].transmission * flux_range, wav_range, axis=-1)
        b = np.trapz(wav_range * filters[i].transmission, wav_range)
        flx.append(a/b)
        #and assuming a SNR of 30
        flxe.append(0.03 * flx[i])
    flux_mag = np.asarray(flx)
    unc_mag = np.asarray(flxe)
    
    obs = {}
    #put some useful things in our dictionary. Prospector exepcts to see, at the least, the filters, photmetry
    #and errors, and if available, the spectrum information. I also include the full powderday SED for easy 
    #access later
    obs['filters'] = filters
    obs['maggies'] = flux_mag
    obs['maggies_unc'] = unc_mag
    obs['phot_mask'] = np.isfinite(flux_mag)
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['pd_sed'] = maggies
    obs['pd_wav'] = wav

    return obs



#-------------------
# Put it all together
#-------------------


def build_all(pd_dir,**kwargs):

    return (build_obs(pd_dir,**kwargs), build_model(**kwargs),
            build_sps(**kwargs))


#parameters that will be passed to dynesty, the posterior sampler. typically can just ignore these / use these defaults
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



#now run it!
#Desika: "oh you kids and your pythonic coding"
if __name__ == '__main__':

    pd_dir ='snap305.galaxy100.rtout.sed'
    obs, model, sps = build_all(pd_dir,**run_params)
    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__
    hfile = "galaxy_100_nonpara_fit.h5"
    print('Running fits')
    output = fit_model(obs, model, sps, [None,None],**run_params)
    print('Done. Writing now')
    writer.write_hdf5(hfile, run_params, model, obs,
              output["sampling"][0], output["optimization"][0],
              tsample=output["sampling"][1],
              toptimize=output["optimization"][1])



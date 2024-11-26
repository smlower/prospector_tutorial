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

#----------------------
# SSP function
#-----------------------

def build_sps(**kwargs):
    """
    This is our stellar population model which generates the spectra for stars of a given age and mass. 
    Most of the time, you aren't going to need to pay attention to this. 
    """
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=1, compute_vega_mags=0)
    return sps



#--------------------
# Model Setup
#--------------------


def build_model(**kwargs):
    from prospect.models import sedmodel
    from prospect.models import priors, transforms
    """
    Function to build model components for SFH and dust. 
    The model params are defined by their name, whether they are a free parameter
    their initial value, and their prior distribution if they are variable. The model 
    params are then fed to the prospector SedModel class
    
    All parameters except 'mass' correspond to fsps model parameters, the definitions of which you can find here:
    https://dfm.io/python-fsps/current/stellarpop_api/
    
    """
    
    model_params = []
    #luminosity distance of galaxy. for a z=0 simba galaxy, i typically just set this to be 10 pc
    model_params.append({'name': "lumdist", "N": 1, "isfree": False,"init": 1.0e-5,"units": "Mpc"})
    #IMF model which will be used by the simple stellar population model
    model_params.append({'name': 'imf_type', 'N': 1,'isfree': False,'init': 2, 'prior': None})
    #stellar mass of a galaxy -- what we're interested in! So we'll set it as a free parameter
    model_params.append({'name': 'mass', 'N': 1,'isfree': True, 'init': 1e10,'prior': priors.TopHat(mini=1e8, maxi=1e12)})
    #stellar metallicity, in units of log(Z/Z_sun)
    model_params.append({'name': 'logzsol', 'N': 1,'isfree': True,'init': -0.5,'prior': priors.TopHat(mini=-1.6, maxi=0.1)})
    #SFH model. here, we are choosing the 'delayed-tau' model and has two free parameters: the age and the e-folding time
    model_params.append({'name': "sfh", "N": 1, "isfree": False, "init": 4, 'prior': None})
    #age of the galaxy
    model_params.append({'name': "tage", 'N': 1, 'isfree': True, 'init': 5., 'units': 'Gyr', 'prior': priors.TopHat(mini=0.001, maxi=13.8)})
    #e-folding time
    model_params.append({'name': "tau", 'N': 1, 'isfree': True,'init': 1., 'units': 'Gyr', 'prior': priors.LogUniform(mini=0.1, maxi=30)})
    #dust attenuation model, from Calzetti 2001
    model_params.append({'name': 'dust_type', 'N': 1,'isfree': False,'init': 2,'prior': None})
    #the attenuation (in magnitudes) in the V-band
    model_params.append({'name': 'dust2', 'N': 1,'isfree': True, 'init': 0.1,'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=0.0, sigma=0.3)})
    #dust emission model -- only 1 choice, from Draine & Li 2007
    model_params.append({'name': 'add_dust_emission', 'N': 1,'isfree': False,'init': 1,'prior': None})
    #mass fraction of warm dust
    model_params.append({'name': 'duste_gamma', 'N': 1,'isfree': True,'init': 0.01,'prior': priors.TopHat(mini=0.0, maxi=1.0)})
    #minimum radiation field
    model_params.append({'name': 'duste_umin', 'N': 1,'isfree': True,'init': 1.0,'prior': priors.TopHat(mini=0.1, maxi=20.0)})
    #mass fraction of dust in PAHs
    model_params.append({'name': 'duste_qpah', 'N': 1,'isfree': False,'init': 3.0,'prior': priors.TopHat(mini=0.0, maxi=6.0)})
    
    
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
    flux = lum/(4.*3.14*dl**2.) #*(1+z) this is where you would scale the flux by 1+z
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
    hfile = "galaxy_100_para_fit.h5"
    print('Running fits')
    output = fit_model(obs, model, sps, [None,None],**run_params)
    print('Done. Writing now')
    writer.write_hdf5(hfile, run_params, model, obs,
              output["sampling"][0], output["optimization"][0],
              tsample=output["sampling"][1],
              toptimize=output["optimization"][1])



# prospector_tutorial
Crash course tutorial on SED modeling, specifically with Prospector. 


Prospector (https://github.com/bd-j/prospector) is an SED modeling code that wraps FSPS stellar modeling and Dynesty dynamical nested Bayesian sampling. 
It enables flexible modeling of galaxy SEDs with several model choices and data handling functions. 
In this tutorial, I walk through the fundamentals of both SED modeling and Bayesian inference, with two example fits to show hands-on how to use Prospector 
and understand the output of Prospector. 


A few prereqs (also mentioned in the tutorial notebook preamble):

1. sedpy : this manages the details of our observations and allows us to interface with the properties of the photometric filters used. You can find sedpy here: https://github.com/bd-j/sedpy

2. dynesty : this is the backbone of propsector that handles the actual fitting methods. It is a form of Bayesian modeling similar to MCMC codes like 'emcee' but a bit more sophisticated. You can find dynesty here: https://github.com/joshspeagle/dynesty

3. fsps : this handles the stellar modeling for prospector. by itself, it's a super useful tool for generating stellar spectra. The core of fsps is a Fortran code but these days, the python bindings for fsps now come with its own fsps source code, meaning we no longer have to compile the Fortran code first then install the python wrapper. All we need is to clone the fortran fsps

         export SPS_HOME="/path/where/you/want/to/download/fsps"
         git clone https://github.com/cconroy20/fsps.git $SPS_HOME
And then pip install python-fsps

            python -m pip install fsps

4. Prospector : prospector is the tool that combines the above packages to model the SEDs of galaxies. You can find it here: https://github.com/bd-j/prospector It also has a pretty decent demo on how to use prospector: https://github.com/bd-j/prospector/blob/main/demo/InteractiveDemo.ipynb
For visualization purposes, we'll also want to install 2 packages: corner, which helps us plot corner plots (https://github.com/dfm/corner.py) and arviz, which allows us to do 'advanced' things with corner (conda install arviz)
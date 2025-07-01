from fermipy.gtanalysis import GTAnalysis
from astropy.io import fits
import yaml
import numpy as np
import os
import importlib.resources

with importlib.resources.path('flashcurve', 'gll_psc_v31.fit') as resource_path:
    cat_path = str(resource_path)

def setup_config(source_dir=None, data_dir = None, target_dir = None,  source_name=None, ra=0.0, dec=0.0, t_int = None, sc_dir=None, emin=None, default_config_path = None):

    print('Setting up config')
    source_name = source_name.replace("_", " ")

    if target_dir is None:
        target_dir = source_dir

    if data_dir is None:
        data_dir = source_dir

    files = os.listdir(source_dir)
    ph_files = [os.path.join(source_dir, f) for f in files if "PH" in f]
    with open(os.path.join(source_dir, "events.txt"), 'w+') as event_file:
            event_file.write("\n".join(ph_files))
    
    config_path = os.path.join(target_dir, "config.yaml")
    if default_config_path is None:
        with open(importlib.resources.path('flashcurve', 'default.yaml'), "r") as stream:
            config = yaml.safe_load(stream)
    else:
        with open(default_config_path, "r") as stream:
            config = yaml.safe_load(stream)

    config['selection']['target'] = source_name #check Martina's code for cases where source name is not known

    config['selection']['ra'] = float(ra)
    config['selection']['dec'] = float(dec)
    config['data']['evfile'] = os.path.join(source_dir, "events.txt")
    # spacecraft_dir = glob.glob(os.path.join(source_dir, "*SC*.fits"))
    if sc_dir is not None:
        spacecraft_dir = sc_dir
    else:
        spacecraft_dir = 'flashcurve/fermi_stuff/lat_spacecraft_merged.fits'
    config['data']['scfile'] = spacecraft_dir

    config['model']['catalogs'] = cat_path

    if emin is not None:
        config['selection']['emin'] = emin
    files = os.listdir(source_dir)
    if t_int is None:
        files = [f for f in files if 'PH' in f]
        tmin = []
        tmax = []
        for f in files:
            x = fits.open(os.path.join(source_dir, f))
            tmin.append(x[1].header['TSTART'])
            tmax.append(x[1].header['TSTOP'])
    
        tmin = float(np.min(tmin))
        tmax = float(np.max(tmax))
    
    else:
        tmin = t_int[0]
        tmax = t_int[1]

    config['selection']['tmin'] = tmin
    config['selection']['tmax'] = tmax
    
    config['fileio']['outdir'] = os.path.join(target_dir, 'fermi_data')
    with open(config_path, 'w+') as stream:
        config = yaml.dump(config, stream, default_flow_style=False)

    print('Config saved in ' + config_path)

    return
    

def setup_gta(config_path, source_name, delete_weak = False):

    print('Initialize analysis:')
    gta = GTAnalysis(config_path, logging={'verbosity': 3})
    print("setup analysis")
    gta.setup()
    print("optimize")
    gta.optimize()
    if delete_weak:
        print("delete weak sources")
        gta.delete_sources(minmax_ts=[-1,3],exclude=['isodiff','galdiff'])
        gta.delete_sources(minmax_npred=[-1,1],exclude=['isodiff','galdiff'])
    print(f'free source parameters near {source_name}')
    gta.free_sources(distance=5.0,pars='norm')
    gta.free_sources(minmax_ts=[10,None],pars='norm')
    gta.free_source('galdiff')
    gta.free_source('isodiff')
    gta.free_source(source_name)
    print('fit gta')
    fit1=gta.fit()
    print('save final roi')
    fixed_sources = gta.free_sources(free=False) #freezing source params
    gta.write_roi('fit1',make_plots=True)
        
    return gta


def create_lc(source_name, gta, lc_bins=None, n_bins = None, multithread=True, nthread = None, save_bins = False, target_dir = None, free_rad = 20):
    print('creating lightcurve')
    source_name = source_name.replace("_", " ")
    lc_path = os.path.join(target_dir, "lc_data")
    if "lc_data" not in os.listdir(target_dir):
        os.mkdir(lc_path)
    if lc_bins is None:
        gta.lightcurve(source_name, write_fits = False, nbins = n_bins, save_bin_data = save_bins,
                            multithread=multithread, nthread = nthread, outdir = lc_path,
                            free_radius = free_rad
                            )      
    else:    
        gta.lightcurve(source_name, write_fits = False, time_bins = list(lc_bins), save_bin_data = save_bins,
                            multithread=multithread, nthread = nthread, outdir = lc_path,
                            free_radius = free_rad
                            )

    return



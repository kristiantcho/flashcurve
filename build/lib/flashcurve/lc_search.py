import os
import sys
from flashcurve import image_tools as it
from flashcurve import data_tools as dt
import numpy as np
import glob
import shutil
from astropy.io import fits
import importlib.resources
import io

# additional tools for post processing with fermipy (optional, requires fermipy installed)
# from flashcurve import fermi_tools as ft

#Stop tflite from printing info message constantly:

# class SuppressTFLiteLogging(io.StringIO):
#     def write(self, message):
#         if "INFO: Created TensorFlow Lite XNNPACK delegate for CPU." not in message:
#             super().write(message)

# sys.stderr = SuppressTFLiteLogging()
with importlib.resources.path('flashcurve', 'gll_psc_v31.fit') as resource_path:
    cat_path = str(resource_path)
# to avoid issues with multiprocesses run under this if statement:

if __name__ == '__main__':
    
    source = '4FGL J0509.4+0542' # name of the source
    source_n = source.replace(' ', '_')

    data_dir = os.path.join('fermi_lc_tests', source_n + '_test') # define directory where all data and time bins will be saved
    source_locs = fits.open(cat_path) 
    # source_locs = fits.open('flashcurve/flashcurve/lc_stuff/catalogs/gll_psc_v31.fit') # get source RA and Dec from here (example)
    src_indx = np.where(source_locs[1].data['Source_Name']==source)[0]
    ra = float(source_locs[1].data['RAJ2000'][src_indx])
    dec = float(source_locs[1].data['DEJ2000'][src_indx])

    print(f'Working on {data_dir}')
    images_obj = it.fermimage(fermi_path = data_dir, ra=ra, dec=dec, num_threads=2, num_workers=2) # initialize fermimage object with choice of number cpus (workers) and threads

    # dt.get_ul_data(ra=ra, dec=dec, data_dir=data_dir, get_sc=True, max_angle=12,t_int = [5.8e8,6e8]) # function for downloading fermi-lat data, optional, you can also download the data manually

    fermi_df = images_obj.create_fermi_df() # make a pandas dataframe out of the data

    timebins, ts_list, _ = images_obj.create_LC_bins(save_ts=True, ts_opt = [50,75], e_check=1000, min_time=2*3600*24, p_check=1, quiet = False) 
    
    # create_LC_bins is the time bin search function, 
    # ts_opt sets the range for the optimal TS which each
    # time bin should have, other parameters like
    # min_time determine the size of the time window
    # while e_check and p_check filter which events' timestamps
    # should be used within the time windows to test the TS of the time bins
    # based off whether their energy (above) and proximity (below)
    # the set threshold (this speeds up the search)
    
    np.save(os.path.join(data_dir, source_n + '_t_bins.npy'), timebins) # time bins done!!!
    np.save(os.path.join(data_dir, source_n +'_pred_ts.npy'), ts_list)

# to create lightcurve with fermi_tools which uses fermipy

    # timebins = np.load(os.path.join(data_dir, source_n + '_t_bins.npy'))
    # sc_dir = glob.glob(os.path.join(data_dir, '*_SC*.fits'))[0]
    # ft.setup_config(source_dir=data_dir, ra=ra, dec=dec, source_name=source, sc_dir=sc_dir, emin=emin)
    # gta = ft.setup_gta(os.path.join(data_dir, 'config.yaml'), source, delete_weak = True)
    # ft.create_lc(source_name=source, gta=gta, nthread=num_cpus, lc_bins=timebins, target_dir=data_dir, save_bins=False) 
    
    # use data_tools.plot_lc(lightcurve_path\lightcurve*.npy) to plot the lightcurve!
    
    # print('Fermi lightcurve complete')
    
# delete data files: 

    # old_files = glob.glob(os.path.join(data_dir, '*PH*.fits'))
    # sc_old_files = glob.glob(os.path.join(data_dir, '*SC*.fits'))
    # if sc_old_files:     
    #     old_files.append(*sc_old_files)
    # for f in old_files:
    #     print("delete ", f)
    #     os.remove(f)
        
    # shutil.rmtree(os.path.join(data_dir, 'lc_data'))

# delete fermipy files (except for lightcurve)

    # old_lc_files = glob.glob(os.path.join(data_dir, 'fermi_data/*'))
    # for f in old_lc_files:
    #     if ('fit1' not in f) and ('4fgl' not in f):
    #         print("delete ", f)
    #         os.remove(f)

    
    
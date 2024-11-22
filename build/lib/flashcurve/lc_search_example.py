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

# additional tools for post-processing with fermipy (optional, requires fermipy installed):
# from flashcurve import fermi_tools as ft

#Stop tflite from printing info messages constantly if this occurs for you:

# class SuppressTFLiteLogging(io.StringIO):
#     def write(self, message):
#         if "INFO: Created TensorFlow Lite XNNPACK delegate for CPU." not in message:
#             super().write(message)
# sys.stderr = SuppressTFLiteLogging()

with importlib.resources.path('flashcurve', 'gll_psc_v31.fit') as resource_path:
    cat_path = str(resource_path)

## To avoid issues with multiple processes run everything under this if statement:

if __name__ == '__main__':
    
    source = '4FGL J0509.4+0542' # 4FGL catalogue name of the source of interest (this is TXS 0506+056)
    source_n = source.replace(' ', '_')

    data_dir = os.path.join('./', source_n + '_test') # define directory where all data and time bins will be saved
    source_locs = fits.open(cat_path) 
    # source_locs = fits.open('flashcurve/flashcurve/gll_psc_v31.fit') # or get source RA and Dec from here (example)
    src_indx = np.where(source_locs[1].data['Source_Name']==source)[0]
    ra = float(source_locs[1].data['RAJ2000'][src_indx])
    dec = float(source_locs[1].data['DEJ2000'][src_indx])

    print(f'Saving data in {data_dir}')

    dt.get_ul_data(ra=ra, dec=dec, data_dir=data_dir, get_sc=False, max_angle=12,t_int = [5.8e8,6e8]) # function for downloading fermi-lat fits data, optional, you can also download the data manually
    # if you plan to use the fermi_tools for post-processing then get_sc should be set to True to get the spacecraft file 

    images_obj = it.fermimage(fermi_path = data_dir, ra=ra, dec=dec, num_threads=2, num_workers=2) # initialize fermimage object with directory where fits data is stored, source location, and choice of number cpus (workers) and threads
    
    timebins, ts_list, _ = images_obj.create_LC_bins(save_ts=True, save_arr=False, ts_opt = [16,25], e_check=1000, min_time=0.5*3600*24, p_check=1, verbose=1) 
    # 'create_LC_bins' is the time bin search function, 
    # 'ts_opt' sets the range for the optimal TS, which each time bin should have. 
    # 'min_time' determines the size of the time window
    # 'e_check' (in MeV) and 'p_check' (in deg) filter which events' timestamps
    # should be used within the time windows to test the TS of the time bins
    # based on whether their energy (above) and proximity (below) the set threshold (this speeds up the search)
    # 'verbose' parameter controls progress printouts (0 = silent, 1 = partial, 2 = full printout)
    # 'save_ts' and 'save_arr' parameters determine whether to output the list of predicted TS for each time bin
    # as well as the image array used to make the prediction - if False, empty lists are returned along with the time bins
    
    np.save(os.path.join(data_dir, source_n + '_t_bins.npy'), timebins) # time bins done!!!
    np.save(os.path.join(data_dir, source_n +'_pred_ts.npy'), ts_list)

## optional: create the lightcurve with fermi_tools which uses fermipy

    # timebins = np.load(os.path.join(data_dir, source_n + '_t_bins.npy'))
    # sc_dir = glob.glob(os.path.join(data_dir, '*_SC*.fits'))[0]
    # ft.setup_config(source_dir=data_dir, ra=ra, dec=dec, source_name=source, sc_dir=sc_dir, emin=emin)
    # gta = ft.setup_gta(os.path.join(data_dir, 'config.yaml'), source, delete_weak = True)
    # ft.create_lc(source_name=source, gta=gta, nthread=num_cpus, lc_bins=timebins, target_dir=data_dir, save_bins=False) 
    
    ## use flashcurve.data_tools.plot_lc('<lightcurve path>\lightcurve*.npy') to plot the lightcurve!
    
    # print('Fermi lightcurve complete')
    
## delete data files: 

    # old_files = glob.glob(os.path.join(data_dir, '*PH*.fits'))
    # sc_old_files = glob.glob(os.path.join(data_dir, '*SC*.fits'))
    # if sc_old_files:     
    #     old_files.append(*sc_old_files)
    # for f in old_files:
    #     print("delete ", f)
    #     os.remove(f)
        
    # shutil.rmtree(os.path.join(data_dir, 'lc_data'))

## delete fermipy files (except for lightcurve)

    # old_lc_files = glob.glob(os.path.join(data_dir, 'fermi_data/*'))
    # for f in old_lc_files:
    #     if ('fit1' not in f) and ('4fgl' not in f):
    #         print("delete ", f)
    #         os.remove(f)

    
    

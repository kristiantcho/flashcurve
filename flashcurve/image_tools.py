import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
from numpy.lib import recfunctions as rfn
from astropy.io import fits
import os
from data_tools import great_circle_dis, angle_rotation
import matplotlib as mpl
import multiprocessing as mp
import time
import importlib.resources

with importlib.resources.path('flashcurve', 'gll_psc_v31.fit') as resource_path:
    cat_path = str(resource_path)
    
with importlib.resources.path('flashcurve', 'model.tflite') as resource_path:
    model_path = str(resource_path)


class fermimage:
    """
    Class which provides methods for creating Fermi dataframes, generating 2D zoom images,
    printing images with optional markers, and creating light curve bins based on specified criteria.
    
    :param fermi_path: The `fermi_path` parameter is the path to the directory containing FITS files. These FITS files are used to create a dataframe for further processing
    :param t_int: The `t_int` parameter in the `create_LC_bins` method refers to the time interval for which you want to create the light curve bins. It is a list containing two elements - the start time
     and end time of the interval for which you want to analyze the data and create the time bins
    :param allow_multiprocessing: boolean for whether multiprocessing is to be used for time bin search
    :param num_threads: specify the number of threads to be used for the tf_lite interpreter, for predicting TS per image
    :param num_workers:  number of workers (processors) to use for parallel
    processing of images in one batch within a time window 
    :param image_dir: used to specify the directory where the generated images will be saved. If you have a specific directory path in
    mind where you want the images to be saved, you can provide that path
    :param array_dir: specify the directory where arrays will be saved. This directory will be used to store the arrays generated
     during the data processing. It can be set to a specific directory path where you want the arrays to be saved
    :param image: image array for post-processing if an image has already been generated
    :param tick_skip: used to determine the interval at which ticks are displayed on the plot axes when generating the altitude-zoom images.
    It specifies how many ticks to skip between each displayed tick on the plot axes for RA and Dec differences, defaults to 7 (optional)
    """

    def __init__(self, fermi_path = None, bin_num = 56, ra = None, dec = None, model_path = model_path, max_psi = 12, 
                old_image=False, max_energy=3e5, image_dir=None, psi_square=True, array_dir=None, image = None, num_workers=3, 
                image_pos=None, tick_skip = 14, alt_max_angle = 12, allow_multiprocessing = True, num_threads=3):
        
        self.fermi_path = fermi_path
        self.bin_num = bin_num
        if (model_path is not None) and not allow_multiprocessing:
            with open(model_path, 'rb') as f:
                tflite_model = f.read()
            self.model = tflite.Interpreter(model_content=tflite_model, num_threads=num_threads)
            self.model.allocate_tensors()
        self.psi_square = psi_square
        if psi_square:
            self.max_psi = max_psi
        else:
            self.max_psi = 8.4
        self.old_image = old_image
        self.max_energy = max_energy
        self.image_dir = image_dir
        self.array_dir = array_dir
        self.source_loc = [ra, dec]
        self.image = image
        self.image_pos = image_pos
        self.tick_skip = tick_skip
        self.alt_max_angle = alt_max_angle
        self.model_path = model_path
        self.num_threads = num_threads
        self.allow_mp = allow_multiprocessing
        self.num_workers = num_workers
        return


    def create_fermi_df(self, cut_zoom = True): #create dataframe from fits files (unbinned)

        files = os.listdir(self.fermi_path)
        ph_files = [os.path.join(self.fermi_path, f) for f in files if "PH" in f]

        for i, path in enumerate(ph_files):
            if i == 0:
                fermi_data = fits.open(path)[1].data
                fermi_data = rfn.require_fields(fermi_data, [("ENERGY", "f"), ("RA", "f"), ("DEC", "f"), ("TIME", "f")])
            else:
                fd1 = fits.open(path)[1].data
                fd1 = rfn.require_fields(fd1,[("ENERGY", "f"), ("RA", "f"), ("DEC", "f"), ("TIME", "f")])
                fermi_data = np.append(fermi_data, fd1)
        
        src_lc_df = pd.DataFrame(fermi_data)
        src_lc_df['PSI'] = np.absolute(great_circle_dis(self.source_loc[0], self.source_loc[1], src_lc_df['RA'], src_lc_df['DEC']))**2
        new_ras, new_decs = angle_rotation(self.source_loc[1], self.source_loc[0], src_lc_df['DEC'], src_lc_df['RA'])
        ra_difs = np.array(new_ras).squeeze()
        
        # in case abs difference greater than 180 for RA: 
        ra_difs[ra_difs < -180] += 360
        ra_difs[ra_difs > 180] -= 360
        src_lc_df['RA_DIF'] = ra_difs
        src_lc_df['DEC_DIF'] = new_decs
        
        if cut_zoom:
            e_bins = [99]
            e_bins.extend(10**np.arange(2.5,5, 0.5))
            e_bins.append(3e5)
            max_angs = [12,5,3,1.5,1,0.6]
            temp_df = src_lc_df.copy()
            for i in range(len(max_angs)): 
                mask = (temp_df.ENERGY <= e_bins[i]) & (temp_df.ENERGY >= e_bins[i+1])
                final_mask = mask & ((np.abs(temp_df.RA_DIF) > max_angs[i]) | (np.abs(temp_df.DEC_DIF) > max_angs[i]))
                temp_df = temp_df.drop(temp_df[final_mask].index)
                temp_df = temp_df.drop(temp_df[final_mask].index)
            return temp_df
        return src_lc_df 
    
    
    def create_2d_alt_zoom_images(self, fermi_df = None, t_bins = None, source_n = 'none', t_int = None, overlap=False):
        """
        Creates 2D images by binning photon events based on RA, Dec, energy, and time
        intervals.
        
        :param fermi_df: dataframe containing Fermi data. If this parameter is not provided, the function will create a
        Fermi dataframe using the `create_fermi_df` method from the same class
        :param t_bins: time bins that can be used for grouping the data. It is a list of tuples where each tuple defines a
        time interval. Used to generate multiple images at once/
        :param source_n: name of the gamma-ray source. Used to label the data points in the final output, defaults to none (optional)
        :param t_int: specific time interval for filtering the data. If `t_int` is provided, the function will filter
        out data points outside the specified time interval before further processing. Used for generating a single image.
        :param overlap: boolean flag that determines how time bins are handled. When `overlap` is set to `False`, the function
        creates non-overlapping time bins using the specified `t_bins`. Defaults to False (optional)
        :return: returns either a pandas Series object with counts per time bin, RA bin, Dec bin, and energy bin if multiple time bins are provided or a
        numpy array representing an image of counts if a single image is being generated.
        """

        
        if fermi_df is None:
            src_lc_df = self.create_fermi_df().copy()
        else:
            src_lc_df = fermi_df.copy()

        # generate list of linearly spaced RA/Dec difference bins per energy bin (6)
        # with different maximum angular resolutions:
        
        ra_bins = np.empty((6, self.bin_num+1))
        for j, max_ang in enumerate([12,5,3,1.5,1,0.6]):
            ra_bins[j][0] = -max_ang
            for i in range(self.bin_num):
                ra_bins[j][i+1] = (round(-max_ang + (2*max_ang/self.bin_num)*(i+1), 3))

        dec_bins = ra_bins

        # labels of bins needed for counting bins later, 
        # using the labels only from the first set of RA bins, otherwise order of photon events gets mixed
        # and image comes out jumbled (but events still get binned with different angular resolutions)
        ra_bin_labels = []
        dec_bin_labels = []

        for j in range(len(ra_bins[0])-1):
            ra_bin_labels.append('RA:' + str(ra_bins[0][j])+'_'+str(ra_bins[0][j+1]))
            dec_bin_labels.append('Dec:' + str(ra_bins[0][j])+'_'+str(ra_bins[0][j+1]))

        # energy bins:
        e_bins = [99]
        e_bins.extend(10**np.arange(2.5,5, 0.5))
        e_bins.append(3e5)

        e_bin_labels = []
        
        for j in range(len(e_bins)-1):
            e_bin_labels.append(str(e_bins[j])+'_'+str(e_bins[j+1]))

        if t_bins is not None:
            t_bin_labels = [] 

            for k in range(len(t_bins)):
                t_bin_labels.append(str(round(t_bins[k][0])) + '_' + str(round(t_bins[k][1])))

            # need this to make sure pandas bins with these specific bins (because
            # there are gaps between bins!!!)
            if not overlap:    
                t_bins = pd.IntervalIndex.from_tuples(t_bins)
            else:  
                def find_time_bins(time, bins, bin_labels):
                    time_bins = []
                    for i in range(len(bins)):
                        if (float(bins[i][0]) <= float(time)) and (float(time) < float(bins[i][1])):
                            time_bins.append(bin_labels[i]) 
                    return time_bins
        # loops through energy bins:
        for i in range(6):
            temp_df = src_lc_df.copy()
            
            # For case of single images needed to be generated
            if t_int is not None:
                temp_df = temp_df.drop(temp_df[temp_df.TIME > t_int[1]].index)
                temp_df = temp_df.drop(temp_df[temp_df.TIME < t_int[0]].index)
            
            # filter out energies outside of current energy bin
            temp_df = temp_df.drop(temp_df[temp_df.ENERGY <= e_bins[i]].index)
            temp_df = temp_df.drop(temp_df[temp_df.ENERGY >= e_bins[i+1]].index)
            # bin RA/Dec:
            temp_df["RA_Bin"] = pd.cut(temp_df["RA_DIF"], ra_bins[i], labels = ra_bin_labels, right=False)
            temp_df["Dec_Bin"] = pd.cut(temp_df["DEC_DIF"], dec_bins[i], labels = dec_bin_labels, right=False)
            # bin time (if multiple bins given)
            if (t_bins is not None) and (t_int is None):
                if overlap:
                    temp_df["Time_Bin"] = temp_df["TIME"].apply(lambda x: find_time_bins(x, t_bins, t_bin_labels))
                    temp_df = temp_df.explode('Time_Bin').dropna(subset=['Time_Bin'])
                else:
                    temp_df["Time_Bin"] = pd.cut(temp_df["TIME"], t_bins, labels = t_bin_labels, right=False)
            temp_df = temp_df.dropna()
            if i == 0:
                final_df = temp_df.copy()
            else:
                # combine dataframes from last (combined) and current energy bin(s)
                # for some reason, if I try to bin all the energy bins at the same time, the method doesn't work
                final_df = pd.concat([final_df,temp_df]) 
        final_df = final_df.reset_index(drop=True) #need to do this (I don't know/forgot why it doesn't work without it)
        # bin energy
        final_df["Energy_Bin"] = pd.cut(final_df["ENERGY"], e_bins, labels = e_bin_labels, right=False)
        source_ns = []
        for i in range(len(final_df)):
            source_ns.append(source_n)
        final_df["Source_Name"] = source_ns
        if (t_bins is not None) and (t_int is None):
            # groupby().size() to get counts per timebin per RA bin per Dec bin per energy bin
            # This creates a special pandas series object with structured labels in columns (from my understanding/experimenting)
            counts = final_df.groupby(['Source_Name','Time_Bin','RA_Bin','Dec_Bin', 'Energy_Bin']).size()
            counts.columns = ['Source_Name','Time_Bin','RA_Bin','Dec_Bin', 'Energy_Bin', 'Counts']
            return counts
        else:
            # same but without time bin (for single image)
            counts = final_df.groupby(['RA_Bin','Dec_Bin', 'Energy_Bin'],  observed=False).size()
            counts.columns = ['RA_Bin','Dec_Bin', 'Energy_Bin', 'Counts']
            image =  np.rollaxis(np.expand_dims(counts.values.reshape((56,56,6)),axis=0), 2,1)
            return image[0]


    def print_alt_zoom_image(self, src_image = None, loc_only = False):
        """
        Generates a plot with multiple subplots displaying 2d count images, with option to display nearby sources
        
        :param src_image: nearby sources input image data. Passes additional image data to the function for overlay with the main image data (optional)
        :param loc_only: boolean flag that determines whether only the location of stars should be plotted or if additional
        information should be included, such as their average flux, which changes the size of the stars accordingly (optional)
        :return: returns 2d image plot as a matplotlib figure object along with subplot object
        """

        ra_bins = np.empty((6, self.bin_num+1))
        y_ticks = np.copy(ra_bins)
        for j, max_ang in enumerate([12,5,3,1.5,1,0.6]):
            ra_bins[j][0] = str(-max_ang)
            y_ticks[j][0] = 0
            for i in range(self.bin_num):
                ra_bins[j][i+1] = str(round(-max_ang + (2*max_ang/self.bin_num)*(i+1), 3))
                y_ticks[j][i+1] = i+1

        dec_bins = np.copy(ra_bins)

        mask = y_ticks[0] % self.tick_skip == 0
        mask[len(y_ticks[0])//2] = True
        y_ticks1 = []
        dec_bins1 = []
        for i in range(6):
            y_ticks1.append(y_ticks[i][mask])
            dec_bins1.append(dec_bins[i][mask])
        for i in range(6):
            if y_ticks[i][-1] not in y_ticks1[i]:
                y_ticks1[i] = np.append(y_ticks1[i], y_ticks[i][-1])
                y_ticks1[i] = np.delete(y_ticks1[i], -2)
                dec_bins1[i] = np.append(dec_bins1[i], dec_bins[i][-1])
                dec_bins1[i] = np.delete(dec_bins1[i], -2)
        
        y_ticks1 = np.array(y_ticks1)
        dec_bins1 = np.array(dec_bins1)
        
        x_ticks1 = np.copy(y_ticks1)
        ra_bins1 = np.copy(dec_bins1)

        image1 = np.copy(self.image).astype(float)
        image1[image1<0.4] = None
        if src_image is not None:
            src_image1 = np.copy(src_image).astype(float)
            src_image1[src_image1==0.0] = None

        figure, ax = plt.subplots(2,3, sharex=False, sharey=False, constrained_layout=True)

        e_bins = [r'$10^2$ - $10^{2.5}$',
                  r'$10^{2.5}$ - $10^3$',
                  r'$10^3$ - $10^{3.5}$',
                  r'$10^{3.5}$ - $10^4$',
                  r'$10^4$ - $10^{4.5}$',
                  r'$10^{4.5}$ - $3 \cdot 10^5$']
        
        cmap = mpl.cm.Reds(np.linspace(0.30,0.87,20))
        cmap = mpl.colors.ListedColormap(cmap)
        for i in range(6):
            if i <= 2:
                n_plt = ax[0,i]
            else:
                n_plt = ax[1,i-3]
            im = n_plt.pcolormesh(image1[i]) # double check if image is not flipped
     
            if src_image is not None:
                src_image_masked = np.ma.masked_invalid(src_image1[i])
                x,y = np.meshgrid(np.arange(src_image1.shape[1]), np.arange(src_image1.shape[2]))
                y = y[src_image_masked.mask == False]
                x = x[src_image_masked.mask == False]
                sizes = src_image1[i][src_image_masked.mask == False] * 200  # Adjust multiplier for desired size
                center_x = self.bin_num // 2
                center_y = self.bin_num // 2
                distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                min_distance_idx = np.argmin(distances)

                # Plot all stars in red first
                if not loc_only:    
                    n_plt.scatter(x, y, s=sizes, color='red', marker='*', alpha=0.7)
                else:   
                    n_plt.scatter(x, y, color='red', marker='*', alpha=0.7)

                # Plot the most central star in black
                if src_image1[i][y[min_distance_idx], x[min_distance_idx]] is not None:
                    if not loc_only:
                        n_plt.scatter(x[min_distance_idx], y[min_distance_idx], s=sizes[min_distance_idx], color='black', marker='*')
                    else:
                        n_plt.scatter(x[min_distance_idx], y[min_distance_idx], color='black', marker='*')
            
            n_plt.set_xticks(ticks = x_ticks1[i], labels = ra_bins1[i])
            n_plt.set_yticks(ticks = y_ticks1[i], labels = dec_bins1[i])
            if i % 3 == 0:  # For leftmost plots
                n_plt.set_ylabel('Y-Axis [Deg]')
            if i >= 3:  # For bottom plots
                n_plt.set_xlabel('X-Axis [Deg]')
            n_plt.set_title(e_bins[i] + ' MeV')

            if i <= 2:
                ax[0,i] = n_plt
            else:
                ax[1,i-3] = n_plt
            colorbar = figure.colorbar(im, ax = n_plt)
            if i == 2:
                colorbar.set_label('Photon Count')
            elif i == 5:
                colorbar.set_label('Photon Count')

            if i <= 2:
                ax[0,i] = n_plt
            else:
                ax[1,i-3] = n_plt
            
        figure.set_figwidth(9)
        if self.image_dir is not None:
            if self.image_pos is not None:
                count = self.image_pos
                image_name = 'counts_image_' + str(count) + '.png'
            else:    
                files = os.listdir(self.image_dir)
                image_name = 'counts_image_0.png'
                count = 0
                while image_name in files:
                    count += 1
                    image_name = 'counts_image_' + str(count) + '.png' 
            figure.savefig(self.image_dir + image_name)
            
        return figure, ax
    
    
    def make_nearby_srcs_image(self, src_name = None, cat_path = cat_path, loc_only = False):
        """
        Generates an image based on nearby sources of a specified source in the 4FGL catalogue
        
        :param src_name: name of the gamma-ray source in the 4FGL catalog of interest
        :param loc_only: boolean for whether to only consider the location of the nearby
        sources without adding their flux values to the image, defaults to False (optional)
        :return: Returns a 3D numpy array `image` that represents
        locations of nearby sources in different bins based on their relative positions to a
        specified source or a default source location.
        """

        catal = fits.open(cat_path)[1].data
        if src_name is not None:
            ra = catal['RAJ2000'][np.where(catal['Source_Name'] == src_name)[0]][0]
            dec = catal['DEJ2000'][np.where(catal['Source_Name'] == src_name)[0]][0]
        else:
            ra = self.source_loc[0]
            dec = self.source_loc[1]

        ra_bins = np.empty((6, self.bin_num+1))
        for j, max_ang in enumerate([12,5,3,1.5,1,0.6]):
            ra_bins[j][0] = -max_ang
            for i in range(self.bin_num):
                ra_bins[j][i+1] = (round(-max_ang + (2*max_ang/self.bin_num)*(i+1), 3))

        dec_bins = ra_bins
        
        image = np.zeros((6,self.bin_num,self.bin_num))
        catal = rfn.require_fields(catal, [("Source_Name", "U25"), ("RAJ2000", "f"), ("DEJ2000", "f"), ("Flux1000", "f"), ("Variability_Index", "f")])
        catal = pd.DataFrame(catal)
        catal['PSI'] = np.absolute(great_circle_dis(ra, dec, catal['RAJ2000'], catal['DEJ2000']))
        catal = catal.drop(catal[catal.PSI > self.max_psi].index)
        max_flux = max(catal['Flux1000'])
        catal['Flux1000'] /= max_flux
        new_ras, new_decs = angle_rotation(dec, ra, catal['DEJ2000'], catal['RAJ2000'])
        ra_difs = np.array(new_ras).squeeze()
        ra_difs[ra_difs < -180] += 360
        ra_difs[ra_difs > 180] -= 360
        catal['RA_DIF'] = ra_difs
        catal['DEC_DIF'] = new_decs
        catal = catal.sort_values(by='PSI', ignore_index = True)
        for j in range(len(catal['Flux1000'])):
            for i in range(6):
                for ra_ind in range(self.bin_num):
                    for dec_ind in range(self.bin_num):
                        if ra_bins[i][ra_ind] <= float(catal['RA_DIF'][j]) < ra_bins[i][ra_ind + 1]:
                            if dec_bins[i][dec_ind] <= float(catal['DEC_DIF'][j]) < dec_bins[i][dec_ind + 1]:
                                if not loc_only:
                                    image[i][dec_ind][ra_ind] += catal['Flux1000'][j]
                                else:
                                    image[i][dec_ind][ra_ind] = 1
        return image
    
                
    def create_LC_bins(self, ts_opt=[16, 25], max_ts=1, e_check = 1000, p_check = 0.5, 
                            save_ts=False, save_arr=False, min_time=3600*24, verbose = 1):
        """
        This function creates adaptive time bins based on certain criteria, generates images, and estimates TS
        values within those time bins.
        
        :param ts_opt: list that specifies the range of TS (Test Statistic) values within which each time bin's TS should fall.
        :param e_check: threshold value for filtering out potential time bins within a time window based on their (additional) events' energy in MeV. Defaults to 1 GeV, any 
        events BELOW this threshold are not considered.
        :param p_check: threshold value for filtering out potential time bins within a time window based on their (additional) events' proximity to the source in degrees; any 
        events ABOVE this threshold are not considered. Defaults to 0.5 deg.
        :param save_ts: determines whether to save the calculated TS values of the final time bins. Defaults to False.
        :param save_arr: determines whether to save the generated image arrays of the final time bins. Defaults to False.
        :param min_time: size of the time window in seconds within which to check all overlapping time bins' TS within this at a time. Defaults to 1 day.
        :param verbose: control whether or not to print out progress messages and information during the time bin search. 0 = no messages, 1 = include messages when a timebin is found (default), 2 = include messages about attempted time windows, 3 = include messages about each predicted TS in the time windows.
        :return: Returns final time bins for the given fermi dataframe, as well as their predicted TS and image arrays if their corresponding save booleans are set to True (otherwise, these are just empty lists)
        """
        p_t_df = self.create_fermi_df().copy()
        p_t_df = p_t_df.sort_values(by='TIME').reset_index()
        timebins = [p_t_df['TIME'][0]]
        l_time = p_t_df['TIME'][0]
        ts_list = []
        image_arrays = []
        search_finished = True
        con_count = 1
        image_batch = []
        ts_batch = []
        t_ints = []
        filter_df = p_t_df.copy().drop(p_t_df[(p_t_df.ENERGY < e_check) & ((np.abs(p_t_df.RA_DIF) > p_check) | (np.abs(p_t_df.DEC_DIF) > p_check))].index)
        filter_df = filter_df.drop_duplicates(subset ='TIME',keep= 'last')
        for r_time in filter_df['TIME']:
            if search_finished:
                con_count = 1 
            if r_time == filter_df['TIME'].iloc[-1]:
                t_ints.append([l_time, r_time])
                r_time = p_t_df['TIME'].iloc[-1]
                t_ints.append([l_time, r_time])
            elif r_time - l_time <= con_count*min_time:
                t_ints.append([l_time, r_time])
                continue
            if (len(t_ints) <=1) and (r_time != p_t_df['TIME'].iloc[-1]):
                t_ints.append([l_time, r_time])
                search_finished = False
                con_count += 1
                continue
            time_window = [l_time, l_time + con_count*min_time]
            if verbose >= 2:
                print(f'Checking time window: {time_window}', flush=True)
            if not self.allow_mp:    
                # if not quiet:
                #     print('Making images')
                init_time = time.time()
                counts = self.create_2d_alt_zoom_images(fermi_df=p_t_df, t_bins = t_ints, overlap=True)
                images = np.rollaxis(counts.values.reshape((int(len(counts)/(56*56*6)),56,56,6)), 2,1).astype('uint16')
                image_batch.extend(images)
                if verbose >= 3:
                    print(f'Image gen took {time.time() - init_time} s, now estimating TS')
                init_time = time.time()
                ts_batch.extend(np.array(mini_predict_ts(images, model_path=self.model_path, max_ts=max_ts, show_ts=(verbose >= 3), num_threads=1)).squeeze())
                ts_batch = np.array(ts_batch).squeeze()
                if verbose >= 3:
                    print(f'TS estimation took {time.time() - init_time} s')

            else:
                init_time = time.time()  
                images, ts = parallel_process(fermi_df = p_t_df, t_ints = t_ints, max_ts=max_ts, num_threads=self.num_threads, 
                                        num_workers=self.num_workers, model_path=self.model_path, create_image_func = self.create_2d_alt_zoom_images, show_ts=(verbose >= 3)
                                        )
                image_batch.extend(images)
                ts_batch.extend(ts)
                ts_batch = np.array(ts_batch).squeeze()
                if len(t_ints) > len(ts_batch):
                    t_ints = t_ints[:len(ts_batch)]
                if verbose >= 3:
                    print(f'Image gen + TS estimation took {time.time() - init_time} s')
            if np.shape(ts_batch) and (np.shape(ts_batch != (0,))):
                best_inds = np.argwhere((ts_batch <= ts_opt[1]) & (ts_batch >= ts_opt[0]))
                if len(best_inds) == 0:
                    if (np.all(ts_batch < ts_opt[0])):
                        if (r_time < filter_df['TIME'].iloc[-1]) and (r_time < p_t_df['TIME'].iloc[-1]):
                            if verbose >= 2:
                                print('No high TS, going to next time window')
                            search_finished = False
                            con_count += 1
                            t_ints = []
                            image_batch = []
                            ts_batch = []
                            continue
                        else:
                            if save_ts:
                                ts_list.append(ts_batch[-1])
                            if save_arr:
                                image_arrays.append(image_batch[-1])
                            timebins.append(t_ints[-1][1])
                            if verbose >= 1:
                                print(f'Last timebin = {t_ints[-1]}')
                                print(f'Last TS found = {ts_batch[-1]}')
                            break
                    else:
                        best_inds = np.argwhere(ts_batch >= ts_opt[0])
                        best_ind = best_inds[np.argmin(ts_batch[ts_batch >= ts_opt[0]])][0]
                    
                else:
                    best_ind = best_inds[-1][0]
                over_ts_count = len(ts_batch[ts_batch >= ts_opt[1]])
                if over_ts_count == 0:
                    if verbose >= 2:
                        print('Last good TS was still in range of threshold, going to next time window')
                    search_finished = False
                    con_count += 1
                    t_ints = [t_ints[best_ind]]
                    ts_batch = []
                    image_batch = []
                    continue
                elif over_ts_count > 0:
                    if verbose >= 2:
                        print('There was a too high TS present, taking nearest best previous TS')
                    best_ind = np.argwhere(ts_batch >= ts_opt[1])[0][0]
                    if (ts_batch[best_ind-1] >= ts_opt[0]) and (best_ind != 0):
                        best_ind -= 1
            elif np.shape(ts_batch) == (0,):
                if verbose >= 2:
                    print('No high energy/close proximity events at all, going to next time window')
                search_finished = False
                con_count += 1
                t_ints = []
                ts_batch = []
                image_batch = []
                continue
            else:
                best_ind = -1
                ts_batch = [ts_batch[0]]
                image_batch = [image_batch[0]]
                if r_time < filter_df['TIME'].iloc[-1]:
                    t_ints = [t_ints[-1]]
                    ts_batch = []
                    image_batch = []
                    continue
            if save_ts:
                ts_list.append(ts_batch[best_ind])
            if save_arr:
                image_arrays.append(image_batch[best_ind])
            
            timebins.append(t_ints[best_ind][1])
            if verbose >= 1:
                print(f'Best timebin = {t_ints[best_ind]}')
                print(f'Best TS found = {ts_batch[best_ind]}')
                print(f'Total time bins: {len(timebins) - 1}')
                perc = round((t_ints[best_ind][1] - p_t_df['TIME'].iloc[0])/(p_t_df['TIME'].iloc[-1]- p_t_df['TIME'].iloc[0]) * 100, 2)  
                print(f'{perc}% of total time processed', flush=True)
            l_time = t_ints[best_ind][1]
            t_ints = []
            ts_batch = []
            image_batch = []
            search_finished = True
            if r_time == filter_df['TIME'].iloc[-1]:
                if r_time != t_ints[best_ind][1]:
                    image = self.create_2d_alt_zoom_images(fermi_df=p_t_df, t_int = [l_time, r_time])
                    ts = mini_predict_ts(image, self.model_path, max_ts, show_ts=(verbose >= 3))
                    if save_ts:
                        ts_list.append(ts)
                    if save_arr:
                        image_arrays.append(image)
                    timebins.append([l_time, r_time])
                    if verbose >= 1:
                        print(f'Last timebin = {[l_time, r_time]}')
                        print(f'Last TS found = {ts}')
                break
        if timebins[-1] == timebins[-2]:
            timebins.pop(-1)
        elif timebins[-1] < timebins[-2]:
            timebins.pop(-2)
            ts_list.pop(-2)
        if verbose >= 0:
            print(f'Total time bins: {len(timebins) - 1}')
            print('Timebin search complete!')
        return timebins, ts_list, image_arrays

    
def worker_func(fermi_df, t_int, max_ts, model_path, num_threads, create_image_func, show_ts):
    image = create_image_func(fermi_df=fermi_df, t_int = t_int)
    ts = mini_predict_ts(image, model_path, num_threads, max_ts, show_ts=show_ts)
    return image, ts

def parallel_process(fermi_df, t_ints, max_ts, model_path, num_threads, create_image_func, num_workers=None, show_ts=False):
    if num_workers is None:
        num_workers = mp.cpu_count()

    args = [(fermi_df, t_int, max_ts, model_path, num_threads, create_image_func, show_ts) for t_int in t_ints]
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(worker_func, args)

    images, ts_batch = zip(*results)
    images = np.concatenate(images, axis=0).squeeze()
    ts_batch = np.concatenate(ts_batch, axis=0).squeeze()

    return images, ts_batch 

def mini_predict_ts(image, model_path, num_threads, max_ts = 1, show_ts = False):
    interpreter = tflite.Interpreter(model_path=model_path,num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, axis=0).astype(np.float32))
    interpreter.invoke()
    ts = max_ts*interpreter.get_tensor(output_details[0]['index'])
    if len(image.shape) < 4:
        if show_ts:
            print('predicted ts: ' + str(ts[0]), flush=True)
        return ts[0]
    else:
        if show_ts:
            print('predicted ts: ' + str(ts), flush=True)
        return ts
    

# alternative slower method, not working, do not use!
    
    # def create_LC_bins_alt(self, ts_opt=[16, 25], sens=5, max_ts=1, e_check = 1000, p_check = 0.5, no_restart=False,
    #             save_ts=False, save_arr=False, triple_check=True, trip_sens = 5, min_time=None):
    #     p_t_df = self.create_fermi_df().copy()
    #     p_t_df = p_t_df.sort_values(by='TIME').reset_index()
    #     l = len(p_t_df) // sens
    #     timebins = [p_t_df['TIME'][0]]
    #     t0 = 0
    #     t1 = np.argmin(np.abs(p_t_df['TIME'] - min_time - p_t_df['TIME'][0])) if min_time is not None else 400 - sens
    #     ts_list, image_arrays = [], []
    #     image_check = 0
    #     restarted = False
    #     while True:
    #         time_skip_count = 0
    #         t1 += sens
    #         if t1 < len(p_t_df) - 1:
    #             while (p_t_df['TIME'].iloc[t1] == p_t_df['TIME'].iloc[t1 + 1]) and (t1 < len(p_t_df) - 1): 
    #                 t1 += 1
    #                 time_skip_count += 1
    #                 if t1 >= len(p_t_df) - 1: 
    #                     break
                    
    #         if t1 >= len(p_t_df) - 1:
    #             timebins.append(p_t_df['TIME'].iloc[-1])
    #             t_int = [p_t_df['TIME'].iloc[t0], p_t_df['TIME'].iloc[-1]]
    #             image = self.create_2d_alt_zoom_images(fermi_df=p_t_df, t_int=t_int)
    #             ts = predict_ts(image, self.model, max_ts, show_ts=True)
    #             if save_ts:
    #                 ts_list.append(ts)
    #             if save_arr:
    #                 image_arrays.append(image)
    #             break
            
    #         energy_check = np.array(p_t_df['ENERGY'].iloc[(t1-time_skip_count-sens):t1])
    #         ra_check = np.array(p_t_df['RA_DIF'].iloc[(t1-time_skip_count-sens):t1])
    #         dec_check = np.array(p_t_df['DEC_DIF'].iloc[(t1-time_skip_count-sens):t1])
            
    #         high_energy_count = len(energy_check[energy_check>=e_check])
    #         prox_mask = (np.abs(ra_check)<=p_check) & (np.abs(dec_check)<=p_check)
    #         prox_check = len(ra_check[prox_mask])
            
    #         if (high_energy_count < 1) and (prox_check < 1):
    #             continue
                
    #         t_int = [p_t_df['TIME'].iloc[t0], p_t_df['TIME'].iloc[t1]]
    #         image = self.create_2d_alt_zoom_images(fermi_df=p_t_df, t_int=t_int)
    #         if (np.sum(image) < 400) or (np.sum(image) == image_check):
    #             continue
            
    #         print(f'Time interval = {t_int}')
    #         ts = predict_ts(image, self.model, max_ts, show_ts=True)
    #         if ts_opt[0] <= ts <= ts_opt[1]:
    #             if triple_check and (len(p_t_df) - 1 - (t1 + 6*trip_sens) > 0):
    #                 ts_trip, t1_trip, image_trip = self.trip_test(p_t_df=p_t_df,sens=trip_sens,ts=ts, t_int=[t0,t1], max_ts=max_ts, image=image)
    #                 if np.mean(ts_trip) < ts_opt[0]:
    #                     print('Predicted ts does not pass triple check')
    #                     continue

    #             if save_ts:
    #                 ts_list.append(ts)
    #             if save_arr:
    #                 image_arrays.append(image)

    #             timebins.append(p_t_df['TIME'].iloc[t1])
    #             print(f'time bin found! Total time bins: {len(timebins) - 1}')
    #             print(f'{round(t1 / len(p_t_df) * 100, 2)}% of total time processed', flush=True)
    #             t0 = t1
    #             image_check = 0
    #             restarted = False
    #             t1 += 400 - sens
    #         elif ts > ts_opt[1]:
    #             if triple_check and (len(p_t_df) - 1 - (t1 + 6*trip_sens) > 0):
    #                 ts_trip, t1_trip, image_trip = self.trip_test(p_t_df=p_t_df,sens=trip_sens,ts=ts, t_int=[t0,t1], max_ts=max_ts, image=image)
    #                 if np.mean(ts_trip) < ts_opt[0]:
    #                     print('Predicted ts does not pass triple check')
    #                     image_check = np.sum(image)
    #                     continue
    #                 elif np.mean(ts_trip) >= ts_opt[0]:
    #                     good_ts = np.argwhere((ts_opt[0] <= np.array(ts_trip)) & (np.array(ts_trip) <= ts_opt[1]))
    #                     if (len(good_ts) >= 1) or (restarted or no_restart):
    #                         if len(good_ts) >= 1:
    #                             print('Found a good TS in the triple check')
    #                             good_ind = good_ts[0][0]
    #                             ts = ts_trip[good_ind]
    #                             image = image_trip[good_ind]
    #                             t1 = t1_trip[good_ind]
    #                         else:
    #                             print("Already restarted search, taking this bin")
    #                         if save_ts:
    #                             ts_list.append(ts)
    #                         if save_arr:
    #                             image_arrays.append(image)
    #                         timebins.append(p_t_df['TIME'].iloc[t1])
    #                         print(f'time bin found! Total time bins: {len(timebins) - 1}')
    #                         print(f'{round(t1 / len(p_t_df) * 100, 2)}% of total time processed', flush=True)
    #                         t0 = t1
    #                         image_check = 0
    #                         t1 += 400 - sens
    #                         restarted = False
    #                     else:
    #                         print("Restarting search without headstart")
    #                         t1 = t0
    #                         restarted = True     
    #         else:
    #             image_check = np.sum(image)

    #     if timebins[-1] == timebins[-2]:
    #         timebins.pop(-1)

    #     return timebins, ts_list, image_arrays


    # def trip_test(self,p_t_df, sens, t_int, ts, max_ts, image):
    #     print('Triple checking ts')
    #     ts_trip = [ts]
    #     trip_count = 0
    #     trip_skip = 0
    #     t1_trip = []
    #     image_trip = []
    #     for j in range(1, 4):
    #         while trip_count < j*sens:
    #             t_trip_int = [p_t_df['TIME'].iloc[t_int[0]], p_t_df['TIME'].iloc[t_int[1] + trip_skip + j*sens]]
    #             trip_image = self.create_2d_alt_zoom_images(fermi_df = p_t_df, t_int=t_trip_int)
    #             trip_count = np.sum(trip_image) - np.sum(image)
    #             if trip_count < j*sens:
    #                 trip_skip += 1
    #         ts_trip.append(predict_ts(trip_image,self.model, max_ts, show_ts=False))
    #         t1_trip.append(t_int[1] + trip_skip + j*sens)
    #         image_trip.append(trip_image)
            
    #     return ts_trip, t1_trip, image_trip

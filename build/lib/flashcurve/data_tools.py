import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as au
# import astrotools as at
# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
from mechanize import Browser
import requests
import time
import glob
import os

        
def predict_ts(image, model, max_ts = 1, show_ts = False):
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        if len(image.shape) < 4:
            model.set_tensor(input_details[0]['index'], np.expand_dims(image, axis=0).astype(np.float32))
            ts = max_ts*model.get_tensor(output_details[0]['index'])
        else:
            input_shape = input_details[0]['shape']
            batch_size = input_shape[0]
            num_images = len(image)
            num_batches = num_images // batch_size
            results = []

            for i in range(num_batches):
                batch = image[i*batch_size:(i+1)*batch_size].astype("float32")
                model.set_tensor(input_details[0]['index'], batch)
                model.invoke()
                output_data = model.get_tensor(output_details[0]['index'])
                results.append(output_data)

            # Handle any remaining images that don't fit into a full batch
            remaining_images = image[num_batches*batch_size:].astype("float32")
            if len(remaining_images) > 0:
                padding = np.zeros((batch_size - len(remaining_images), 56, 56, 6), dtype=np.float32)
                padded_images = np.concatenate([remaining_images, padding], axis=0)
                model.set_tensor(input_details[0]['index'], padded_images)
                model.invoke()
                output_data = model.get_tensor(output_details[0]['index'])
                results.append(output_data[:len(remaining_images)])

            ts = max_ts*np.concatenate(results, axis=0)
       
        if len(image.shape) < 4:
            if show_ts:
                print('predicted ts: ' + str(ts[0]), flush=True)
            return ts[0]
        else:
            if show_ts:
                print('predicted ts: ' + str(ts), flush=True)
            return ts


def run_inference(batch, model_path, num_threads=2):
    
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], batch)

    # Run inference
    interpreter.invoke()

    # Get the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

# Step 3: Use Multiprocessing for Parallel Inference
def process_batch(images, start_idx, end_idx, model_path, num_threads=2):
    batch = images[start_idx:end_idx]

    model = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    model.allocate_tensors()
    input_details = model.get_input_details()
    input_shape = input_details[0]['shape']
    
    # Check if batch size matches, if not pad the batch
    batch_size = len(batch)
    if batch_size < input_shape[0]:
        padding = np.zeros((input_shape[0] - batch_size, 1, 6, 56, 56), dtype=np.float32)
        batch = np.concatenate([batch, padding], axis=0)

    # Run inference on the batch
    result = run_inference(batch.astype(np.float32), model_path, num_threads)

    # Remove padding results if batch was padded
    if batch_size < input_shape[0]:
        result = result[:batch_size]

    return result


def MJD_to_MET(mjd_time):
    MJDREF = 51910.0 + 7.428703703703703E-4
    return (mjd_time - MJDREF)*86400


def great_circle_dis(ra0,dec0,ra1,dec1):
    return (180/np.pi)*np.arccos(np.sin(dec0*np.pi/180)*np.sin(dec1*np.pi/180) +
            np.cos(dec0*np.pi/180)*np.cos(dec1*np.pi/180)*np.cos((ra0-ra1)*np.pi/180))


def angle_rotation(dec0, ra0, dec1, ra1):
    """
    Calculates rotation matrix to align dec0 & ra0 to x-axis
    Then returns transformed ra1 & dec1
    """

    main_dec = dec0
    main_ra = ra0
    old_dec = dec1
    old_ra = ra1
    # old_vec = at.coord.ang2vec(phi= old_ra, theta = old_dec)
    old_vec = SkyCoord(ra=old_ra, dec=old_dec, frame='icrs', unit=(au.deg, au.deg)).cartesian.xyz.value
    # unit vectors
    # v1 = at.coord.ang2vec(phi= main_ra, theta = main_dec)
    v1 = SkyCoord(ra=main_ra, dec=main_dec, frame='icrs', unit=(au.deg, au.deg)).cartesian.xyz.value
    v2 = [1,0,0]
    u = v1 / np.linalg.norm(v1)
    Ru = v2 / np.linalg.norm(v2)
    # dimension of the space and identity
    dim = u.size
    I = np.identity(dim)
    # the cos angle between the vectors
    c = np.dot(u, Ru)
    # a small number
    eps = 1.0e-10
    if np.abs(c - 1.0) < eps:
        # same direction
        R = I
    elif np.abs(c + 1.0) < eps:
        # opposite direction
        R = -I
    else:
        # the cross product matrix of a vector to rotate around
        K = np.outer(Ru, u) - np.outer(u, Ru)
        # Rodrigues' formula
        R = I + K + (K @ K) / (1 + c)
    a = R @ old_vec
    new_dec = np.arcsin(a[2])*180/np.pi
    new_ra = np.arctan2(a[1],a[0])*180/np.pi
    if isinstance(new_ra, np.floating):
        if new_ra < 0:
            new_ra += 360
    else:
        new_ra[new_ra<0] += 360

    return new_ra, new_dec


def plot_lc(pred_lc_path, fine_lc_path = None, alpha =0.5):
    tb_centers = []
    data = np.load(pred_lc_path, allow_pickle=True).item()
    tb_err = []
    for i in range(len(data['tmin'])):
        average = (data['tmin'][i]+data['tmax'][i])/2
        tb_centers.append(average)
        err = abs(data['tmax'][i] - average)
        tb_err.append(err)
    if fine_lc_path is not None:
        fine_tb_centers = []
        fine_data = np.load(fine_lc_path, allow_pickle=True).item()
        for i in range(len(fine_data['tmin'])):
            average = (fine_data['tmin'][i]+fine_data['tmax'][i])/2
            fine_tb_centers.append(average)
    fig, ax = plt.subplots(figsize = (8,5))
    ax.set_ylabel(r'Photon Flux [ph cm$^{−2}$ s$^{-1}$]')
    ax.set_xlabel('Date [MET]')
    ax.errorbar(tb_centers, list(data['flux']), xerr=tb_err, yerr = list(data['flux_err']), marker = '.', color = 'red', ls = 'none', label = 'Model Estimator', alpha = alpha)
    if fine_lc_path is not None:
        ax.plot(fine_tb_centers, list(fine_data['flux']), color = 'blue', marker = '')

    return fig, ax


def get_ul_data(ra, dec, data_dir, years = 5, get_sc = False, t_int = None, max_energy = 300000, max_angle = 12): 
    """
    Retrieve and download data files from the Fermi-LAT data server.
    
    :param ra: Right ascension coordinate for the data query.
    :param dec: Declination coordinate for the data query. 
    :param data_dir: Directory where the downloaded data files will be saved (string)
    :param get_sc: If set to `True`, the spacecraft file will be included in the data download process. Defaults to `False` (optional)
    :param t_int: Time interval for the data query. mut be a list containing start and end times of the
    interval in MET (Mission Elapsed Time) format.
    :param max_energy: Maximum energy value in MeV for the data query. Defaults to 300 GeV (optional)
    :param max_angle: Maximum angle in degrees for the shape of the region of interest. Specifies the
    angular radius for the data query. Defaults to 12 degrees (optional)
    :return: The `get_ul_data` function is returning None.
    """
    
    def get_download_links(html):
            split = html.decode().split('wget')
            status = int(html.decode().split('he state of your query is ')[1][:1])
            if status == 2:
                return [(i.split('</pre>'))[0].strip().replace('\n', '')
                        for i in split[2:]]
            else:
                return []

    def download_file(url, outfolder):
        local_filename = os.path.join(outfolder, url.split('/')[-1])
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk: 
                    f.write(chunk)
        return
    
    if t_int is None:
        time_period = str(2024-years) +'-01-01 00:00:01 , 2024-01-01 00:00:01'
    else:
        if t_int[0] < 239557417:
            t_int[0] = 239557417
        time_period = str(int(t_int[0])) + ' , ' + str(int(t_int[1]))

    url = "https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi"

    br = Browser()
    br.set_handle_robots(False)
    br.open(url)
    br.select_form(nr=1)
    br["coordfield"] = str(ra) + ", " + str(dec)
    br["coordsystem"] = [u'J2000']
    if t_int is not None:
        br["timetype"] = [u'MET']
    br["timefield"] = time_period
    br["shapefield"] = str(max_angle)
    br["energyfield"] = "100, " + str(max_energy)
    
    # we load the spacecraft file separately
    br.form.find_control('spacecraft').items[0].selected = get_sc

    response = br.submit()
    r_text = response.get_data()
    query_url = r_text.decode().split('at <a href="')[1].split('"')[0]

    print('Query URL {}'.format(query_url))
    seconds = float(r_text.decode().split('complete is ')[1].split(' ')[0])
    wait = 0.75 * seconds
    print('Wait at least {} seconds for the query to complete'.format(wait))
    time.sleep(wait)

    html = requests.get(query_url).text.encode('utf8')
    download_urls = get_download_links(html)
    while len(download_urls) == 0:
        print('Query not yet finished...Wait 10 more seconds')
        time.sleep(10)
        html = requests.get(query_url).text.encode('utf8')
        download_urls = get_download_links(html)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # remove old data

    old_files = glob.glob(os.path.join(data_dir, '*PH*.fits'))
    sc_old_files = glob.glob(os.path.join(data_dir, '*SC*.fits'))
    if sc_old_files:     
        old_files.append(*sc_old_files)
    for f in old_files:
        print("delete ", f)
        os.remove(f)

    for tmp_url in download_urls:
        download_file(tmp_url, data_dir)

    return
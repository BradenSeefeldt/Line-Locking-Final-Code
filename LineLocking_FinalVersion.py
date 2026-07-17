from astropy.table import Table
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from tqdm.notebook import tqdm
import astropy.constants as const
import matplotlib.pyplot as plt
from astropy.io import fits
import fitsio
from scipy.integrate import trapezoid
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.constants import m_e, c, e
from scipy.signal import medfilt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)




def determine_spec_fileName(trough, specdir):
    """
    This function constructs the file path for the spectra file based on the 
    values provided in the 'trough' dictionary and a specified directory 
    where the spectra files are stored. The file name is generated using 
    the 'PLATE', 'FIBERID', and 'MJD' keys from the 'trough' dictionary, 
    and the resulting file is located under the specified 'specdir' directory.

    Args:
        trough (dict): A dictionary containing the keys 'PLATE', 'FIBERID', 
                       and 'MJD', which are used to construct the file name.
        specdir (str): The base directory where the spectra files are stored.

    Returns:
        str: The constructed file path for the spectra file in the format 
             "{specdir}/{plate}/spec-{plate}-{mjd}-{fiberid:04d}.fits".
    """

    plate = int(trough['PLATE'])
    fiberid = int(trough['FIBERID'])
    mjd = int(trough['MJD'])
    fileName = str(specdir) +  f"{plate}/spec-{plate}-{mjd}-{fiberid:04d}.fits"

    return fileName



def read_spec(fileName):
    """
    This function loads spectral data from a specified FITS file, extracts 
    relevant columns ('FLUX', 'LOGLAM', 'MODEL'), and converts them into 
    a pandas DataFrame. Additionally, it computes the observed wavelength 
    ('WAVE_OBS') based on the 'LOGLAM' column.

    Args:
        fileName (str): The path to the FITS file containing the spectral data.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted columns 
                           ('FLUX', 'LOGLAM', 'MODEL') and an additional 
                           'WAVE_OBS' column, representing the observed 
                           wavelength.
    """
    
    # Load in the spectra
    spec_table = Table.read(fileName, hdu=1)

    # Convert table to a pandas dataframe
    cols = ["FLUX", "LOGLAM", "MODEL"]
    spec_df = spec_table[cols].to_pandas()

    # Determine a few additional quantities
    spec_df['WAVE_OBS'] = 10**spec_df['LOGLAM']
        
    return spec_df



def read_PCA_cont(balfilename, pcaeigenfile, trough_table, trough_num, rest_wl_QSO):
    """
    Generate a PCA-based continuum model for a quasar based on the provided PCA coefficients and eigenvalues.

    This function reads the PCA coefficients and eigenvalue data, matches the appropriate quasar based 
    on the provided trough catalog, interpolates the PCA components onto the rest wavelength scale of the 
    quasar's spectrum, and returns a model continuum for the quasar.

    Parameters:
    -----------
    balfilename : str
        The path to a FITS file containing the PCA information for a set of quasars. This file should 
        include PCA coefficients, plate, MJD, and fiberID data.
    
    pcaeigenfile : str
        The path to a FITS file containing the eigenvalue data, which includes the wavelengths and PCA 
        component values.
    
    trough_table : pandas.DataFrame
        A DataFrame containing the catalog of troughs, which includes the plate, MJD, and fiberID columns 
        to match against the PCA catalog.
    
    trough_num : int
        The index of the current trough in `trough_table` for which the PCA continuum model is being generated.
    
    rest_wl_QSO : numpy.ndarray
        An array of quasar rest-frame wavelengths for the quasar spectrum, onto which the PCA components will be 
        interpolated and combined.

    Returns:
    --------
    numpy.ndarray
        A 1D array containing the model continuum for the quasar, constructed by combining the PCA components 
        weighted by the PCA coefficients.
    
    Notes:
    ------
    The function assumes that the PCA catalog (`balfilename`) and the trough catalog (`trough_table`) 
    are correctly aligned by plate, MJD, and fiberID. The interpolation is performed on the PCA components 
    based on the rest wavelength grid of the quasar.
    
    The PCA eigenvalue file (`pcaeigenfile`) must contain the wavelength grid ('WAVE') and the corresponding 
    PCA components, which will be used to build the final continuum model for the quasar.

    Example:
    --------
    cont_model = read_PCA_cont("pca_catalog.fits", "eigenvalues.fits", trough_data, 0, rest_wavelengths)
    """

    # read in catalog containing the PCA information
    balcat_info_fits = fits.open(balfilename)

    # find quasar with PCA info matching current quasar -- same quasar name in the PCA catalog and trough catalog
    bigcond = np.where((balcat_info_fits[1].data['PLATE'] == int((trough_table['PLATE'].values)[trough_num] )) 
                    & (balcat_info_fits[1].data['MJD'] == int((trough_table['MJD'].values)[trough_num] )) 
                    & (balcat_info_fits[1].data['FIBERID'] == int((trough_table['FIBERID'].values)[trough_num] )))[0][0]

    # these are the pca coefficients for each quasar 
    pca_coeffs = balcat_info_fits[1].data['PCA_COEFFS'][bigcond]

    # this is the eigenvalues information (wavelength and components)
    pcaeigen = fitsio.read(pcaeigenfile)

    # Interpolate the PCA components onto the REST WAVELENGTH VALUES OF THE QUASAR spectrum
    pca_eigen = []
    for item in pcaeigen.dtype.names:
        if item == 'WAVE' : continue
        pca_eigen.append(np.interp(rest_wl_QSO, pcaeigen['WAVE'], pcaeigen[item]))

    ### 2-D array with interpolated PCA components that replaces the hardcoded components
    ipca = np.concatenate(pca_eigen,axis=0).reshape(len(pca_eigen),len(pca_eigen[0]))

    cont_model = np.zeros(len(rest_wl_QSO), dtype=float)
    for ss in range(len(pca_coeffs)):
        cont_model += pca_coeffs[ss]*ipca[ss]

    return cont_model

    

def find_sky_line(wavelengths):
    """
    This function takes an array of wavelengths and returns a numpy array 
    of the same length, where each element is either 0 or 1. A value of 
    1 indicates that the corresponding wavelength is compromised by a 
    known sky line, while 0 means the wavelength is unaffected. The function 
    checks for specific wavelength ranges that are known to be contaminated 
    by sky lines, based on typical atmospheric features.

    Args:
        wavelengths (numpy.ndarray or pandas.Series): Array or Series of 
                                                      wavelength values.

    Returns:
        numpy.ndarray: A binary array with the same length as the input, 
                        where 1 represents a wavelength affected by a sky line 
                        and 0 represents a clean wavelength.
    """
    sky_flag = np.zeros(len(wavelengths),dtype=int)
    for x in range(len(wavelengths)):
        longona = (wavelengths.values)[x]
        if ((longona>4042.  and longona<4050.0)  or (longona>4355.6 and longona<4363.6) or 
            (longona>5458.5 and longona<5466.6)  or (longona>5572.9 and longona<5585.0)  or 
            (longona>5682.  and longona<5695.) or (longona>5885.5 and longona<5902.2) or 
            (longona>6235.9 and longona<6241.) or (longona>6256.  and longona<6263.)  or 
            (longona>6296.5 and longona<6311.1)  or (longona>6329.9 and longona<6332.9)  or 
            (longona>6362.1 and longona<6369.5) or (longona>6498.3 and longona<6502.8) or 
            (longona>6553.9 and longona<6559.9)  or (longona>6826.6 and longona<6839.0) or 
            (longona>6862.8 and longona<6873.8)  or (longona>6880.  and longona<6884.) or 
            (longona>6888.5 and longona<6892.) or (longona>6900.  and longona<6903.) or 
            (longona>6913.  and longona<6915.)or (longona>6923.2 and longona<6927.8) or 
            (longona>6940.  and longona<6942.9) or (longona>6948.7 and longona<6953.3) or 
            (longona>6977.6 and longona<6982.2) or (longona>3932.  and longona<3937.) or 
            (longona>3966.  and longona<3972.)):
            sky_flag[x] = 1
    return sky_flag



def get_real_spectra_stats(trough_table, spectrum_directory, Continuum_Type='PCA', balfilename='N/A', pcaeigenfile='N/A', meanContinuum='N/A'):
    """
    Analyzes spectral data from a set of troughs, calculates various statistics for each trough, and returns the results 
    as a pandas DataFrame along with the spectra to be stacked.

    The function computes the following for each trough:
    - Depth at the blue (1548.2 Å) and red (1550.7 Å) lines.
    - Flux noise, significance, trough depth, and width.
    - A rejection counter indicating the number of troughs rejected due to poor data quality.
    
    The function reads spectral data for each trough from a provided directory, normalizes the flux to the model, 
    smooths it, and performs statistical analysis on it. It also checks for contamination by sky background and 
    rejects the trough if contamination is found.

    Parameters:
    -----------
    trough_table : pandas.DataFrame
        A DataFrame containing information about troughs, including redshift and velocity bounds for each trough.
        
    spectrum_directory : str
        Path to the directory containing the spectral data files.

    use_PCA : str, optional, default='Auto'
        If 'Yes', the function uses PCA models for continuum subtraction. 
        If 'No', it uses the `MODEL` data from the spectra.
        If 'Auto', the function chooses between the PCA model and the `MODEL` data from the spectra based off of chi squared
        
    balfilename : str, optional, default='N/A'
        The path to the PCA catalog file. Required if `use_PCA` is True.
        
    pcaeigenfile : str, optional, default='N/A'
        The path to the PCA eigenvalue file. Required if `use_PCA` is True.

    Returns:
    --------
    tuple : (pandas.DataFrame, list)
        A tuple containing:
        - A pandas DataFrame with the following columns:
            - "Blue Depths": Depth at the blue line (1548.2 Å) for each trough.
            - "Red Depths": Depth at the red line (1550.7 Å) for each trough.
            - "Noises": Estimated noise level for each trough.
            - "Significances": Significance of the trough depth.
            - "Trough Depths": Depth of the trough (normalized flux).
            - "Trough Widths": Width of the trough in Å.
            - "Rejection Counter": The number of rejected troughs.
        - A list of spectra (arrays) to be stacked.

    Notes:
    ------
    The function uses a restframe correction based on the trough redshift to align the spectra and performs 
    rebinning to a fixed wavelength grid (0.3 Å). It also applies a Gaussian smoothing to the flux and computes 
    the noise and significance based on the smoothed flux values.
    
    Troughs are rejected if their flux at the blue or red line is contaminated by sky background (indicated by 
    the sky flag).
    """
    BlueDepths = []
    RedDepths = []
    TripDepths = []
    Noises = []
    TroughWidths = []
    TroughDepths = []
    spectra_2be_stacked = []
    weights = []
    red_shifts = []
    outflow_speeds = []
    trough_center =[]

    rejection_counter = 0
    blue_line = 1548.2
    red_line = 1550.77
    trip_line = 1545.63
    c = const.c.value/1000 #Km/s

    # Find the first quasar name, read in the spectral data, and determine the sky line pixels for this first trough. This will setup a few important 
    # variables including the 'previous_qso_name', 'current_spec_data', and 'current_sky_flag'
    previous_qso_name = determine_spec_fileName(trough_table.iloc[0], spectrum_directory)
    current_spec_data = read_spec(previous_qso_name)
    current_sky_flag = find_sky_line(current_spec_data['WAVE_OBS'])

    # loop over the troughs
    total_troughs = len(trough_table)
    for i in tqdm(range(total_troughs)):
        # current quasar name to be used for this trough
        specname = determine_spec_fileName(trough_table.iloc[i], spectrum_directory)

        if specname != previous_qso_name:
            # Read in new spectrum data
            current_spec_data = read_spec(specname)
            current_sky_flag = find_sky_line(current_spec_data['WAVE_OBS'])
            previous_qso_name = specname

        # Shift Spectrum to the trough restframe
        current_spec_data['WAVE_CIV'] = current_spec_data['WAVE_OBS'] /( 1. + trough_table.iloc[i]['Z_MIN'])

        # Rebin everything into equally spaced 0.3 Angstrom bins
        rebin_wavelengths = np.arange(900.2, 1599.8, 0.3)
        rebin_sky_flag = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, current_sky_flag)
        rebin_sky_flag[np.where(rebin_sky_flag > 0.01)] = 1
        
        rebin_flux = np.zeros(len(rebin_wavelengths),dtype=float)
        rebin_flux = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, current_spec_data['FLUX'])

        # Determine the continuum spectra for this quasar
        if Continuum_Type == 'PCA':
            rest_wl_QSO = current_spec_data['WAVE_OBS'] /( 1. + trough_table.iloc[i]['Z_PCA'])
            pca_model = read_PCA_cont(balfilename, pcaeigenfile, trough_table, i, rest_wl_QSO)
            # Rebin
            rebin_model = np.zeros(len(rebin_wavelengths), dtype=float)
            rebin_model = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, pca_model)
        elif Continuum_Type == 'Model':
            rebin_model = np.zeros(len(rebin_wavelengths), dtype=float)
            rebin_model = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, current_spec_data['MODEL'])
        elif Continuum_Type == 'Auto':
            if trough_table['PCA_CHI2'].iloc[i] > trough_table['SDSS_CHI2'].iloc[i]:
                rest_wl_QSO = current_spec_data['WAVE_OBS'] /( 1. + trough_table.iloc[i]['Z_PCA'])
                pca_model = read_PCA_cont(balfilename, pcaeigenfile, trough_table, i, rest_wl_QSO)
                # Rebin
                rebin_model = np.zeros(len(rebin_wavelengths), dtype=float)
                rebin_model = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, pca_model)
            else:
                rebin_model = np.zeros(len(rebin_wavelengths), dtype=float)
                rebin_model = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, current_spec_data['MODEL'])
        elif Continuum_Type == 'Mean':
            model = np.loadtxt(meanContinuum, delimiter=' ', skiprows=1, unpack=True)
            # Rebin
            rebin_model = np.zeros(len(rebin_wavelengths), dtype=float)
            rebin_model = np.interp(rebin_wavelengths, model[0], model[1])

        else:
            print('Warning: Unknown Continuum_Type argument!')
            return

        # Find the index that is closest to the 1548.2 A value
        blue_index = np.argmin(np.abs(rebin_wavelengths - blue_line))
        red_index = np.argmin(np.abs(rebin_wavelengths - red_line))
        trip_index = np.argmin(np.abs(rebin_wavelengths - trip_line))
        
        # Determine if the flux at any of the three lines has been compromised from sky background
        if (rebin_sky_flag[blue_index] != 1) and (rebin_sky_flag[red_index] != 1) and (rebin_sky_flag[trip_index] != 1):
            
            # Find the lower bound
            wl_trough_lower = blue_line*(1-((trough_table.iloc[i]['VMAX']-trough_table.iloc[i]['POSMIN'])/c))
            trough_lower_index = np.argmin(np.abs(rebin_wavelengths - wl_trough_lower))
            if rebin_wavelengths[trough_lower_index]<wl_trough_lower:
                trough_lower_index+=1
            # Find the upper bound
            wl_trough_upper = blue_line*(((trough_table.iloc[i]['POSMIN']-trough_table.iloc[i]['VMIN'])/c)+1)
            trough_upper_index = np.argmin(np.abs(rebin_wavelengths - wl_trough_upper))
            if rebin_wavelengths[trough_upper_index]>wl_trough_upper:
                trough_upper_index-=1


            # Determine flux and wavelength values between these bounds
            Trough_FLUX = np.array(rebin_flux[trough_lower_index:trough_upper_index+1])
            Trough_MODEL = np.array(rebin_model[trough_lower_index:trough_upper_index+1])
            Trough_SKYFLAG = np.array(rebin_sky_flag[trough_lower_index:trough_upper_index+1])
            Trough_Wavelengths = np.array(rebin_wavelengths[trough_lower_index:trough_upper_index+1])

            # Refind the new index corresponding to the 1548.2 feature
            new_blue_index = np.argmin(np.abs(Trough_Wavelengths - blue_line))
            new_red_index = np.argmin(np.abs(Trough_Wavelengths - red_line))
            new_trip_index = np.argmin(np.abs(Trough_Wavelengths - trip_line))

            # Normalize the flux 
            Trough_FLUX_Normalized = Trough_FLUX/Trough_MODEL
            if np.max(Trough_FLUX_Normalized)<100:

                # Smooth the flux 
                Smooth = gaussian_filter1d(Trough_FLUX_Normalized, 5)
                Trough_FLUX_Smoothed = Trough_FLUX_Normalized/Smooth

                noise_points = []
                for l, point in enumerate(Trough_FLUX_Smoothed):
                    if (l != new_blue_index) and (l != (new_blue_index-1)) and (l != (new_blue_index+1)) and (Trough_SKYFLAG[l] != 1) and (l>=4) and (l<(len(Trough_FLUX_Smoothed)-4)):
                        noise_points.append(point)
                
                if len(noise_points) > 8:
                    # Calculate quantities
                    t_width = wl_trough_upper-wl_trough_lower
                    t_depth = 1-np.median(Smooth[3:-4])
                    noise = (1-t_depth)*np.std(noise_points, ddof=1)
                    b_depth = 1-Trough_FLUX_Normalized[new_blue_index]
                    r_depth = 1-Trough_FLUX_Normalized[new_red_index]
                    trip_depth = 1-Trough_FLUX_Normalized[new_trip_index]

                    # Append values
                    if (noise>0) and (noise<2) and (b_depth<1) and (r_depth<1) and (b_depth>0) and (r_depth>0) and (trip_depth<1) and (t_depth>0) and (t_depth<1) and (t_width>=5.7):
                        BlueDepths.append(b_depth)
                        RedDepths.append(r_depth)
                        TripDepths.append(trip_depth)
                        Noises.append(noise)
                        TroughWidths.append(np.log10(t_width))
                        TroughDepths.append(t_depth)
                        weights.append(1/(trough_table.iloc[i]["SN_MEDIAN_ALL"]**(-2)+0.1**2))
                        red_shifts.append(trough_table.iloc[i]['Z_PCA'])
                        outflow_speeds.append(trough_table.iloc[i]['POSMIN'])
                        trough_center.append((wl_trough_lower+wl_trough_upper)/2.) 

                        # Append Spectra to later be Stacked
                        temp_array=[]
                        min_wavelength = np.min(current_spec_data['WAVE_CIV'])
                        for j in range(len(rebin_wavelengths)):
                            if rebin_sky_flag[j] != 1 and rebin_wavelengths[j]>min_wavelength:
                                temp_array.append(rebin_flux[j]/rebin_model[j])
                            else:
                                temp_array.append(np.nan)
                                
                        spectra_2be_stacked.append(np.array(temp_array))
                    else:
                        rejection_counter+=1
                else:
                    rejection_counter+=1
            else:
                rejection_counter+=1
        else:
            rejection_counter+=1

    # Create a pandas dataframe to hold all of the information
    df = pd.DataFrame({"Blue Depths": np.array(BlueDepths), 
                   "Red Depths": np.array(RedDepths), 
                   "Triplet Depths": np.array(TripDepths),
                   "Noises": np.array(Noises), 
                   "Trough Depths": np.array(TroughDepths), 
                   "Trough Widths": np.array(TroughWidths), 
                   "Weights": np.array(weights),
                   "Red Shifts": np.array(red_shifts),
                   "Outflow Speeds": np.array(outflow_speeds),
                   "Trough Center": np.array(trough_center),
                   "Rejection Counter": np.array(rejection_counter)})
    return df, spectra_2be_stacked


def get_mock_spectra_stats(Trough_Table):
    """
    This function processes each trough in the input `trough_table`, reads the associated spectra 
    from a given directory, and calculates various properties, including the depth at the blue and red 
    lines (C IV), the significance of the trough, trough depth, and width. It also tracks how many troughs 
    are rejected based on certain criteria, such as contamination by sky lines. The resulting data is returned 
    in a pandas DataFrame.

    Args:
        trough_table (pandas.DataFrame): A DataFrame containing information about each trough, including 
                                          the quasar redshift and other relevant parameters for identifying 
                                          trough boundaries.
        spectrum_directory (str): The directory containing the spectrum files. The spectra are read for 
                                  each trough in the table using this directory.

    Returns:
        pandas.DataFrame: A DataFrame containing calculated statistics for each trough, including:
            - 'Blue Depths': The depth of the blue line (C IV 1548.2 Å).
            - 'Red Depths': The depth of the red line (C IV 1550.7 Å).
            - 'Noises': The noise level calculated from the surrounding flux values.
            - 'Significances': The significance of the trough based on the flux and noise.
            - 'Trough Depths': The depth of the trough, measured as 1 minus the normalized flux.
            - 'Trough Widths': The width of the trough, measured in Å.
            - 'Rejection Counter': The number of troughs rejected due to various criteria.
            
        list: A list of spectra to be stacked, where each spectrum is represented as a normalized flux 
              array over the region of interest.
    """
    BlueDepths = []
    RedDepths = []
    TripDepths = []
    Noises = []
    TroughWidths = []
    TroughDepths = []
    mock_spectra_2be_stacked = []
    weights = []

    rejection_counter = 0
    blue_counter = 0
    red_counter = 0
    noise_counter = 0
    blue_line = 1548.2
    red_line = 1550.77
    trip_line = 1545.63
    num_of_troughs = len(Trough_Table)

    for i in tqdm(range(num_of_troughs)):
        wavelengths = Trough_Table['Wavelengths'][i]
        trough_width = Trough_Table['Trough Width'][i]
        box_center = Trough_Table['Trough Center'][i]

        # Find trough boundaries
        box_min_wave = box_center-(trough_width/2)
        trough_lower_index = np.argmin(np.abs(wavelengths - box_min_wave))
        if wavelengths[trough_lower_index]<box_min_wave:
            trough_lower_index+=1
        box_max_wave = box_center+(trough_width/2)
        trough_upper_index = np.argmin(np.abs(wavelengths - box_max_wave))
        if wavelengths[trough_upper_index]>box_max_wave:
            trough_upper_index-=1

        Flux = Trough_Table['Flux'][i]
        Trough_Flux = Flux[trough_lower_index:trough_upper_index+1]
        Trough_Wavelengths = wavelengths[trough_lower_index:trough_upper_index+1]


        # Smooth the flux
        Smooth = gaussian_filter1d(Trough_Flux, 5)
        Flux_Smoothed = Trough_Flux/Smooth
        # Find the blue, red, and triplet indexs
        blue_idx = np.argmin(np.abs(Trough_Wavelengths - blue_line))
        red_idx = np.argmin(np.abs(Trough_Wavelengths - red_line))
        trip_idx = np.argmin(np.abs(Trough_Wavelengths - trip_line))
        
        # Calculate the Noise
        noise_points = []
        for j in range(len(Trough_Wavelengths)):
            if (j>=4) and (j<(len(Trough_Wavelengths)-4)) and (j !=blue_idx) and (j != (blue_idx+1)) and (j != (blue_idx-1)):
                noise_points.append(Flux_Smoothed[j])
        if len(noise_points)>8:
            # Add Values to arrays
            TroughDepths.append(1-np.median(Smooth[3:-4]))
            Noise = np.median(Smooth[3:-4])*np.std(noise_points, ddof=1)
            Noises.append(Noise)
            RedDepths.append(1-Trough_Flux[red_idx])
            BlueDepths.append(1-Trough_Flux[blue_idx])
            TripDepths.append(1-Trough_Flux[trip_idx])
            TroughWidths.append(np.log10(trough_width))
            weights.append(1/((Noise*np.median(Smooth[3:-4]))**(2)+0.1**2))

            # Save spectra to be stacked later
            min_idx = np.argmin(Flux[25:-25])+25 
            mock_spectra_2be_stacked.append(np.array(Flux[min_idx-25:min_idx+25]))
            if (wavelengths[min_idx] < (1548.2+0.31)) and (wavelengths[min_idx] > (1548.2-0.31)):
                blue_counter+=1
            elif (wavelengths[min_idx] < (1550.77+0.31)) and (wavelengths[min_idx] > (1550.77-0.31)):
                red_counter+=1
            else:
                noise_counter+=1
        else:
            rejection_counter+=1

    # Create a pandas dataframe to hold all of the information
    df = pd.DataFrame({"Blue Depths": np.array(BlueDepths), 
                   "Red Depths": np.array(RedDepths),
                   "Triplet Depths": np.array(TripDepths), 
                   "Noises": np.array(Noises), 
                   "Weights": np.array(weights),
                   "Trough Depths": np.array(TroughDepths), 
                   "Trough Widths": np.array(TroughWidths), 
                   "Rejection Counter": np.array(rejection_counter),
                   "Blue Stack Counter": np.array(blue_counter),
                   "Red Stack Counter": np.array(red_counter),
                   "Noise Stack Counter": np.array(noise_counter)})
    return df, mock_spectra_2be_stacked



def Doublet(wavelength, delta, b, r):
    return np.exp(-b*np.exp(-(wavelength - 1548.2)**2/(2*delta**2))-r*np.exp(-(wavelength - 1550.77)**2/(2* delta**2)))


def Triplet(wavelength, delta, b, r, t):
    return np.exp( -b*np.exp(-(wavelength - 1548.2)**2/(2*delta**2))-r*np.exp(-(wavelength - 1550.77)**2/(2* delta**2))-t*np.exp(-(wavelength - 1545.63)**2/(2* delta**2)))


def Boxcar(wavelength, box_center, box_depth, box_width, red_depth, triplet_depth, Type='standard'):
    #if triplet_depth == None:
    #    center = box_center
    #else:
    center = box_center
    # Determine Trough boundaries
    box_min_wave = np.min([1548.1, center-(box_width/2)])
    box_max_wave = center+(box_width/2)
    # Determine slope and y-intercept params for slants
    m1 = -box_depth/(1548.2-box_min_wave)
    m2 = box_depth/(box_max_wave-1548.2)
    b1 = 1-m1*box_min_wave
    b2 = 1-m2*box_max_wave

    if (wavelength>box_min_wave) and (wavelength<1548.2):
        #if triplet_depth==None:
        #    return 1-box_depth
        #else:
        if Type=='standard':
                if triplet_depth<(box_depth):
                    return m1*wavelength+b1
                else:
                    return 1-box_depth
        elif Type=='slant':
                return m1*wavelength+b1
        else:
                return 1-box_depth
    elif (wavelength>=1548.2) and (wavelength<box_max_wave):
        if Type == 'standard':
            if red_depth<(box_depth):
                return m2*wavelength+b2
            else:
                return 1-box_depth
        elif Type == 'box':
            return 1-box_depth
        elif Type == 'slant':
            return m2*wavelength+b2
    else:
        return 1
    

def Sawtooth(wavelength, box_depth, box_width, red_depth):
    doublet_center = np.mean([1550.77, 1548.2])
    box_min_wave = doublet_center-(box_width/2)
    box_max_wave = doublet_center+(box_width/2)
    m1 = -box_depth/(1548.2-box_min_wave)
    m2 = box_depth/(box_max_wave-1548.2)
    b1 = 1-m1*box_min_wave
    b2 = 1-m2*box_max_wave
    if (wavelength>1548.2) and (wavelength<box_max_wave):
        if red_depth<box_depth:
            return m2*wavelength+b2
        else:
            return 1-box_depth
    else:
        return 1
    


def Convert_to_depth_params(blue_depth, red_depth, absorption_width):
    """
    Given the required blue and red line depths and the absorption width, 
    this function solves for the parameters (b, c) that describe the 
    Gaussian profiles for the blue and red lines of a C IV doublet.

    Args:
        blue_depth (float): Depth of the blue line (C IV 1548.2 Å), between 0 and 1.
        red_depth (float): Depth of the red line (C IV 1550.7 Å), between 0 and 1.
        absorption_width (float): Width of the absorption feature, typically in Ångströms.

    Returns:
        tuple: A tuple (b, c) containing the parameters that define the Gaussian profiles for the doublet.
    """
    alpha = np.exp((-(1550.77-1548.2)**2)/(2*absorption_width**2))
    F_blue = np.log(1-blue_depth)
    F_red = np.log(1-red_depth)
    
    b = (alpha*F_red-F_blue)/(1-alpha**2)
    c = -F_red-b*alpha
    return b, c

def Convert_to_depth_params_triplet(blue_depth, red_depth, triplet_depth, absorption_width):
    """
    Given the required blue and red line depths and the absorption width, 
    this function solves for the parameters (b, c) that describe the 
    Gaussian profiles for the blue and red lines of a C IV doublet.

    Args:
        blue_depth (float): Depth of the blue line (C IV 1548.2 Å), between 0 and 1.
        red_depth (float): Depth of the red line (C IV 1550.7 Å), between 0 and 1.
        absorption_width (float): Width of the absorption feature, typically in Ångströms.

    Returns:
        tuple: A tuple (b, r, t) containing the parameters that define the Gaussian profiles for the doublet.
    """
    alpha = np.exp((-(1550.77-1548.2)**2)/(2*absorption_width**2))
    beta = np.exp((-(1550.77-1545.63)**2)/(2*absorption_width**2))
    
    # Set up a system of equations in matrix form to solve
    SoE_Matrix = np.array([[-1, -alpha, -alpha],
                           [-alpha, -1, -beta],
                           [-alpha, -beta, -1]])
    SoE_Vector = np.array([[np.log(1-blue_depth)],
                            [np.log(1-red_depth)],
                             [np.log(1-triplet_depth)]])
    
    solution = np.dot(np.linalg.inv(SoE_Matrix), SoE_Vector)
    
    return solution[0][0], solution[1][0], solution[2][0]



def generate_spectra(wavelength_range, KDE, return_params=False, trough_type='standard', triplet=False):
    """
    This function generates a synthetic spectrum by sampling parameters 
    (blue depth, red depth, noise, trough depth, and trough width) from a 
    Kernel Density Estimate (KDE) distribution. It combines the doublet and 
    boxcar models for absorption troughs and adds noise to create a realistic spectrum.

    Args:
        wavelength_range (tuple): A tuple specifying the range of wavelengths 
                                  for the spectrum, in the form (min_wavelength, max_wavelength).
        KDE (object): A trained KDE object used to sample parameters for the synthetic spectrum.
        return_params (bool, optional): If True, the function returns the sampled parameters 
                                        along with the wavelength and flux data. Default is False.

    Returns:
        tuple: 
            - numpy.ndarray: The wavelengths for the generated spectrum.
            - numpy.ndarray: The generated flux values for each wavelength.
            - pandas.DataFrame (optional): A DataFrame containing the sampled parameters, if `return_params=True`.
    """
    wavelengths = np.arange(wavelength_range[0], wavelength_range[1], 0.3)
    blue_idx = np.argmin(np.abs(wavelengths - 1548.2))
    red_idx = np.argmin(np.abs(wavelengths - 1550.77))
    trip_idx = np.argmin(np.abs(wavelengths - 1545.63))

    # This is used to ensure there are no negative flux values
    Flux = [-1,-1]
    while np.min(Flux)<0:
        # pull values for the depth, noise, and line ratio parameters from the KDE distributions
        resample = KDE.resample(size=1)
        blue_depth = resample[0][0]
        red_depth = resample[1][0]
        noise = resample[2][0]
        trough_depth = resample[3][0]
        trough_width = 10**resample[4][0]
        trough_center = resample[6][0]
        trip_depth = resample[5][0]
        #if triplet:
        #    trip_depth = resample[5][0]
        #    while (noise<0) or (noise>2) or (blue_depth>1) or (red_depth>1) or (trip_depth>1) or (trough_depth<0) or (trough_depth>1) or (trough_width<6.0):
        #        resample = KDE.resample(size=1)
        #        blue_depth = resample[0][0]
        #        red_depth = resample[1][0]
        #        noise = resample[2][0]
        #        trough_depth = resample[3][0]
        #        trough_width = 10**resample[4][0]
        #        trip_depth = resample[5][0]
        while (noise<0) or (noise>2) or (blue_depth>1) or (red_depth>1) or (trough_depth<0) or (trough_depth>1) or (trough_width<6.0):
                resample = KDE.resample(size=1)
                blue_depth = resample[0][0]
                red_depth = resample[1][0]
                noise = resample[2][0]
                trough_depth = resample[3][0]
                trough_width = 10**resample[4][0]
                trip_depth = resample[5][0]
                trough_center = resample[6][0]

        # Keep the line width constant
        absorption_width=0.5
        # Convert the blue and red depth values into the required parameters for the doublet function
        if triplet:
            blue_depth_param, red_depth_param, trip_depth_param = Convert_to_depth_params_triplet(blue_depth, red_depth, trip_depth, absorption_width)
        else:
            blue_depth_param, red_depth_param = Convert_to_depth_params(blue_depth, red_depth, absorption_width)

        Absorption_Lines = []
        Trough = []
        # Create 2 arrays, one for the boxcar function and one for the absorption function
        for wave in wavelengths:
            if triplet:
                Absorption_Lines.append(Triplet(wave, absorption_width, blue_depth_param, red_depth_param, trip_depth_param))
            else:
                Absorption_Lines.append(Doublet(wave, absorption_width, blue_depth_param, red_depth_param))

            if trough_type == 'sawtooth':
                Trough.append(Sawtooth(wave, trough_depth, trough_width, red_depth))
            elif trough_type == 'standard' or trough_type == 'box' or trough_type=='slant':
                #if triplet:
                Trough.append(Boxcar(wave, trough_center, trough_depth, trough_width, red_depth, triplet_depth=trip_depth, Type=trough_type))
                #else:
                #Trough.append(Boxcar(wave, trough_center, trough_depth, trough_width, red_depth, Type=trough_type))
            else:
                print('Unknown trough_type parameter!!!')
                return

        Trough = np.array(Trough)- 0.0505 # systematic offset to match the real data distribution and stacking spectrum


        Flux = []
        continuum = np.random.normal(loc=0, scale= noise, size = len(wavelengths))
        # Compare the boxcar and doublet funcstions and take the minimum value
        for i in range(len(wavelengths)):
            if (i==blue_idx)  or (i == red_idx) : # or (i == (blue_idx-1)) or (i == (blue_idx+1)) or (i == (red_idx-1)) or (i == (red_idx+1)) :
                Flux.append(Absorption_Lines[i])
            elif ((i == trip_idx) ) and triplet:  # or (i == (trip_idx-1)) or (i == (trip_idx+1))
                Flux.append(Absorption_Lines[i])
            else:
                if (Absorption_Lines[i] < Trough[i]) and (Absorption_Lines[i]<0.99):
                    Flux.append(Absorption_Lines[i])     
                else:
                    Flux.append(Trough[i]+continuum[i])
        Flux = np.array(Flux)
    
    
    #if triplet:
    df = pd.DataFrame({"Blue Depth": blue_depth, 
                   "Red Depth": red_depth,
                   "Triplet Depth": trip_depth, 
                   "Noise": noise, 
                   "Trough Depth": trough_depth, 
                   "Trough Width": trough_width,
                   "Trough Center": trough_center}, index=[0])
    #else:
    #    df = pd.DataFrame({"Blue Depth": blue_depth, 
    #                "Red Depth": red_depth, 
    #                "Noise": noise, 
    #                "Trough Depth": trough_depth, 
    #                "Trough Width": trough_width,
    #                "Trough Center": trough_center}, index=[0])



    if return_params:
        return wavelengths, Flux, df
    else:
        return wavelengths, Flux


def Full_Spectrum_Stack(trough_table, spectrum_directory='N/A', balfilename='N/A', pcaeigenfile='N/A', Continuum_Type='PCA'):
    previous_qso_name = determine_spec_fileName(trough_table.iloc[0], spectrum_directory)
    current_spec_data = read_spec(previous_qso_name)
    current_sky_flag = find_sky_line(current_spec_data['WAVE_OBS'])

    spectra_2be_stacked = []
    outflow_velocity = []
    weight = []
    total_troughs = len(trough_table)
    for i in tqdm(range(total_troughs)):
        # current quasar name to be used for this trough
        specname = determine_spec_fileName(trough_table.iloc[i], spectrum_directory)

        if specname != previous_qso_name:
            # Read in new spectrum data
            current_spec_data = read_spec(specname)
            current_sky_flag = find_sky_line(current_spec_data['WAVE_OBS'])
            previous_qso_name = specname

        # Shift Spectrum to the trough restframe
        current_spec_data['WAVE_CIV'] = current_spec_data['WAVE_OBS'] /( 1. + trough_table.iloc[i]['Z_MIN'])

        # Rebin everything into equally spaced 0.3 Angstrom bins
        rebin_wavelengths = np.arange(900.2, 1599.8, 0.3)
        rebin_sky_flag = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, current_sky_flag)
        rebin_sky_flag[np.where(rebin_sky_flag > 0.01)] = 1
        
        rebin_flux = np.zeros(len(rebin_wavelengths),dtype=float)
        rebin_flux = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, current_spec_data['FLUX'])

        # Determine the continuum spectra for this quasar
        if Continuum_Type == 'PCA':
            rest_wl_QSO = current_spec_data['WAVE_OBS'] /( 1. + trough_table.iloc[i]['Z_PCA'])
            pca_model = read_PCA_cont(balfilename, pcaeigenfile, trough_table, i, rest_wl_QSO)
            # Rebin
            rebin_model = np.zeros(len(rebin_wavelengths), dtype=float)
            rebin_model = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, pca_model)
        elif Continuum_Type == 'Model':
            rebin_model = np.zeros(len(rebin_wavelengths), dtype=float)
            rebin_model = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, current_spec_data['MODEL'])
        elif Continuum_Type == 'Auto':
            if trough_table['PCA_CHI2'].iloc[i] > trough_table['SDSS_CHI2'].iloc[i]:
                rest_wl_QSO = current_spec_data['WAVE_OBS'] /( 1. + trough_table.iloc[i]['Z_PCA'])
                pca_model = read_PCA_cont(balfilename, pcaeigenfile, trough_table, i, rest_wl_QSO)
                # Rebin
                rebin_model = np.zeros(len(rebin_wavelengths), dtype=float)
                rebin_model = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, pca_model)
            else:
                rebin_model = np.zeros(len(rebin_wavelengths), dtype=float)
                rebin_model = np.interp(rebin_wavelengths, current_spec_data['WAVE_CIV'].values, current_spec_data['MODEL'])
        elif Continuum_Type == 'Mean':
            model = np.loadtxt(meanContinuum, delimiter=' ', skiprows=1, unpack=True)
            # Rebin
            rebin_model = np.zeros(len(rebin_wavelengths), dtype=float)
            rebin_model = np.interp(rebin_wavelengths, model[0], model[1])

        else:
            print('Warning: Unknown Continuum_Type argument!')
            return
        
        # Append Spectra to later be Stacked
        temp_array=[]
        min_wavelength = np.min(current_spec_data['WAVE_CIV'])
        for j in range(len(rebin_wavelengths)):
            if rebin_sky_flag[j] != 1 and rebin_wavelengths[j]>min_wavelength and rebin_model[j] != 0:
                temp_array.append(rebin_flux[j]/rebin_model[j])

            else:
                temp_array.append(np.nan)
        spectra_2be_stacked.append(np.array(temp_array))
    return spectra_2be_stacked



def get_area(trough_nums, trough_table, specdir, balfilename, pcaeigenfile, relativeAbsorptionFilename='N/A', plot=False, verbose=False):	
    
    # Read in Spectra
    qso_name = determine_spec_fileName(trough_table.iloc[trough_nums[0]], specdir)
    spec_data = read_spec(qso_name)
    blue_line = 1548.20
    c = const.c.value/1000 #Km/s
    Z_qso = trough_table.iloc[trough_nums[0]]['Z_PCA']

    rest_wl_QSO = spec_data['WAVE_OBS'] /( 1. + Z_qso)
    pca_model = read_PCA_cont(balfilename, pcaeigenfile, trough_table, trough_nums[0], rest_wl_QSO)

    # Read in Relative Absorption values
    if relativeAbsorptionFilename != 'N/A':
        relativeAbsorptionData = np.load(relativeAbsorptionFilename)

    # Calculate total luminosity
    total_emission = trapezoid(pca_model, spec_data['WAVE_OBS'])*10**(-17)
    D_L = cosmo.luminosity_distance(Z_qso)
    D_L_cm = D_L.to(u.cm)
    total_luminosity = 4*np.pi*D_L_cm.value**2*total_emission
    if verbose:
        print(f"Total Luminosity of Quasar: {total_luminosity} erg/s")

    if plot:
        labelsize=20
        ticksize=15
        plt.figure(figsize=(7, 5))
        plt.xlabel(r"Wavelength [$\mathrm{\AA}$]", fontsize=labelsize)
        plt.ylabel(r" $F_\lambda$ [$10^{-17}$erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]", fontsize=labelsize)
        plt.tick_params(axis='both', which='major', direction='in', top=True, right=True,
                    length=5, width=1, labelsize=ticksize+2)
        plt.tick_params(axis='both', which='minor', direction='in', top=True, right=True,
                    length=3, width=1)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.tight_layout()

    total_area = 0
    widths = []
    speeds = []
    total_absorbed_luminosity = 0
    total_Mdot = 0
    total_kinetic_luminosity = 0
    lowest_wl = 1450
    highest_wl = 1560

    for num in trough_nums:
        widths.append(trough_table.iloc[num]['WIDTH'])
        # Get the shifted wavelengths in the trough restframe and PCA continuum
        wls = spec_data['WAVE_OBS'] /( 1. + trough_table.iloc[num]['Z_MIN'])

        #   Find the lower bound
        wl_trough_lower = blue_line*(1-((trough_table.iloc[num]['VMAX']-trough_table.iloc[num]['POSMIN'])/c))
        trough_lower_index = np.argmin(np.abs(wls - wl_trough_lower))
        # Find the upper bound
        wl_trough_upper = blue_line*(((trough_table.iloc[num]['POSMIN']-trough_table.iloc[num]['VMIN'])/c)+1)
        trough_upper_index = np.argmin(np.abs(wls - wl_trough_upper))

        # Determine flux and wavelength values between these bounds
        Trough_SpecificFLUX = np.array(spec_data['FLUX'][trough_lower_index:trough_upper_index+1])
        Trough_SpecificMODEL = np.array(pca_model[trough_lower_index:trough_upper_index+1])
        Trough_Wavelengths = np.array(wls[trough_lower_index:trough_upper_index+1])       

        # Only integrate area where actual data is below model
        flux_diff = np.where(Trough_SpecificFLUX < Trough_SpecificMODEL, Trough_SpecificMODEL - Trough_SpecificFLUX, 0)

        # Integrate using trapezoidal rule
        area_below_model = trapezoid(flux_diff, Trough_Wavelengths*(1+trough_table.iloc[num]['Z_MIN']))
        # Find outflow velocity
        vmin = trough_table.iloc[num]['VMIN']
        vmax = trough_table.iloc[num]['VMAX']
        vavg = 0.5 * (vmin + vmax)
        speeds.append(vavg)
        # Account for other absorption lines
        if relativeAbsorptionFilename != 'N/A':
            lineFactor = 1/np.interp(vavg, relativeAbsorptionData['velocities'], relativeAbsorptionData['relative_vals'])
            area_below_model = area_below_model*lineFactor
        total_area+=area_below_model
        # Find absorbed luminosity
        luminosity = 4*np.pi*D_L_cm.value**2*area_below_model*10**(-17)
        total_absorbed_luminosity +=luminosity
        # Find Mdot
        Mdot = ((luminosity*u.erg/u.s)/(vavg*u.km/u.s*const.c)).to(u.Msun / u.yr)
        total_Mdot +=Mdot
        # Find other usefull quantities
        kinetic_luminosity = (0.5*Mdot*(vavg*u.km/u.s)**2).to(u.erg/u.s)
        total_kinetic_luminosity +=kinetic_luminosity

        if verbose:
            print(f"Trough {num} has area: {area_below_model*10**(-17):.2e} erg/s/cm^2, and Mdot:{Mdot.value:.2e} Msun/yr")

        if plot:
            #shift wls back to quasar frame
            plot_wls = Trough_Wavelengths*( 1. + trough_table.iloc[num]['Z_MIN']) /( 1. + trough_table.iloc[num]['Z_PCA'])
            if plot_wls[0] < lowest_wl:
                lowest_wl = plot_wls[0]
            if plot_wls[-1] > highest_wl:
                highest_wl = plot_wls[-1]

            plt.fill_between(plot_wls, Trough_SpecificFLUX, Trough_SpecificMODEL,
                        where=(Trough_SpecificFLUX < Trough_SpecificMODEL), color='blue', alpha=0.5, label='Integrated Area')
    if plot:
        minn = np.argmin(np.abs(rest_wl_QSO-(lowest_wl-10)))
        maxx = np.argmin(np.abs(rest_wl_QSO-(highest_wl+20)))
        plt.plot(rest_wl_QSO[minn:maxx], spec_data['FLUX'][minn:maxx], 'k', label='Spectra')
        plt.plot(rest_wl_QSO[minn:maxx], pca_model[minn:maxx], label="PCA Continuum", color="green", linewidth=2)
        plt.legend(fontsize=labelsize-4)
        plt.savefig("./Results/Figures/Mdot_Example.png", bbox_inches="tight", dpi=300)
        plt.show()

    if verbose:
        print(f"Total Area: {total_area*10**(-17):.2e} erg/s/cm^2, made from {len(trough_nums)} troughs.")
        print(f"Total Mdot: {total_Mdot.value:.2e} Msun/yr, made from {len(trough_nums)} troughs.")

    # final calculations
    biggest_cloud_idx = np.argmax(widths)

    # Find mass inflow rate
    M_inflow = (total_luminosity*u.erg/u.s)/(0.1*(c*u.km/u.s)**2)
    M_inflow = M_inflow.to(u.Msun / u.yr)

    return {
    'L_bol': total_luminosity,
    'L_absorbed': total_absorbed_luminosity,
    'Mdot': total_Mdot.value,  # in Msun/yr
    'L_kin': total_kinetic_luminosity.value,  # in erg/s
    'Mass_Ratio': total_Mdot.value/M_inflow.value,
    'Num Troughs': len(trough_nums),
    'BAL Prob': trough_table.iloc[trough_nums[0]]['BAL_PROB'],
    'Total Width': np.sum(widths),
    'Max Width': np.max(widths),
    'Total Velocity': np.sum(speeds),
    'Max Velocity': np.max(speeds),
    'Biggest Cloud Velocity': speeds[biggest_cloud_idx],
    "Width/Velocity": widths[biggest_cloud_idx]/speeds[biggest_cloud_idx],
    "index": trough_nums[0],
    "Thing ID": trough_table.iloc[trough_nums[0]]['THING_ID'],
    "Fiber ID": trough_table.iloc[trough_nums[0]]['FIBERID']
    }

def column_density_from_tau(tau, line):
    """
    Calculate column density from optical depth at line center.

    Parameters
    ----------
    tau : float
        Optical depth at line center (dimensionless)
    f : float
        Oscillator strength of the transition
    lambda_rest : Quantity
        Rest wavelength of the line (e.g., 1548.2 * u.AA)
    b : Quantity
        Doppler width (velocity) in units of velocity, e.g., km/s

    Returns
    -------
    N : Quantity
        Column density [1 / cm^2]
    """
    if line == 1548:
        f = 0.18999
        lambda_rest = 1548.204 * u.AA
        b = 100 * u.km / u.s
    elif line == 1551:
        f = 0.09520
        lambda_rest = 1550.770 * u.AA
        b = 100 * u.km / u.s
    elif line == 977:
        f = 0.75700
        lambda_rest = 977.0201 * u.AA
        b = 100 * u.km / u.s
    elif line == 1032:
        f = 0.13250
        lambda_rest = 1031.926 * u.AA
        b = 100 * u.km / u.s
    elif line == 1038:
        f = 0.06580
        lambda_rest = 1037.616 * u.AA
        b = 100 * u.km / u.s
    elif line == 1239:
        f = 0.15600
        lambda_rest = 1238.821 * u.AA
        b = 100 * u.km / u.s
    elif line == 1243:
        f = 0.07770
        lambda_rest = 1242.804 * u.AA
        b = 100 * u.km / u.s
    elif line == 1216:
        f = 0.41640
        lambda_rest = 1215.670 * u.AA
        b = 100 * u.km / u.s
    
    # Use CGS units — statCoulomb, cm, g, etc.
    prefactor = (m_e * c) / (np.sqrt(np.pi) * e.gauss**2)
    N = prefactor * (tau * b) / (f * lambda_rest)
    return N.to(1 / u.cm**2)


def wrapper(args):
    return get_area(*args)

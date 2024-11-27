import os

import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import colors

import xarray as xr
import csv
import plotly.express as px

import astropy.io.fits as fits
import astropy.units as u, astropy.constants as consts

# from ipywidgets import interact

import scipy.ndimage as ndimage
import scipy.optimize as opt

import pyklip.instruments.Instrument as Instrument
import pyklip.parallelized as parallelized

######################################################
###### Preparing datasets functions
######################################################


def binning_datacube(init_cube, sizebin=20):
    size_init = init_cube.shape

    binned_cube = np.zeros((size_init[0] // sizebin, size_init[1], size_init[2]), dtype=init_cube.dtype)

    for i in range(size_init[0] // sizebin):
        subcube = init_cube[i * sizebin:(i + 1) * sizebin]
        binned_cube[i] = np.mean(subcube, axis=0)

    return binned_cube


def binning_parangs(init_parangs, sizebin=20):
    size_init = init_parangs.shape

    binned_parangs = np.zeros(size_init[0] // sizebin, dtype=init_parangs.dtype)

    for i in range(size_init[0] // sizebin):
        binned_parangs[i] = np.mean(init_parangs[i * sizebin:(i + 1) * sizebin], axis=0)

    return binned_parangs


######################################################
###### Plotting Functions
######################################################


def quick_heat_map(datacube, range_color=[0, 500], title="",coords_3D=None, reverse_x_ra = False):


    # line to solve error : numpy array is not native-endianness
    datacube = datacube.astype(datacube.dtype.newbyteorder("="))

    if len(datacube.shape) == 3:


        for key in coords_3D.keys():
            if "x (" in key:
                x_name = key
                continue
            if "y (" in key:
                y_name = key
                continue
            legend_slicers = key

        toto = xr.DataArray(coords=coords_3D, data=datacube, dims=[legend_slicers,y_name, x_name])

        fig = px.imshow(toto,
                        range_color=range_color,
                        title=title,
                        animation_frame=legend_slicers,
                        width=500,
                        aspect='equal',
                        origin='lower')

    elif len(datacube.shape) == 2:
        toto = xr.DataArray(data=datacube, dims=["y (pix)", "x (pix)"])

        fig = px.imshow(toto, range_color=range_color, title=title, width=500, aspect='equal', origin='lower')

    else:
        return
    
    if reverse_x_ra:
        fig['layout']['xaxis']['autorange'] = "reversed"


    fig.show()


# def show_plane(ax, plane, cmap="gray", title=None, vmin=None, vmax=None):
#     if vmin is None:
#         vmin = np.nanmin(plane)
#     if vmax is None:
#         vmin = np.nanmax(plane)

#     plt.pcolor(plane, cmap='plasma', vmin=vmin, vmax=vmax)
#     ax.imshow(plane)
#     ax.axis("off")

#     if title:
#         ax.set_title(title, fontsize=18)

# def explore_slices(data, cmap="gray", vmin=None, vmax=None, title=''):

#     N = len(data)

#     @interact(plane=(0, N - 1))
#     def display_slice(plane=0):
#         fig, ax = plt.subplots(figsize=(8, 8), dpi=80)

#         show_plane(ax, data[plane], title=title + f' number {plane}', cmap=cmap, vmin=vmin, vmax=vmax)

#         plt.show()

#     return display_slice

# def show_psf(PSF, vmin=None, vmax=None, title=None):
#     # PSF = remove_mean_2d(PSF)
#     if vmin is None:
#         vmin = np.nanmin(PSF)
#     if vmax is None:
#         vmin = np.nanmax(PSF)

#     fig = plt.figure(figsize=(8, 8), dpi=80)
#     ax = fig.add_subplot(111)
#     plt.axis('off')
#     ax.set_aspect('equal', adjustable='box')

#     plt.pcolor(PSF, cmap='plasma', vmin=vmin, vmax=vmax)
#     if title:
#         ax.set_title(title, fontsize=18)

#     plt.show()
#     plt.close()


def remove_mean_2d(image):
    return image - np.nanmean(image)


def quick_crop(image, dimout):
    """ --------------------------------------------------
    if dimout < dim : cropped image around pixel (dim/2,dim/2)

    Parameters
    ----------
    image : 2D array (float, double or complex)
            dim x dim array

    dimout : int
         dimension of the output array

    Returns
    ------
    im_out : 2D array (float)
        cropped image


    -------------------------------------------------- """
    return image[int((image.shape[0] - dimout) / 2):int((image.shape[0] + dimout) / 2),
                 int((image.shape[1] - dimout) / 2):int((image.shape[1] + dimout) / 2)]


def quick_crop3d(image, dimout):

    imcrop = np.zeros((image.shape[0], dimout, dimout))

    for i in range(image.shape[0]):
        imcrop[i] = quick_crop(image[i], dimout)

    return imcrop


######################################################
###### PSF Sub traction functions
######################################################


def simple_pca_already_centered(datacube_SPHERE_binned,
                                centers_images,
                                parangs_binned,
                                PCA_components,
                                output_dir='',
                                output_prefix=''):

    dataset = Instrument.GenericData(datacube_SPHERE_binned, centers_images, parangs=parangs_binned, wvs=None)
    # dataset.IWA = 12
    # dataset.OWA = 75

    dataset.old_centers = np.copy(dataset.centers)
    dataset.old_wcs = np.copy(dataset.wcs)
    dataset.old_PAs = np.copy(dataset.PAs)
    dataset.aligned_and_scaled = np.copy(dataset.input)

    parallelized.klip_dataset(dataset,
                              outputdir=output_dir,
                              fileprefix=output_prefix + "reduc_PCA",
                              numbasis=PCA_components,
                              annuli=1,
                              subsections=1,
                              movement=5,
                              mode="ADI",
                              restored_aligned=dataset.aligned_and_scaled,
                              lite=True,
                              verbose=False)

    print("end PCA")
    reduc_PCA = fits.getdata(os.path.join(output_dir, output_prefix + "reduc_PCA-KLmodes-all.fits"))

    return reduc_PCA


def subtract_classical_adi_median(datacube_init, output_dir=''):

    print("Start cADI")
    datacube_corr = np.zeros(datacube_init.shape)
    for i in range(datacube_init.shape[0]):
        datacube_here = np.copy(datacube_init)

        datacube_here[i] *= np.nan
        psf = np.nanmedian(datacube_here, axis=0)

        datacube_corr[i] = datacube_init[i] - psf

    print("End cADI")
    # fits.writeto(output_dir + "datacube_SPHERE_classical_adi_sub.fits",
    #              datacube_corr,
    #              overwrite=True)

    return datacube_corr


def derotate_and_mean_classical_adi(datacube_init, parang, output_dir=''):

    datacube_SPHERE_corr = datacube_init * 0.
    for i in range(len(parang)):
        datacube_SPHERE_corr[i] = ndimage.rotate(datacube_init[i], -parang[i], reshape=False)

    reduc_adi = np.nanmean(datacube_SPHERE_corr, axis=0)
    # fits.writeto(output_dir + "reduc_classical_adi.fits",
    #              reduc_adi,
    #              overwrite=True)

    return reduc_adi


######################################################
###### Measure positions functions
######################################################


def separation_planet(relative_RA, relative_dec):
    return np.sqrt((relative_RA)**2 + (relative_dec)**2)


def PA_planet(relative_RA, relative_dec):
    return np.degrees(np.arctan2(relative_dec, relative_RA))


def roundpupil(dim_pp, prad, xcenter, ycenter):
    """ --------------------------------------------------
    Create a circular aperture. The center of the aperture is located  in xcenter, ycenter

    Parameters
    ----------
    dim_pp : int
        Size of the image (in pixels)
    prad : float
        Size of the pupil radius (in pixels)


    Returns
    ------
    pupilnormal : 2D array
        Output circular pupil
    
    
    -------------------------------------------------- """

    xx, yy = np.meshgrid(np.arange(dim_pp) - xcenter, np.arange(dim_pp) - ycenter)
    rr = np.hypot(yy, xx)
    pupilnormal = np.zeros((dim_pp, dim_pp))
    pupilnormal[rr <= prad] = 1.0

    return pupilnormal


def Gaussian2d(xy, amplitude, sigma_x, sigma_y, xo, yo, theta, h, flatten=True):
    """ --------------------------------------------------
    Create a gaussian in 2D.
    
    # https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m

    Parameters
    ----------
    xy: Tuple object (2,dim1,dim2)  
        which can be created with:
        x, y = np.mgrid[0:dim1, 0:dim2]
        xy=(x,y)

    amplitude: float
        Peak of the gaussian function

    sigma_x: float
        Standard deviation of the gaussian function in the x direction

    sigma_y: float
        Standard deviation of the gaussian function in the y direction
    xo: float
        Position of the Gaussian peak in the x direction
    yo: float
        Position of the Gaussian peak in the y direction
    h: float
        Floor amplitude
    theta: float
        angle
    flatten : bool, default True
        if True (default), the 2D-array is flatten into 1D-array

    Returns
    ------
    gauss: 2d numpy array
        2D gaussian function



    -------------------------------------------------- """
    x = xy[0]
    y = xy[1]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x**2) + (np.sin(theta)**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta)**2) / (2 * sigma_x**2) + (np.cos(theta)**2) / (2 * sigma_y**2)
    g = (amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2))) + h)
    if flatten == True:
        g = g.flatten()
    return g


def run_gauss_fit(image, prior_x, prior_y):
    print("starting least square with prior position ({0},{1})".format(prior_x, prior_y))

    dim_im = image.shape[0]

    aperture_noise = roundpupil(dim_im, 10, prior_x - 2 * (prior_x - dim_im / 2), prior_y - 2 * (prior_y - dim_im / 2))
    noise = np.nanstd(image[np.where(aperture_noise == 1)])

    aperture_planet = roundpupil(dim_im, 10, prior_x, prior_y)

    image[np.where(image != image)] = 0.

    data = image * aperture_planet

    noise = data * 0 + noise

    # 2D-Gaussian fit
    popt = np.zeros(8)
    w, h = data.shape
    x, y = np.mgrid[0:w, 0:h]
    xy = (x, y)

    # Fit 2D Gaussian with fixed parameters
    initial_guess = (np.nanmax(data), 3, 3, prior_y, prior_x, 0, 0)

    # gauss = Gaussian2d(xy, np.nanmax(data), 1, 1,prior_y,  prior_x,  0, 0, flatten=False)
    # fits.writeto("/Users/jmazoyer/Desktop/data.fits", data, overwrite=True)
    # fits.writeto("/Users/jmazoyer/Desktop/gauss.fits", toto, overwrite=True)
    # asd

    try:
        popt, pcov = opt.curve_fit(Gaussian2d, xy, data.flatten(), sigma=noise.flatten(), p0=initial_guess)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        print("Error - curve_fit failed")

    position_in_pix = (popt[4], popt[3])
    position_in_pix_err = (perr[4], perr[3])

    bestGaussfit = Gaussian2d(xy, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], flatten=False)
    residuals = data - bestGaussfit
    # fits.writeto("/Users/jmazoyer/Desktop/data.fits", data, overwrite=True)
    # fits.writeto("/Users/jmazoyer/Desktop/bestGaussfit.fits", bestGaussfit, overwrite=True)
    # fits.writeto("/Users/jmazoyer/Desktop/residuals.fits", residuals, overwrite=True)
    return position_in_pix, position_in_pix_err


######################################################
###### fit orbits
######################################################


def write_orbit_in_cvs(filename, rows):

    #clear filename if exist
    open(filename, 'w').close()

    # open the file in the write mode
    with open(filename, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        for row in rows:
            writer.writerow(row)


def third_kepler_law(sma, star_mass):

    per = np.sqrt((4 * (np.pi**2) * (sma * u.AU)**3) / (consts.G * (star_mass * u.Msun)))
    period = per.to(u.year).value

    return period


def initialize_walkers(theta_init, num_temps, num_walkers):

    curr_pos = np.zeros((num_temps, num_walkers, len(theta_init)))

    for i in range(len(theta_init)):
        curr_pos[:, :, i] = np.random.uniform(0.999 * theta_init[i],
                                              1.001 * theta_init[i],
                                              size=(num_temps, num_walkers))

    return curr_pos
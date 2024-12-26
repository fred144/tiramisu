# %%
import sys

sys.path.append("../../")
from astropy.visualization import make_lupton_rgb
import yt
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.colors import LogNorm
from astropy import units as u
from astropy import constants as const
from tools import plotstyle
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from astropy.modeling.models import Sersic1D
import matplotlib
import glob
from tools.fscanner import filter_snapshots
import os
import scipy
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.size": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
    }
)

def snapshot_from_time(snapshots, time, split_sym="-", snap_idx=1, time_idx=2):
    """
    Given a list of postprocesed pop ii snapshot files, get the corresponding time

    Parameters
    ----------
    time : TYPE
        DESCRIPTION.
    snapshots : TYPE
        DESCRIPTION.
    split_sym : TYPE, optional
        DESCRIPTION. The default is "-".
    snap_idx : TYPE, optional
        DESCRIPTION. The default is 1.
    time_idx : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    """
    filepaths = snapshots
    uni_age = []
    snapshot = []
    for f in snapshots:
        name = os.path.basename(os.path.normpath(f))
        sn_numbers = float(name.split(split_sym)[snap_idx])
        tmyr = float(name.split(split_sym)[time_idx].replace("_", "."))

        uni_age.append(tmyr)
        snapshot.append(sn_numbers)

    uni_age = np.array([uni_age])
    snapshots = np.array(snapshot)
    residuals = np.abs(uni_age - np.array(time)[:, np.newaxis])
    closest_match_idxs = np.argmin(residuals, axis=1).astype(int)

    matching_snaps = snapshots[closest_match_idxs]
    matching_files = list(np.take(filepaths, closest_match_idxs))

    return matching_snaps, matching_files

def surface_brightness(x, y, lums, pw=400, star_bins=800):
    # plot width
    stellar_lumperbin, _, _ = np.histogram2d(
        x,
        y,
        bins=star_bins,
        weights=lums,
        range=[
            [-pw / 2, pw / 2],
            [-pw / 2, pw / 2],
        ],
    )
    pxl_area = (pw / star_bins) ** 2
    surface = stellar_lumperbin / pxl_area  # now has units of erg/s/AA/pc^2

    return surface

def mag_ab(x, y, lums, redshift, pw=400, star_bins=800, highest_mag=29.5):
    # plot width
    stellar_lumperbin, _, _ = np.histogram2d(
        x,
        y,
        bins=star_bins,
        weights=lums,
        range=[
            [-pw / 2, pw / 2],
            [-pw / 2, pw / 2],
        ],
    )
    pxl_area = (pw / star_bins) ** 2
    surface_brightness = stellar_lumperbin.T / pxl_area  # now has units of erg/s/AA/pc^2
    mu_ab = (
        23.4
        - 2.5
        * np.log10(
            surface_brightness / 1e36,
            out=np.full_like(surface_brightness, np.nan),
            where=(surface_brightness != 0),
        )
        + 10 * np.log10((1 + redshift) / 10)
    )  # mag/arcsec^2
    print(np.nanmax(mu_ab))
    mu_ab[np.isnan(mu_ab)] = highest_mag
    return mu_ab

def lum_lookup_filtered(
    stellar_ages: float,
    z,
    filter_name="F200W",
    stellar_masses=10,
    m_gal=1e6,
):
    """
    Computes luminosities from galaxy spectrum data using the given filter.

    Parameters
    ----------
    stellar_ages : float
        ages of the stars in Myr
    z : float
        redshift of the galaxy
    filter_name : str
        name of JWST filter to use, defaults to "F200W"
    table_file : str
        filepath to the table of spectrum data
    stellar_masses : float
        mass of the individual stars
    m_gal : TYPE, optional
        mass of the galaxy [Msun] from the starburst model. Default is 10^6 Msun

    Returns
    -------
    luminosities : array
        returns the luminosity of the individual stars, default UV luminosity

    """
    filter_data = np.loadtxt(
        "./silmaril_data/mean_throughputs/{}_mean_system_throughput.txt".format(
            filter_name
        ),
        skiprows=1,
    )

    wav_angs = (
        filter_data[:, 0] * 1e4 / (1 + z)
    )  # convert microns to angstroms and blueshift

    ages = np.concatenate(
        (range(1, 20), range(20, 100, 10), range(100, 1000, 100))
    )  # in Myr

    starburst = np.loadtxt("./silmaril_data/fig7e.dat", skiprows=3)

    starburst[:, 1:] = np.power(10, starburst[:, 1:])  # convert from log to linear

    mean_phot_rate = np.zeros(len(ages))  # initialize empty array

    for i in range(len(ages)):
        lum = np.interp(wav_angs, starburst[:, 0], starburst[:, i + 1])

        mean_phot_rate[i] = np.trapz(
            wav_angs * lum * filter_data[:, 1], wav_angs
        ) / np.trapz(wav_angs * filter_data[:, 1], wav_angs)

    lookup = scipy.interpolate.CubicSpline(ages, mean_phot_rate)

    return lookup(stellar_ages) * (stellar_masses / m_gal)

times = [512, 542, 552, 570,  580, 590, 595, 627, 659]

paths = filter_snapshots(
    "../../../container_tiramisu/post_processed/pop2/CC-Fiducial",
    153,
    466,
    1,
    snapshot_type="pop2_processed",
)
snap_nums, files = snapshot_from_time(paths, times)

path = "../../../container_tiramisu/post_processed/bsc_catalogues/CC-Fiducial"
field_width = [300, 300, 300, 300, 300, 300, 400, 400, 400,150, 150, 150]  # pc
mu_ab_limit = [24, 29.5]
fig, ax = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
plt.subplots_adjust(wspace=0, hspace=.004)
ax = ax.ravel()

axmin_1 = ax[0].inset_axes([0, 1.005, 0.50,0.50])
axmin_2 = ax[0].inset_axes([0.5, 1.005, 0.50, 0.50])
axmin_3 = ax[1].inset_axes([0.0, 1.005, 0.50, 0.50])
axmin_4 = ax[1].inset_axes([0.5, 1.005, 0.50, 0.50])
axmin_5 = ax[2].inset_axes([0.0, 1.005, 0.50, 0.50])
axmin_6 = ax[2].inset_axes([0.5, 1.005, 0.50, 0.50])

little_plots = [axmin_1, axmin_2, axmin_3, axmin_4, axmin_5, axmin_6]
ax = np.concatenate((little_plots,ax))



for i, pop2 in enumerate(files):
    snapshot = "info_00{:}".format(int(snap_nums[i]))
    clumped_cat = glob.glob(os.path.join(os.path.join(path, snapshot), "profiled*.txt"))
    clumped_dat = np.loadtxt(clumped_cat[0])
    clump_masses = clumped_dat[:, 8]
    # the bulge is the most massive
    bulge_id = clumped_dat[:, 0][np.argmax(clump_masses)]

    bulge_group_x = clumped_dat[:, 1][np.argmax(clump_masses)]
    bulge_group_y = clumped_dat[:, 2][np.argmax(clump_masses)]
    bulge_group_z = clumped_dat[:, 3][np.argmax(clump_masses)]

    full_dat = np.loadtxt(pop2)
    # parsec
    star_bins = 1000
    lum_erg_per_s_ang = 10 ** full_dat[:, 3]
    tmyr, redshift = full_dat[0:2, 0]
    all_pop2_mass = full_dat[:, -1]
    all_ages = full_dat[:, 2]
    creation_time = tmyr - all_ages

    allx, ally, allz = full_dat[:, 4:7].T

    # lum_F200W = lum_lookup_filtered(all_ages,z =redshift,filter_name="F200W")
    # lum_F300M =  lum_lookup_filtered(all_ages,z =redshift,filter_name="F300M")
    # lum_F480M = lum_lookup_filtered(all_ages,z =redshift,filter_name="F480M")
    # total_lum = lum_F200W + lum_F300M + lum_F480M

    # sb_b = surface_brightness(
    #     allx - bulge_group_x, ally - bulge_group_y, lum_F200W,  pw,star_bins
    # )
    # sb_g = surface_brightness(
    # allx - bulge_group_x, ally - bulge_group_y, lum_F300M,  pw,star_bins
    # )
    # sb_r = surface_brightness(
    # allx - bulge_group_x, ally - bulge_group_y,  lum_F480M,  pw,star_bins
    # )

    # mu_ab_F200W  = surface_brightness(
    #     allx - bulge_group_x, ally - bulge_group_y, lum_F200W, redshift, pw,star_bins
    # )
    # mu_ab_F480M  = surface_brightness(
    #     allx - bulge_group_x, ally - bulge_group_y, lum_F480M, redshift, pw,star_bins
    # )
    # image = make_lupton_rgb(sb_r, sb_g, sb_b, stretch=0.5)
    pw = field_width[i]
    mu_ab = mag_ab(
        allx - bulge_group_x,
        ally - bulge_group_y,
        lum_erg_per_s_ang,
        redshift,
        pw,
        star_bins,
    )

    img = ax[i].imshow(
        mu_ab,
        cmap="cmr.dusk_r",
        origin="lower",
        extent=[-pw / 2, pw / 2, -pw / 2, pw / 2],
        interpolation="gaussian",
        vmin=mu_ab_limit[0],
        vmax=mu_ab_limit[1],
    )
    ax[i].axis("off")
    ax[i].set_facecolor("k")
    
    if i > 5:
        time_txt= r"$t = {:.0f}\:{{\rm Myr}}$" "\n" r"$z={:.1f}$".format(tmyr, redshift)
        fontsize=11
    else:
        time_txt= r"$t = {:.0f}\:{{\rm Myr}}$".format(tmyr)
        fontsize=9
        
    ax[i].text(
        0.05,
        0.95,
        time_txt,
        va="top",
        ha="left",
        color="white",
        fontsize=fontsize,
        transform=ax[i].transAxes,
    )

    if (i == 0) or (i == 6):  # add scale bar
        scale = patches.Rectangle(
            xy=(pw / 2 * -0.85, -pw / 2 * 0.80),
            width=pw / 2 * 0.5,
            height=0.020 * pw / 2,
            linewidth=0,
            edgecolor="white",
            facecolor="white",
            clip_on=False,
            alpha=0.8,
        )
        if i ==0:
            f = 9 # fontsize
        else:
            f = 11
        ax[i].text(
            pw / 2 * -0.60,
            -pw / 2 * 0.90,
            r"$\mathrm{{{:.0f}\:pc}}$".format(pw / 2 * 0.5),
            ha="center",
            va="center",
            color="white",
            fontsize=f,
            
        )
        ax[i].add_patch(scale)

    # inset zoomed in images
    if  i == 6 or i == 8 :
        # pw = 150
        star_bins = 150
        if i == 6:
            zoom_ax = ax[i].inset_axes([0, -1.505, 1.5, 1.5])
            mark_inset(ax[i], zoom_ax, loc1=2, loc2=2, edgecolor="white", alpha=0.5, zorder=3,fc="None",ec="white")
        else:
            zoom_ax = ax[i].inset_axes([-0.5, -1.505, 1.5, 1.5])
            mark_inset(ax[i], zoom_ax, loc1=1, loc2=1, edgecolor="white", alpha=0.5)

        if i == 6:
            vmin, vmax = 25, 29
            ticks = [25.5, 26.5, 27.5, 28.5]
            contour_lvls = [25, 26, 27]
        else:
            vmin, vmax = 26, 30
            ticks = [26.5, 27.5, 28.5, 29.5]
            contour_lvls = [26, 27, 28]
        # plot the zoomed in image as well for certain snapshots
        mu_ab = mag_ab(
            allx - bulge_group_x,
            ally - bulge_group_y,
            lum_erg_per_s_ang,
            redshift,
            pw,
            star_bins,
            highest_mag=vmax,
        )
        zoom_img = zoom_ax.imshow(
            mu_ab,
            cmap="cmr.dusk_r",
            origin="lower",
            extent=[-pw / 2, pw / 2, -pw / 2, pw / 2],
            interpolation="gaussian",
            vmin=vmin,
            vmax=vmax,
        )
        zoom_ax.set(xlim=(-125 / 2, 125 / 2), ylim=(-125 / 2, 125 / 2))
        zoom_ax.axis("off")
        zoom_ax.set_facecolor("k")
        # less noisy contours
        mu_ab_undersample = mag_ab(
            allx - bulge_group_x,
            ally - bulge_group_y,
            lum_erg_per_s_ang,
            redshift,
            pw,
            150,
            highest_mag=vmax,
        )
        x = np.linspace(-0.5 * pw, 0.5 * pw, mu_ab_undersample.shape[0])
        y = np.linspace(-0.5 * pw, 0.5 * pw, mu_ab_undersample.shape[1])
        X, Y = np.meshgrid(x, y)
        Z = np.array(mu_ab_undersample)

        mag_contours = zoom_ax.contour(
            X, Y, Z, contour_lvls, origin="lower", colors="white"
        )
        clables = zoom_ax.clabel(mag_contours, inline=1, fontsize=10, colors="white")
        for label in clables:
            label.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1, foreground="white"),
                    path_effects.Normal(),
                ]
            )

        # put a color bar
        with plt.style.context("dark_background"):
            cbar_ax = zoom_ax.inset_axes([0.05, 0.1, 0.5, 0.05])
            cbar = fig.colorbar(
                zoom_img,
                cax=cbar_ax,
                pad=0,
                orientation="horizontal",
                ticks=ticks,
            )
            cbar.ax.xaxis.set_tick_params(pad=-14,labelsize=10)
            cbar.set_label(r"$\mu_{\rm AB} [{\rm mag\: arcsec^{-2}}]$", labelpad=1)
            cbar.ax.minorticks_on()
            
        if i==6:
            inner_r = patches.Circle((0,0), radius=9, color='cyan', fill=False,ls=":", linewidth=2, zorder=3)
            zoom_ax.add_patch(inner_r)

    
            
            outer_r = patches.Circle((0,0), radius=19, color='cyan', fill=False,ls="--", linewidth=2, zorder=3)
            zoom_ax.add_patch(outer_r )
            

            # Create a custom legend
            with plt.style.context("dark_background"):
                line1 = Line2D([0], [0], color='cyan', linewidth=2, label=r'$9$ pc', ls="--", )
                line2 = Line2D([0], [0], color='cyan', linewidth=2, label=r'$19$ pc', ls=":")
                zoom_ax.add_line(line1)
                zoom_ax.legend(handles=[line1,line2], loc='upper right', frameon=False, fontsize=12)
        else:
            rh= patches.Circle((0,0), radius=26, color='cyan', fill=False,ls="--", linewidth=2, zorder=3)
            zoom_ax.add_patch(rh)
            
            with plt.style.context("dark_background"):
                line1 = Line2D([0], [0], color='cyan', linewidth=2, label=r'$26$ pc', ls="--", )
                zoom_ax.add_line(line1)
                zoom_ax.legend(handles=[line1], loc='upper right', frameon=False, fontsize=12)
            


plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/nuclear_cluster.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01,
)
plt.show()

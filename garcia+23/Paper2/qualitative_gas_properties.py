import sys

sys.path.append("../../")
import yt
import numpy as np
import matplotlib.pyplot as plt
import os
from tools.fscanner import filter_snapshots
import glob
from matplotlib import cm
import matplotlib as mpl
from tools.ram_fields import ram_fields
from src.lum.lum_lookup import lum_look_up_table
from src.lum.pop2 import get_star_ages
from yt.fields.api import ValidateParameter
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.patches as patches
from tools.check_path import check_path
import cmasher as cmr
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
from matplotlib.ticker import LogFormatter

yt.enable_parallelism()
cell_fields, epf = ram_fields()

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 11,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.color": "white",
        "ytick.color": "white",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
    }
)
# plt.style.use("default")


def _my_radial_velocity(field, data):
    if data.has_field_parameter("bulk_velocity"):
        bv = data.get_field_parameter("bulk_velocity").in_units("km/s")
    else:
        bv = data.ds.arr(np.zeros(3), "km/s")

    # x_pos = np.array(data["star", "particle_position_x"])
    # y_pos = np.array(data["star", "particle_position_y"])
    # z_pos = np.array(data["star", "particle_position_z"])

    # ctr_at_code = np.array([np.mean(x_pos), np.mean(y_pos), np.mean(z_pos)])

    # sfregion = data.sphere(ctr_at_code, (r_sf, "pc"))
    # bulk_vel = sfregion.quantities.bulk_velocity()
    # # sfregion.set_field_parameter("bulk_velocity", bulk_vel)
    # bv = bulk_vel.in_units("km/s")

    xv = data["gas", "velocity_x"] - bv[0]
    yv = data["gas", "velocity_y"] - bv[1]
    zv = data["gas", "velocity_z"] - bv[2]
    center = data.get_field_parameter("center")
    x_hat = data["gas", "x"] - center[0]
    y_hat = data["gas", "y"] - center[1]
    z_hat = data["gas", "z"] - center[2]
    r = np.sqrt(x_hat * x_hat + y_hat * y_hat + z_hat * z_hat)
    x_hat /= r
    y_hat /= r
    z_hat /= r
    return xv * x_hat + yv * y_hat + zv * z_hat


yt.add_field(
    ("gas", "my_radial_velocity"),
    function=_my_radial_velocity,
    sampling_type="cell",
    units="km/s",
    take_log=False,
    validators=[ValidateParameter(["center", "bulk_velocity"])],
)


pw = 1000  # plot width on one side in pc
r_sf = 500  # radii for sf in pc
gas_res = 1000  # resolution of the fixed resolution buffer
star_bins = 2000
pxl_size = (pw / star_bins) ** 2
dens_norm = LogNorm(0.002, 2e4)
temp_norm = LogNorm(2e3, 8e6)
met_norm = LogNorm(9e-4, 5)
vrad_norm = colors.SymLogNorm(linthresh=0.1, linscale=1, vmin=-95, vmax=95)
stellar_dens_norm = LogNorm(2, 2e4)

zsun = 0.02
m_proton = 1.67e-24
k_boltz = 1.38e-16  # erg per kelvin
# plotting axis parameters
x, y = (9.5, 14)
xy_r = x / y
img_extent = [-pw / 2, pw / 2, -pw / 2, pw / 2]
dens_cmap = "cubehelix"
vrad_cmap = cmr.amethyst
temp_cmap = "inferno"
metal_cmap = cmr.bubblegum


snaps_cc, snap_strings_cc = filter_snapshots(
    os.path.expanduser("~/test_data/CC-Fiducial/"), 400, 400, sampling=1, str_snaps=True
)
snaps_70, snap_strings_70 = filter_snapshots(
    os.path.expanduser("~/test_data/fs07_refine/"),
    1397,
    1397,
    sampling=1,
    str_snaps=True,
)

snaps_35, snap_strings_35 = filter_snapshots(
    os.path.expanduser("~/test_data/fs035_ms10/"),
    1261,
    1261,
    sampling=1,
    str_snaps=True,
)

snaps = [
    (snaps_cc, snap_strings_cc),
    (snaps_70, snap_strings_70),
    (snaps_35, snap_strings_35),
]
names = [
    "VSFE",
    "high SFE",
    "low SFE",
    # r"$2.0 \times$ Lfid",
]
# %%

# ds = yt.load(
#     "/home/fabg/test_data/CC-Fiducial/output_00392/info_00392.txt",
#     fields=cell_fields,
#     extra_particle_fields=epf,
# )
# ad = ds.all_data()
# x_pos = np.array(ad["star", "particle_position_x"])
# y_pos = np.array(ad["star", "particle_position_y"])
# z_pos = np.array(ad["star", "particle_position_z"])
# pop2_masses = np.array(ad["star", "particle_mass"].to("Msun"))
# x_center = np.mean(x_pos)
# y_center = np.mean(y_pos)
# z_center = np.mean(z_pos)
# x_pos = x_pos - x_center
# y_pos = y_pos - y_center
# z_pos = z_pos - z_center

# ctr_at_code = np.array([x_center, y_center, z_center])
# yt.SlicePlot(
#     ds,
#     "z",
#     ("gas", "pressure"),
#     # weight_field=("gas", "density"),
#     center=ctr_at_code,
#     width=(pw, "pc"),
# ).save()
# %%
fig, ax = plt.subplots(
    nrows=3,
    ncols=1,
    sharex=True,
    sharey=True,
    figsize=(x, y),
    dpi=300,
    # facecolor=cm.Greys_r(0),
)
plt.subplots_adjust(hspace=0.01, wspace=0)
run_axes = ax.ravel()

for i, s in enumerate(snaps):
    ax = run_axes[i]
    for a, (sn, snap_strings) in enumerate(zip(s[0], s[1])):
        print("# ____________________________________________________________________")
        infofile = os.path.abspath(os.path.join(sn, f"info_{snap_strings}.txt"))
        print(infofile)
        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()

        t_myr = float(ds.current_time.in_units("Myr"))
        redshift = ds.current_redshift

        if len(np.array(ad["star", "particle_position_x"])) > 0:
            x_pos = np.array(ad["star", "particle_position_x"])
            y_pos = np.array(ad["star", "particle_position_y"])
            z_pos = np.array(ad["star", "particle_position_z"])
            pop2_masses = np.array(ad["star", "particle_mass"].to("Msun"))
            x_center = np.mean(x_pos)
            y_center = np.mean(y_pos)
            z_center = np.mean(z_pos)
            x_pos = x_pos - x_center
            y_pos = y_pos - y_center
            z_pos = z_pos - z_center

            ctr_at_code = np.array([x_center, y_center, z_center])
        else:
            print("no stars")
            _, ctr_at_code = ds.find_max(("gas", "density"))
            pop2_lums = np.zeros(100)
            star_mass = np.zeros(100)
            pop2_xyz = np.zeros((100, 3))

        # star_mass = np.ones_like(x_pos) * 10

        pop2_xyz = np.array(
            ds.arr(np.vstack([x_pos, y_pos, z_pos]), "code_length").to("pc")
        ).T
        # current_ages = get_star_ages(ram_ds=ds, ram_ad=ad, logsfc=logsfc_path)
        # pop2_lums = lum_look_up_table(
        #     stellar_ages=current_ages * 1e6,  # in myr
        #     stellar_masses=star_mass,
        #     table_link=os.path.join("..", "starburst", "l1500_inst_e.txt"),
        #     column_idx=1,
        #     log=False,
        # )

        stellar_mass_dens, _, _ = np.histogram2d(
            pop2_xyz[:, 0],
            pop2_xyz[:, 1],
            bins=star_bins,
            weights=pop2_masses,
            range=[[-pw / 2, pw / 2], [-pw / 2, pw / 2]],
        )
        stellar_mass_dens = stellar_mass_dens.T / pxl_size  # Msun / pc^2

        # to define the radial velocity, we define a region around the origin (star CoM)

        # sfregion = ds.sphere(ctr_at_code, (r_sf, "pc"))
        # bulk_vel = sfregion.quantities.bulk_velocity()
        # sfregion.set_field_parameter("bulk_velocity", bulk_vel)

        data_fields = [
            ("gas", "density"),
            # ("gas", "my_radial_velocity"),
            ("gas", "pressure"),
            ("ramses", "Metallicity"),
        ]
        prjctns = []

        for df in data_fields:
            p = yt.SlicePlot(ds, "z", fields=df, center=ctr_at_code, width=(pw, "pc"))

            p.set_buff_size(gas_res)
            p_frb = p.to_fits_data()
            p_img = np.array(p_frb[0].data)
            prjctns.append(p_img)

        # defining inset axes makes the aspect ratio more consistent and not have to
        # mess around with the figure size
        vax = ax.inset_axes([1.02, 0, 1, 1])
        tax = ax.inset_axes([2.04, 0, 1, 1])
        mex = ax.inset_axes([3.06, 0, 1, 1])
        axes = [ax, vax, tax, mex]

        dens = vax.imshow(
            np.array(ds.arr(prjctns[0], "g/cm**3")) / m_proton,
            cmap=dens_cmap,
            interpolation="gaussian",
            origin="lower",
            extent=img_extent,
            norm=dens_norm,
        )
        # vrad = vax.imshow(
        #     prjctns[1],
        #     cmap=vrad_cmap,
        #     interpolation="gaussian",
        #     origin="lower",
        #     extent=img_extent,
        #     norm=vrad_norm,
        # )
        # vax.scatter(0, 0, marker="x", color="white", s=30)

        sigma = ax.imshow(
            stellar_mass_dens,
            cmap=vrad_cmap,
            origin="lower",
            extent=img_extent,
            norm=stellar_dens_norm,
        )
        ax.set_facecolor("black")

        temp = tax.imshow(
            ds.arr(prjctns[1], "dyn/cm**2").in_units("erg/cm**3").value / k_boltz,
            cmap=temp_cmap,
            interpolation="gaussian",
            origin="lower",
            extent=img_extent,
            norm=temp_norm,
        )
        metal = mex.imshow(
            prjctns[2] / zsun,
            cmap=metal_cmap,
            interpolation="gaussian",
            origin="lower",
            extent=img_extent,
            norm=met_norm,
        )

        for a in axes:
            a.set(
                ylim=(-pw / 2, pw / 2),
                xlim=(xy_r * -pw / 2, xy_r * pw / 2),
                xticklabels=[],
                yticklabels=[],
            )
            a.spines["bottom"].set_color("black")
            a.spines["top"].set_color("black")
            a.spines["left"].set_color("black")
            a.spines["right"].set_color("black")
            # a.tick_params(axis="x", colors="white")
            # a.tick_params(axis="y", colors="white")

        ax.text(
            0.05,
            0.96,
            (r"t = {:.2f} Myr" "\n" r"z = {:.2f} ").format(
                t_myr,
                redshift,
            ),
            ha="left",
            va="top",
            color="white",
            transform=ax.transAxes,
        )

        ax.text(
            0.95,
            0.96,
            names[i],
            ha="right",
            va="top",
            color="white",
            transform=ax.transAxes,
        )

        if i == 0:
            scale = patches.Rectangle(
                xy=(-((pw / 2) * xy_r) / 3, -pw / 2 * 0.85),
                width=((pw / 2) * xy_r) / 1.5,
                height=0.010 * pw / 2,
                linewidth=0,
                edgecolor="white",
                facecolor="white",
            )
            vax.text(
                0,
                -pw / 2 * 0.90,
                r"$\mathrm{{{:.0f}\:pc}}$".format(((pw / 2) * xy_r) / 1.5),
                ha="center",
                va="center",
                color="white",
            )
            vax.add_patch(scale)

        if i == 2:
            formatter = LogFormatter(10, labelOnlyBase=False)
            dens_cbar_ax = vax.inset_axes([0, -0.05, 1, 0.04])
            dens_cbar_ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
            dens_cbar = fig.colorbar(
                dens,
                cax=dens_cbar_ax,
                orientation="horizontal",
                format=formatter,
                ticks=[0.1, 1, 10, 100, 1000],
            )
            dens_cbar_ax.set_xlabel(
                r"$\mathrm{\:n_{\rm H}}\:\mathrm{\left[cm^{-3}\right]}$"
            )
            dens_cbar_ax.yaxis.set_major_formatter("$10^{{{x:.0f}}}$")

            # tick_locator = ticker.MaxNLocator(nbins=10)
            # dens_cbar.locator = tick_locator
            # dens_cbar.update_ticks()

            vrad_cbar_ax = ax.inset_axes([0, -0.05, 1, 0.04])

            vrad_cbar = fig.colorbar(sigma, cax=vrad_cbar_ax, orientation="horizontal")
            vrad_cbar_ax.set_xlabel(
                # r"$\mathrm{Radial\:Velocity}\:\left[\mathrm{km\:s}^{-1}\right]$"
                r"$\Sigma_{\rm PopII}\:\mathrm{\left[M_\odot\:pc^{-2}\right]}$"
            )

            temp_cbar_ax = tax.inset_axes([0, -0.05, 1, 0.04])

            temp_cbar = fig.colorbar(temp, cax=temp_cbar_ax, orientation="horizontal")
            temp_cbar_ax.set_xlabel(r"$P/ k_{\rm B}\:\left[\mathrm{K\:cm^{-3}}\right]$")

            metal_cbar_ax = mex.inset_axes([0, -0.05, 1, 0.04])
            metal_cbar = fig.colorbar(
                metal, cax=metal_cbar_ax, orientation="horizontal"
            )
            metal_cbar_ax.set_xlabel(
                r"$\mathrm{Metallicity}\:\left[\mathrm{Z_\odot}\right]$"
            )


plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/mosaic.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()
# gd = yt.ProjectionPlot(
#     ds,
#     "z",
#     fields=("gas", "density"),
#     weight_field=("gas", "density"),
#     center=ctr_at_code,
#     width=(plt_wdth, "pc"),
# )
# gd.set_buff_size(gas_res)
# gd_frb = gd.to_fits_data()
# gd_img = np.array(gd_frb[0].data)

# gt = yt.ProjectionPlot(
#     ds,
#     "z",
#     fields=("gas", "temperature"),
#     weight_field=("gas", "density"),
#     center=ctr_at_code,
#     width=(plt_wdth, "pc"),
# )
# gt.set_buff_size(gas_res)
# gt_frb = gt.to_fits_data()
# gt_img = np.array(gt_frb[0].data)

# r_v.set_cmap(field=("gas", "my_radial_velocity"), cmap="RdBu")

# rv_frb = r_v.to_fits_data()
# gas_array = np.array(gas_frb[0].data)

# rv.save("test.png")

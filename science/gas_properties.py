import sys

sys.path.append("../")
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
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "font.size": 14,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
    }
)
plt.style.use("dark_background")


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

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(sys.argv[0], "usage:")
        print(
            "{} snapshot_dir start_snap end_snap step render_nickname".format(
                sys.argv[0]
            )
        )
        exit()
    else:
        print("********************************************************************")
        print(" rendering gas properties movie ")
        print("********************************************************************")

    datadir = sys.argv[1]
    # logsfc_path = sys.argv[2]
    start_snapshot = int(sys.argv[2])
    end_snapshot = int(sys.argv[3])
    step = int(sys.argv[4])
    render_nickname = sys.argv[5]

    sim_run = os.path.basename(os.path.normpath(datadir))
    fpaths, snums = filter_snapshots(
        datadir,
        start_snapshot,
        end_snapshot,
        sampling=step,
        str_snaps=True,
        snapshot_type="ramses_snapshot",
    )

    # datadir = os.path.expanduser("~/test_data/fid-broken-feedback/")
    # logsfc_path = os.path.expanduser(
    #     "~/container_tiramisu/sim_log_files/CC-Fiducial/logSFC"
    # )
    # fpaths, snums = filter_snapshots(
    #     datadir,
    #     390,
    #     390,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )
    # render_nickname = "CC-Fid"

    # =============================================================================
    #                         timelapse paramaters
    # =============================================================================

    pw = 1000  # plot width on one side in pc
    r_sf = 500  # radii for sf in pc
    gas_res = 1000  # resolution of the fixed resolution buffer
    dens_norm = LogNorm(0.008, 1)
    temp_norm = LogNorm(100, 1e6)
    met_norm = LogNorm(2e-3, 0.20)
    vrad_norm = colors.SymLogNorm(linthresh=0.1, linscale=1, vmin=-95, vmax=95)
    zsun = 0.02
    # plotting axis parameters
    x, y = (5, 10)
    xy_r = x / y
    img_extent = [-pw / 2, pw / 2, -pw / 2, pw / 2]
    dens_cmap = "cubehelix"
    vrad_cmap = cmr.pride_r
    temp_cmap = "inferno"
    metal_cmap = cmr.torch

    # run save
    sim_run = os.path.basename(os.path.normpath(datadir))
    render_container = os.path.join(
        "..",
        "..",
        "container_tiramisu",
        "renders",
        sim_run,
        render_nickname,
    )
    check_path(render_container)

    for i, fp in enumerate(fpaths):
        print("# ____________________________________________________________________")
        infofile = os.path.abspath(os.path.join(fp, f"info_{snums[i]}.txt"))
        print(infofile)
        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()

        t_myr = float(ds.current_time.in_units("Myr"))
        redshift = ds.current_redshift

        if len(np.array(ad["star", "particle_position_x"])) > 0:
            x_pos = np.array(ad["star", "particle_position_x"])
            y_pos = np.array(ad["star", "particle_position_y"])
            z_pos = np.array(ad["star", "particle_position_z"])
            x_center = np.mean(x_pos)
            y_center = np.mean(y_pos)
            z_center = np.mean(z_pos)
            x_pos = x_pos - x_center
            y_pos = y_pos - y_center
            z_pos = z_pos - z_center

            ctr_at_code = np.array([x_center, y_center, z_center])
        else:
            _, ctr_at_code = ds.find_max(("gas", "density"))

        # star_mass = np.ones_like(x_pos) * 10

        # pop2_xyz = np.array(
        #     ds.arr(np.vstack([x_pos, y_pos, z_pos]), "code_length").to("pc")
        # ).T
        # current_ages = get_star_ages(ram_ds=ds, ram_ad=ad, logsfc=logsfc_path)
        # pop2_lums = lum_look_up_table(
        #     stellar_ages=current_ages * 1e6,  # in myr
        #     stellar_masses=star_mass,
        #     table_link=os.path.join("..", "starburst", "l1500_inst_e.txt"),
        #     column_idx=1,
        #     log=False,
        # )
        # to define the radial velocity, we define a region around the origin (star CoM)

        sfregion = ds.sphere(ctr_at_code, (r_sf, "pc"))
        bulk_vel = sfregion.quantities.bulk_velocity()
        sfregion.set_field_parameter("bulk_velocity", bulk_vel)

        data_fields = [
            ("gas", "density"),
            ("gas", "my_radial_velocity"),
            ("gas", "temperature"),
            ("ramses", "Metallicity"),
        ]
        prjctns = []

        for df in data_fields:
            if df == ("gas", "density"):
                p = yt.ProjectionPlot(
                    ds, "z", fields=df, center=ctr_at_code, width=(pw, "pc")
                )
            else:
                p = yt.ProjectionPlot(
                    ds,
                    "z",
                    fields=df,
                    weight_field=("gas", "density"),
                    center=ctr_at_code,
                    width=(pw, "pc"),
                )

            p.set_buff_size(gas_res)
            p_frb = p.to_fits_data()
            p_img = np.array(p_frb[0].data)
            prjctns.append(p_img)

        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(x, y),
            dpi=400,
            facecolor=cm.Greys_r(0),
        )

        # defining inset axes makes the aspect ratio more consistent and not have to
        # mess around with the figure size
        vax = ax.inset_axes([1.05, 0, 1, 1])
        tax = ax.inset_axes([2.10, 0, 1, 1])
        mex = ax.inset_axes([3.15, 0, 1, 1])
        axes = [ax, vax, tax, mex]

        dens = ax.imshow(
            prjctns[0],
            cmap=dens_cmap,
            interpolation="gaussian",
            origin="lower",
            extent=img_extent,
            norm=dens_norm,
        )
        vrad = vax.imshow(
            prjctns[1],
            cmap=vrad_cmap,
            interpolation="gaussian",
            origin="lower",
            extent=img_extent,
            norm=vrad_norm,
        )
        vax.scatter(0, 0, marker="x", color="white", s=30)
        temp = tax.imshow(
            prjctns[2],
            cmap=temp_cmap,
            interpolation="gaussian",
            origin="lower",
            extent=img_extent,
            norm=temp_norm,
        )
        metal = mex.imshow(
            prjctns[3] / zsun,
            cmap=metal_cmap,
            interpolation="gaussian",
            origin="lower",
            extent=img_extent,
            norm=met_norm,
        )

        dens_cbar_ax = ax.inset_axes([0, -0.07, 1, 0.04])
        dens_cbar = fig.colorbar(dens, cax=dens_cbar_ax, orientation="horizontal")
        dens_cbar_ax.set_xlabel(
            r"$\mathrm{Gas\:Surface\:Density}\:\mathrm{\left[g\:cm^{-2}\right]}$"
        )

        vrad_cbar_ax = vax.inset_axes([0, -0.07, 1, 0.04])
        vrad_cbar = fig.colorbar(vrad, cax=vrad_cbar_ax, orientation="horizontal")
        vrad_cbar_ax.set_xlabel(
            r"$\mathrm{Radial\:Velocity}\:\left[\mathrm{km\:s}^{-1}\right]$"
        )

        temp_cbar_ax = tax.inset_axes([0, -0.07, 1, 0.04])
        temp_cbar = fig.colorbar(temp, cax=temp_cbar_ax, orientation="horizontal")
        temp_cbar_ax.set_xlabel(r"$\mathrm{Temperature}\:\left[\mathrm{K}\right]$")

        metal_cbar_ax = mex.inset_axes([0, -0.07, 1, 0.04])
        metal_cbar = fig.colorbar(metal, cax=metal_cbar_ax, orientation="horizontal")
        metal_cbar_ax.set_xlabel(
            r"$\mathrm{Metallicity}\:\left[\mathrm{Z_\odot}\right]$"
        )

        for a in axes:
            a.set(
                ylim=(-pw / 2, pw / 2),
                xlim=(xy_r * -pw / 2, xy_r * pw / 2),
                xticklabels=[],
                yticklabels=[],
            )

        scale = patches.Rectangle(
            xy=(-((pw / 2) * xy_r) / 3, -pw / 2 * 0.85),
            width=((pw / 2) * xy_r) / 1.5,
            height=0.010 * pw / 2,
            linewidth=0,
            edgecolor="white",
            facecolor="white",
        )
        ax.text(
            0,
            -pw / 2 * 0.90,
            r"$\mathrm{{{:.0f}\:pc}}$".format(((pw / 2) * xy_r) / 1.5),
            ha="center",
            va="center",
            color="white",
        )
        ax.add_patch(scale)

        ax.text(
            0.05,
            0.96,
            ("{}" "\n" r"t = {:.2f} Myr" "\n" r"z = {:.2f} ").format(
                render_nickname,
                t_myr,
                redshift,
            ),
            ha="left",
            va="top",
            color="white",
            transform=ax.transAxes,
        )

        output_path = os.path.join(
            render_container, "{}-{}.png".format(render_nickname, snums[i])
        )
        plt.savefig(
            os.path.expanduser(output_path),
            # os.path.join(".", "{}.png".format(snums[i])),
            dpi=300,
            bbox_inches="tight"
            # pad_inches=0.05,
        )


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

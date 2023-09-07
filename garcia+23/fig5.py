import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.optimize import curve_fit
import matplotlib.lines as mlines
import matplotlib
import os
from tools import plotstyle
from labellines import labelLines

"""
Graph for plotting the star formation efficeicney as a function of molecular 
cloud mass for a given mean number density of hydrogen
"""


def star_formation_efficiency(n_h: float, mass: float, metallicity: float):
    """
    derive expected SFE for constant SFE runs

    Parameters
    ----------
    n_h : float
        mean hydrogen number density for a cloud
    mass : float
        mass of the cloud
    metallicity : float
        metalicity of the cloud

    Returns
    -------
    efficiency : TYPE
        gas-to-star conversion efficiency

    """
    # n_crit=n_H*0+1.e3/12.0
    # with shell 6 times less dense. At t_relax 12 times less dense
    # f_s reduced by 5 for low met.
    # f_s increases with stronger B-field
    n_crit = 100.0  # (n_h * 0 + 1.0e3) / (4.0 * 2)

    efficiency = (
        (2.0e-2 / 5.0)
        * (mass / 1.0e4) ** 0.4
        * (n_h / n_crit + 1.0) ** (0.91)
        * (metallicity / 1e-3) ** 0.25
    )
    # f_s=4.e-3*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
    efficiency = np.where(efficiency < 0.9, efficiency, 0.9)
    return efficiency


# run = "../../container_tiramisu/sim_log_files/cc-kazu-run"
# run_name = run.split("/")[-1]

# fs070_log_sfc = np.loadtxt(os.path.join(run, "logSFC"))

# # fs070_log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")
# redshft_fs070 = fs070_log_sfc[:, 2]
# r_pc_cloud_fs070 = fs070_log_sfc[:, 4]
# m_sun_cloud_fs070 = fs070_log_sfc[:, 5]
# n_hydrogen_fs070 = fs070_log_sfc[:, 8]
# metal_zun_cloud_fs070 = fs070_log_sfc[:, 9]
# mstar_cloud = fs070_log_sfc[:, 7]
# fs035_log_sfc = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")
# redshft_fs035 = fs035_log_sfc[:, 2]
# r_pc_cloud_fs035 = fs035_log_sfc[:, 4]
# m_sun_cloud_fs035 = fs035_log_sfc[:, 5]
# n_hydrogen_fs035 = fs035_log_sfc[:, 8]
# metal_zun_cloud_fs035 = fs035_log_sfc[:, 9]


def plotting_interface(run_logpath, simulation_name, marker, sfe: str):
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.25), dpi=400)
    latest_redshift = 5

    for i, r in enumerate(run_logpath):
        run_name = os.path.basename(os.path.normpath(r))
        print(os.path.join(r, "logSFC"))
        log_sfc = np.loadtxt(os.path.join(r, "logSFC"))

        redshft = log_sfc[:, 2]
        mask = redshft > latest_redshift
        redshft = redshft[mask]
        r_pc_cloud = log_sfc[:, 4][mask]
        m_sun_cloud = log_sfc[:, 5][mask]
        m_sun_stars = log_sfc[:, 7][mask]
        n_hydrogen = log_sfc[:, 8][mask]
        metal_zsun_cloud = log_sfc[:, 9][mask]  # metalicity is normalized to z_sun

        if sfe[i] == "constant":
            sfe_val = (
                star_formation_efficiency(n_hydrogen, m_sun_cloud, metal_zsun_cloud)
                * 100
            )
        elif sfe[i] == "variable":
            sfe_val = (m_sun_stars / m_sun_cloud) * 100
        else:
            print("sfe is ether constant or variable")
            raise ValueError

        sfe_scatter = ax.scatter(
            m_sun_cloud,
            sfe_val,
            c=np.log10(n_hydrogen),
            label=simulation_name[i],
            cmap="summer",
            marker=marker[i],
            edgecolors="black",
            linewidth=0.5,
            s=40,
            alpha=0.8,
        )

    # ax.scatter(
    #     m_sun_cloud_fs035,
    #     star_formation_efficiency(
    #         n_hydrogen_fs035, m_sun_cloud_fs035, metal_zun_cloud_fs035
    #     )
    #     * 100,
    #     c=np.log10(n_hydrogen_fs035),
    #     label=r"$f_{*} = 0.35 $",
    #     cmap=cmap,
    #     marker="P",
    #     edgecolors="black",
    #     linewidth=0.5,
    #     s=40,
    #     alpha=0.8,
    # )
    # cbar = plt.colorbar(pad=0, orientation = 'horizontal')
    # cbar.set_label(
    #     label=(r"$\log_{10}\:\overline{n_\mathrm{H}}\:\left(\mathrm{cm}^{-3} \right)$"),
    #     fontsize=14,
    # )

    cbar_ax = ax.inset_axes([0, 1.02, 1, 0.05])
    dens_bar = fig.colorbar(sfe_scatter, cax=cbar_ax, pad=0, orientation="horizontal")

    dens_bar.set_label(
        label=(r"$\log_{10}\:\overline{n_\mathrm{H}}\:\left(\mathrm{cm}^{-3} \right)$"),
        fontsize=12,
        labelpad=6,
    )
    dens_bar.ax.xaxis.set_label_position("top")
    dens_bar.ax.xaxis.set_ticks_position("top")
    dens_bar.ax.locator_params(nbins=6)

    ax.set(
        xlabel=r"$M_{\rm MC} (\mathrm{M}_{\odot})$",
        ylabel=r"$\mathrm{SFE}\:(\%)$",
        xscale="log",
        yscale="log",
    )

    h = []
    for n, m in zip(simulation_name, marker):
        sim_label = mlines.Line2D(
            [],
            [],
            color="k",
            marker=m,
            ls="",
            label=n,
        )
        h.append(sim_label)

    ax.legend(loc="lower right", handles=h)

    ax.set_ylim(top=120)
    xmin, _ = ax.get_xlim()
    ax.axhline(y=90, color="grey", ls="--", zorder=1)
    ax.annotate("$90 \%$", (xmin * 1.2, 75), color="grey")

    ax.axhline(y=10, color="grey", ls="--", zorder=1)
    ax.annotate("$10 \%$", (xmin * 1.2, 8), color="grey")

    # ax.tick_params(axis="y", direction="in", which="both")
    # ax.tick_params(axis="x", direction="in", which="both")
    # # manual legend, want to customize colors
    # f70 = mlines.Line2D([], [], color="k", marker="o", ls="", label=r"$f_{*} = 0.70 $")
    # f35 = mlines.Line2D([], [], color="k", marker="P", ls="", label=r"$f_{*} = 0.35 $")
    # ax.legend(
    #     loc="lower right",
    #     title_fontsize=12,
    #     fontsize=12,
    #     handles=[f35, f70],
    # )


if __name__ == "__main__":
    # cmap = matplotlib.colormaps["Set2"]
    # cmap = cmap(np.linspace(0, 1, 8))

    runs = [
        "../../container_tiramisu/sim_log_files/fs07_refine",
        # "../../container_tiramisu/sim_log_files/fs035_ms10",
        "../../container_tiramisu/sim_log_files/CC-Fiducial",
    ]
    names = [
        "$f_* = 0.70$",
        # "$f_* = 0.35$",
        r"${\rm He+19}$",
    ]
    markers = [
        "o",
        # "P",
        "v",
    ]
    calc_type = [
        "constant",
        # "constant",
        "variable",
    ]

    plotting_interface(
        run_logpath=runs, simulation_name=names, marker=markers, sfe=calc_type
    )

    plt.show()
    # plt.savefig(
    #     "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/lowres/sfc_mass_sfe.png",
    #     dpi=300,
    #     bbox_inches="tight",
    #     pad_inches=0.05,
    # )
    # plt.savefig(
    #     "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/sfc_mass_sfe.png",
    #     dpi=400,
    #     bbox_inches="tight",
    #     pad_inches=0.05,
    # )

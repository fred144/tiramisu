import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.optimize import curve_fit
import matplotlib.lines as mlines
import matplotlib as mpl
import os

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
    }
)


"""
initial mass functions and metaliccity function for the molecular clouds or
star forming clouds.
"""
latest_redshift = 8.0

run = "../../container_tiramisu/sim_log_files/CC-radius1"
run_name = run.split("/")[-1]

log_sfc = np.loadtxt(os.path.join(run, "logSFC"))

redshft = log_sfc[:, 2]
mask = redshft > latest_redshift
redshft = redshft[mask]
r_pc_cloud = log_sfc[:, 4][mask]
m_sun_cloud = log_sfc[:, 5][mask]
n_hydrogen = log_sfc[:, 8][mask]
metal_zsun_cloud = log_sfc[:, 9][mask]  # metalicity is normalized to z_sun


# fs070_log_sfc = np.loadtxt("../../container_tiramisu/sim_log_files/fs07_refine/logSFC")
# redshft = fs070_log_sfc[:, 2]
# mask = redshft > latest_redshift
# redshft = redshft[mask]
# r_pc_cloud = fs070_log_sfc[:, 4][mask]
# m_sun_cloud = fs070_log_sfc[:, 5][mask]
# n_hydrogen = fs070_log_sfc[:, 8][mask]
# metal_cloud = fs070_log_sfc[:, 9][mask]

# fs035_log_sfc = np.loadtxt("../sim_log_files/fs035_ms10/logSFC")
# redshft_fs035 = fs035_log_sfc[:, 2]
# mask = redshft_fs035 > latest_redshift
# redshft_fs035 = redshft_fs035[mask]
# r_pc_cloud_fs035 = fs035_log_sfc[:, 4][mask]
# m_sun_cloud_fs035 = fs035_log_sfc[:, 5][mask]
# n_hydrogen_fs035 = fs035_log_sfc[:, 8][mask]
# metal_cloud_fs035 = fs035_log_sfc[:, 9][mask]

# print("Total mass in MCs for 70%", np.sum(m_sun_cloud))
# print("Total mass in MCs for 35%", np.sum(m_sun_cloud_fs035))


def gauss(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def bimodal(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    return amp1 * np.exp(-0.5 * ((x - mean1) / sigma1) ** 2) + amp2 * np.exp(
        -0.5 * ((x - mean2) / sigma2) ** 2
    )


def log_data_function(data, num_bins, bin_range: tuple):
    bin_range = np.log10(bin_range)
    log_data = np.log10(data)
    count, bin_edges = np.histogram(log_data, num_bins, bin_range)
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]
    bin_ctrs = 0.5 * (left_edges + right_edges)
    # normalize with width of the bins
    counts_per_log_solar_mass = count / (right_edges - left_edges)

    return 10**bin_ctrs, counts_per_log_solar_mass


cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))

color = cmap[1]


mass_xrange = (2e3, 4e5)
bns = 15
metal_xrange = (1.5e-4, 1e-2)
radius_xrange = np.arange(1, 14, 1)

# mass function
mass, counts = log_data_function(m_sun_cloud, bns, mass_xrange)
fit_params, _ = curve_fit(
    f=gauss, xdata=np.nan_to_num(np.log10(mass), neginf=0), ydata=counts
)
mass_x = np.log10(np.geomspace(mass.min(), mass.max(), 100))
mass_y = gauss(mass_x, *fit_params)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), dpi=500)

# f70_leg = mlines.Line2D(
#     [], [], color=color, ls="-", lw=4, label="$f_{*} = 0.70$"
# )
# f35_leg = mlines.Line2D(
#     [], [], color=fs35_color, ls="-", lw=4, label="$f_{*} =0.35$"
# )
# leg_title = mlines.Line2D(
#     [], [], color="white", ls="", label="$\mathrm{SFE} \: (f_{*})$"
# )
# leg_title = mlines.Line2D(
#     [],
#     [],
#     color="white",
#     ls="",
#     label="$\mathrm{{z = {:.2f}}}$".format(
#         np.min(np.concatenate([redshft, redshft_fs035]))
#     ),
# )
# leg = fig.legend(
#     title="$\mathrm{{z = {:.2f}}}$".format(
#         np.min(np.concatenate([redshft, redshft_fs035]))
#     ),
#     loc="upper left",
#     handles=[f70_leg, f35_leg],
#     bbox_to_anchor=(0.40, 0.89),
#     ncol=1,
#     # edgecolor="grey",
#     # fontsize=10,
# )
# leg.get_frame().set_boxstyle("Square")

ax[0].plot(mass, counts, drawstyle="steps-mid", linewidth=2.5, alpha=0.8, color=color)
ax[0].fill_between(
    mass,
    counts,
    step="mid",
    alpha=0.4,
    color=color,
)
# plot the fits
ax[0].plot(
    10**mass_x,
    mass_y,
    ls=":",
    linewidth=2,
    alpha=1,
    color="k",
    label=(r"$ ({:.1f}, {:.1f})$").format(fit_params[1], np.abs(fit_params[2])),
)
ax[0].set_xlabel(
    r"$  \mathrm{M_{MC}} \:\:  \left( \mathrm{M}_{\odot} \right) $",
)
ax[0].set_ylabel(
    r"$\mathrm{dN / d\log} \:\: \left(\mathrm{M_{MC}}/\mathrm{M}_{\odot}\right )$",
    labelpad=2,
)

# ax[0].set_xlim(mass_xrange[0], mass_xrange[1])
ax[0].set_ylim(1, 4e4)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].legend(
    title=r"$\log_{{10}}\:(\mu,\:\Sigma)$",
    loc="upper left",
)
#%%metalicitty function
z, z_counts = log_data_function(metal_zsun_cloud, bns, metal_xrange)
ax[1].plot(
    z,
    z_counts,
    drawstyle="steps-mid",
    linewidth=2.5,
    alpha=0.8,
    color=color,
    label=r"${{\rm {}}}$".format(run_name),
)
ax[1].fill_between(
    z,
    z_counts,
    step="mid",
    alpha=0.4,
    color=color,
)
ax[1].legend(
    title=r"$\mathrm{{z = {:.2f}}}$".format(np.min(np.concatenate([redshft]))),
    fontsize=10,
)
ax[1].set_xlabel(
    r"$\mathrm{Z_{MC}} \:\:  \left( \mathrm{Z}_{\odot} \right) $",
)
ax[1].set_ylabel(
    r"$\mathrm{dN / d\log} \:\: \left(\mathrm{Z_{MC}}/\mathrm{Z}_{\odot}\right )$",
    labelpad=2,
)

# ax[1].yaxis.tick_right()
# ax[1].yaxis.set_label_position("right")

# ax[1].set_xlim(metal_xrange[0], metal_xrange[1] + 0.001)
ax[1].set_ylim(1, 4e4)
ax[1].set_xscale("log")
ax[1].set_yscale("log")

#%% cloud radius mass functions.
count, bin_edges = np.histogram(r_pc_cloud, bins=radius_xrange, density=True)
right_edges = bin_edges[1:]
left_edges = bin_edges[:-1]
bin_ctrs = 0.5 * (left_edges + right_edges)

ax[2].plot(
    bin_ctrs,
    count,
    drawstyle="steps-mid",
    linewidth=2.5,
    alpha=0.8,
    color=color,
    label=r"$\mu = {:.2f}$".format(np.mean(r_pc_cloud)),
)
ax[2].fill_between(
    bin_ctrs,
    count,
    step="mid",
    alpha=0.4,
    color=color,
)
ax[2].set(xlabel=r"$\mathrm{R_{MC} \: (pc)}$", ylabel=r"$\mathrm{PDF \: (R_{MC})}$")
ax[2].set_ylim(bottom=0)
ax[2].legend()
# ax[1].legend(
#     title=r"$\log_{{10}}\:(\mu,\:\sigma)$",
#     loc="upper center",
#     ncol=2,
#     # fontsize=10,
#     # title_fontsize=10,
# )
# plt.legend(
#     title="$\mathrm{SFE} \: (f_{*})$",
#     loc="upper left",
# )


plt.subplots_adjust(hspace=0, wspace=0.32)
# plt.savefig(
#     "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/lowres/sfc_mfunc.png",
#     dpi=300,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.savefig(
#     "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/sfc_mfunc.png",
#     dpi=500,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )

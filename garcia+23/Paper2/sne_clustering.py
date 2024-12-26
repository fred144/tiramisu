#%%
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
from tools.cosmo import t_myr_from_z
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from tools.ram_fields import ram_fields

#%%
snaps_cc, snap_strings_cc = filter_snapshots(
    os.path.expanduser("~/container_tiramisu/post_processed/pop2/CC-Fiducial/"),
    400,
    400,
    sampling=1,
    str_snaps=True,
    snapshot_type="pop2_processed",
)

snaps_70, snap_strings_70 = filter_snapshots(
    os.path.expanduser("~/container_tiramisu/post_processed/pop2/fs07_refine/"),
    1397,
    1397,
    sampling=1,
    str_snaps=True,
    snapshot_type="pop2_processed",
)

snaps_35, snap_strings_35 = filter_snapshots(
    os.path.expanduser("~/container_tiramisu/post_processed/pop2/fs035_ms10/"),
    1261,
    1261,
    sampling=1,
    str_snaps=True,
    snapshot_type="pop2_processed",
)


def read_sne(logsfc_path, interuped=False):
    log_sfc = np.loadtxt(logsfc_path)
    if interuped is True:
        log_sfc_1 = np.loadtxt(logsfc_path + "-earlier")
        log_sfc = np.concatenate((log_sfc_1, log_sfc), axis=0)

    redshift = log_sfc[:, 2]

    t_myr = t_myr_from_z(redshift)
    # SNe properties
    # m_ejecta = log_sfc[:, 6]
    # e_thermal_injected = log_sfc[:, 7]
    # ejecta_zsun = log_sfc[:, 8]
    # let's do the accumulation of metals produced
    # mass_in_metals = m_ejecta * ejecta_zsun
    # total_mass_in_metals = np.cumsum(mass_in_metals)
    position = log_sfc[:, 12:16]

    # return t_myr, total_mass_in_metals
# %% try to see how the code units change with time
cell_fields, epf = ram_fields()

fpaths, snums = filter_snapshots(
    os.path.expanduser("~/test_data/CC-Fiducial/"),
    153,
    466,
    sampling=1,
    str_snaps=True,
    snapshot_type="ramses_snapshot",
)

tmyr_list = []
zred_list = []
code_length_in_pc = []

for i, sn in enumerate(fpaths):
    print("# ________________________________________________________________________")
    infofile = os.path.abspath(os.path.join(sn, f"info_{snums[i]}.txt"))
    print("# reading in", infofile)
    try:
        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()
    except:
        print("having trouble reading snapshot, skipping")
        continue

    x_pos = ad["star", "particle_position_x"].to("pc")
    y_pos = ad["star", "particle_position_y"].to("pc")
    z_pos = ad["star", "particle_position_z"].to("pc")
    x_center = np.mean(x_pos)
    y_center = np.mean(y_pos)
    z_center = np.mean(z_pos)
    plt_ctr = ds.arr([x_center, y_center, z_center], "pc")

    z = ds.current_redshift
    tmyr = np.round(ds.current_time.in_units("Myr").value, 1) # current time

    
    code_length_to_pc = ds.length_unit.to("pc")
    
    tmyr_list.append(tmyr)
    zred_list.append(z)
    code_length_in_pc.append(code_length_to_pc)
    
    print(code_length_to_pc, "is the length unit in pc")
    path = os.path.expanduser("~/container_tiramisu/sim_log_files/CC-Fiducial/logSFC")
    log_sfc = np.loadtxt(os.path.expanduser(path))
    position = log_sfc[:, -3:]
    sfc_formation_time = t_myr_from_z(log_sfc[:, 2])
    
    # get the sfcs that formed around a buffer time in this snapshot
    tbuffer = 1
    sfc_mask  = (sfc_formation_time > tmyr - tbuffer) & (sfc_formation_time <= tmyr)
    

    sfc_pos_pc = ds.arr(position[sfc_mask], "code_length").to("pc") 

#%% make an interpolation of the length unit in pc as a function of time 
tmyr_list = np.array(tmyr_list)
code_length_in_pc = np.array(code_length_in_pc)

spl_length = splrep(tmyr_list, code_length_in_pc, k=1)  # k=3 for cubic spline
t_dense = np.linspace(tmyr_list[0], tmyr_list[-1], 100)
code_length_dense = splev(t_dense, spl_length)

fig,ax = plt.subplots(dpi=200, figsize=(12, 6), nrows=1, ncols=2)
ax[0].scatter(zred_list, code_length_in_pc, s=5)
ax[0].set( xlabel="redshift", ylabel="Length unit [pc]", xlim=(13, 8))

ax[1].scatter(tmyr_list, code_length_in_pc, s=5)

ax[1].plot(t_dense, code_length_dense, color="k", alpha=0.5, label="Interpolated Length unit")
ax[1].set( xlabel="Time [Myr]", ylabel="Length unit [pc]", yscale="log")
# try a powerlaw just to see if it fits the points
t_try = np.linspace(tmyr_list[0], tmyr_list[-1] + 500, 100)
powerlaw = lambda x, amp, index: amp * (x ** index)

from scipy.optimize import curve_fit
popt, pcov = curve_fit(powerlaw, tmyr_list, code_length_in_pc, p0=[1, 1])
ax[1].plot(t_try, powerlaw(t_try, *popt),  color="r", alpha=0.5, linestyle="--", label=r"norm ={:.2f}, index={:.2f}".format(*popt))

# just a straight up power law with index 2/3
power_law_norm = code_length_in_pc[0] / (tmyr_list[0] ** (2/3))
ax[1].plot(t_try, powerlaw(t_try, power_law_norm, 2/3), color="k", alpha=0.2, lw = 5, label="Powerlaw index 2/3")

ax[1].legend()
plt.show()


# %% read in the popII centers over time and spline interpolate where they are

def code_length_to_pc(t):
    """just a hard code of a 2/3 power law that looks okay above"""
    power_law_norm = 71925.69335739232
    return  power_law_norm * (t ** (2/3))

pop2_cc, pop2_strings_cc = filter_snapshots(
    os.path.expanduser("~/container_tiramisu/post_processed/pop2/CC-Fiducial/"),
    163,
    466,
    sampling=1,
    str_snaps=True,
    snapshot_type="pop2_processed",
)

sfc_path = os.path.expanduser("~/container_tiramisu/sim_log_files/CC-Fiducial/logSFC")
log_sfc = np.loadtxt(os.path.expanduser(sfc_path))
sfc_position = log_sfc[:, -3:] # in code units
sfc_formation_time = t_myr_from_z(log_sfc[:, 2])
# multiply accross the row
# sfc_position_pc =code_length_to_pc(sfc_formation_time) * sfc_position

# mask out sfcs too early formed
sfc_mask = sfc_formation_time > sfc_formation_time[0]
sfc_position = sfc_position[sfc_mask]
sfc_formation_time = sfc_formation_time[sfc_mask]

x_center = []
y_center = []
z_center = []
sim_myr = []

for i, pop2path in enumerate(pop2_cc):
    full_dat = np.loadtxt(pop2path)
    tmyr, redshift = full_dat[0:2, 0]
    all_pop2_mass = full_dat[:, -1]
    all_ages = full_dat[:, 2]
    allx, ally, allz = full_dat[:, 4:7].T
    lum_erg_per_s_ang = 10 ** full_dat[:, 3]
    creation_time = tmyr - all_ages

    plt_ctr_code_units =  full_dat[2:5,0]
    code_to_pc = code_length_to_pc(tmyr)
    plt_ctr_pc = plt_ctr_code_units * code_to_pc
    
    # now that we have the center of the galaxy in code units, we can convert it to physical units in pc using the fit we made earlier
    
    
    # x_center.append(plt_ctr_pc[0])
    # y_center.append(plt_ctr_pc[1])
    # z_center.append(plt_ctr_pc[2])
    
    x_center.append(plt_ctr_code_units[0])
    y_center.append(plt_ctr_code_units[1])
    z_center.append(plt_ctr_code_units[2])
    
    
    sim_myr.append(tmyr)

#%% spline test

# data from the simulation
x_center = np.array(x_center)
y_center = np.array(y_center)
sim_myr = np.array(sim_myr)

sfc_formation_time_mask = (sfc_formation_time > sim_myr[0]) & (sfc_formation_time < sim_myr[-1])

# Create spline representations for x(t) and y(t)
spl_x = splrep(sim_myr, x_center, k=3)  # k=3 for cubic spline
spl_y = splrep(sim_myr, y_center, k=3)
spl_z = splrep(sim_myr, z_center, k=3)

# Generate a dense range of time points for smooth plotting
t_dense = np.linspace(sim_myr[0], sim_myr[-1], 100)
# Interpolate x and y for dense time points
x_dense = splev(t_dense, spl_x)
y_dense = splev(t_dense, spl_y)

fig, ax = plt.subplots(1,2,dpi=200, figsize=(12, 5))

# plotting, butmask out early and late sfcs
early_sf = sfc_formation_time < 550 # Myr
early_sf1 = sim_myr < 550
spline_time_mask = t_dense < 550


# from snapshots
ax[0].scatter(x_center[early_sf1], y_center[early_sf1], marker="o", color="r", s=1, zorder=10, label="galaxy center from snapshots")
ax[0].plot(x_dense[spline_time_mask ], y_dense[spline_time_mask ], label='Interpolated Path (Spline)', alpha=0.5)
ax[0].scatter(sfc_position[:, 0][early_sf], sfc_position[:, 1][early_sf],  c=sfc_formation_time[early_sf], cmap="viridis", s=10, alpha=0.5)

# now do after 550 Myr
ax[1].scatter(x_center[~early_sf1], y_center[~early_sf1], marker="o",  color="r",   s=1, zorder=10)
ax[1].plot(x_dense[~spline_time_mask ], y_dense[~spline_time_mask], label='Interpolated Path (Spline)', alpha=0.5)
ax[1].scatter(sfc_position[:, 0][~early_sf], sfc_position[:, 1][~early_sf],  c=sfc_formation_time[~early_sf], cmap="viridis", s=10, alpha=0.5)

x_center_at_sfc_form = splev(sfc_formation_time, spl_x)
y_center_at_sfc_form = splev(sfc_formation_time, spl_y)
z_center_at_sfc_form = splev(sfc_formation_time, spl_z)


ax[0].scatter(x_center_at_sfc_form[early_sf], y_center_at_sfc_form[early_sf],  s=1, color="black", label="sfc Interpolated Center")
ax[1].scatter(x_center_at_sfc_form[~early_sf], y_center_at_sfc_form[~early_sf],  s=1, color="black", label="Interpolated Center")


ax[0].legend()
plt.show()
# ax[1].scatter(x_center_interp, y_center_interp, marker="x", c=sfc_formation_time[sfc_formation_time_mask], cmap="viridis", s=1)
# ax[0].set(yscale="log", xscale="log")

# now, we can calculate the distance between the galaxy center at the formation of sfc and the sfc position
sfc_distance = np.sqrt((x_center_at_sfc_form - sfc_position[:, 0])**2 + (y_center_at_sfc_form - sfc_position[:, 1])**2 + (z_center_at_sfc_form - sfc_position[:, 2])**2) 

# can also recenter the sfc positions based on the interpolated center
sfc_position_recentered_x = sfc_position[:, 0] - x_center_at_sfc_form
sfc_position_recentered_y = sfc_position[:, 1] - y_center_at_sfc_form
sfc_position_recentered_z = sfc_position[:, 2] - z_center_at_sfc_form

# now we can plot the distance between the galaxy center at the formation of the sfc and the sfc position

#additionally, we can also plot it in pc units
fig, ax = plt.subplots(dpi=200, figsize=(6, 6)) 

sfc_code_to_pc = code_length_to_pc(sfc_formation_time)

sfc_scatter = ax.scatter(sfc_position_recentered_x*sfc_code_to_pc , sfc_position_recentered_y*sfc_code_to_pc , c=sfc_formation_time, cmap="viridis", s=10)
ax.set(xlim=(-200, 200), ylim=(-200, 200), xlabel="physical x [pc]", ylabel="physical y [pc]")
# make a colorbar with the time of formation
cbar_ax = ax.inset_axes([1, 0.0, 0.05, 1])
cbar = fig.colorbar(sfc_scatter, cax=cbar_ax)
cbar.set_label("Time of formation [Myr]")
ax.text(0.5, 0.9, "SFC position w.r.t galaxy center of mass", transform=ax.transAxes, ha="center")

# plot the distance as a function of time

fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
ax.scatter(sfc_formation_time, sfc_distance*sfc_code_to_pc, s=1)
ax.set(xlabel="Time [Myr]", ylabel="Distance from center [pc]")




    
#%% proof of concept for spline

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

# Example data
t = np.array([0, 1, 2, 3, 4])  # Time points
# x = np.array([0, 1, 0, -1, 0])  # X coordinates
# y = np.array([0, 0, 1, 0, -1])  # Y coordinates
x = np.linspace(-10, 10, 5)
y = np.array([0, 0, 1, 0, -1])  # Y coordinates

# Create spline representations for x(t) and y(t)
spl_x = splrep(t, x, k=3)  # k=3 for cubic spline
spl_y = splrep(t, y, k=3)

# Generate a dense range of time points for smooth plotting
t_dense = np.linspace(t[0], t[-1], 500)

# Interpolate x and y for dense time points
x_dense = splev(t_dense, spl_x)
y_dense = splev(t_dense, spl_y)

# Plot the original points and the interpolated path
t_query = 3.25
x_query, y_query = splev(t_query, spl_x), splev(t_query, spl_y)

# Array of lookup points
t_query_array = np.array([0.5, 1.5, 2.5, 3.25, 3.75])
x_query_array = splev(t_query_array, spl_x)
y_query_array = splev(t_query_array, spl_y)

# Plot the original points, interpolated path, and the array of lookup points
plt.figure(figsize=(8, 6))
plt.plot(x_dense, y_dense, label='Interpolated Path (Spline)', color='blue')
plt.scatter(x, y, color='red', label='Original Points', zorder=5)
plt.scatter(x_query_array, y_query_array, color='green', label='Lookup Points', zorder=6, s=100, edgecolor='black')

# Annotate the lookup points
for i, (tq, xq, yq) in enumerate(zip(t_query_array, x_query_array, y_query_array)):
    plt.annotate(f't={tq:.2f}', (xq, yq), textcoords="offset points", xytext=(10, 10), ha='center')

plt.title('Path Reconstruction with Spline and Multiple Lookup Points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.axis('equal')  # Maintain equal aspect ratio for x and y axes
plt.show()




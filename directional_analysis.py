
import os
import sys
import time
import random
import json
import math
from tqdm import tqdm
import copy

import pandas as pd

from scipy.stats import poisson
from scipy.special import gamma
from scipy.optimize import curve_fit
import scipy.integrate as spi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# define functions for fits
def gaussian(x, amplitude, mean, stddev):
	return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)
def abs_sin(x, yoffset, amplitude, offset):
	return yoffset + amplitude * np.abs(np.sin( (np.pi/360.0) * (x - offset) ))
def sin_sq(x, yoffset, amplitude, xoffset):
	return yoffset + amplitude * np.square(np.sin( (np.pi/360.0) * (x - xoffset) ))
# define data filtering function for neutron captures on Li-6
def filterCaptures(unfiltered_data, truth_data):
	pdg_codes = [1.00002004e+09, 1.00001003e+09, 2112]
	# identify tracks containing He-4 and H-3 nuclei (byproducts of Li-6 capture)
	valid_rows = unfiltered_data.groupby("Row")["trackPDG"].apply(
		lambda pdgs: (pdg_codes[0] in pdgs.values) and (pdg_codes[1] in pdgs.values)
	)
	# select sets of neutron tracks containing Li-6 byproducts
	filtered_data = unfiltered_data[(unfiltered_data["Row"].isin(valid_rows[valid_rows].index))]
	truth_data = truth_data[(truth_data["Row"].isin(valid_rows[valid_rows].index))]
	# remove Li-6 byproducts, keeping only neutrons for neutron events
	filtered_data = filtered_data[filtered_data["trackPDG"] == pdg_codes[2]]
	# return data sets of filtered (neutron and MC truth) events
	return filtered_data, truth_data
# define functions to read truth.txt, neutrons.txt and positrons.txt
def readTruthData(truth_file):
	truth_data = pd.read_csv(truth_file, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
	return truth_data
def readNeutrons(neutron_file, truth_data_set):
	neutron_data = pd.read_csv(neutron_file, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
	neutron_data, truth_data = filterCaptures(neutron_data, truth_data_set)
	return neutron_data, truth_data
def readPositrons(positron_file):
	positron_data = pd.read_csv(positron_file, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
	return positron_data
# define binning function
def binEvents(filtered_data, truth_data, theta, seg_size, matrix_size, plot=False):
	# Initializes matrix for segmentation.
	seg = np.zeros((matrix_size, matrix_size), dtype=dict)
	# Size of half of the total detector grid in mm.
	half_size = seg_size * matrix_size / 2.0
	# Initialize binning matrix for specified segment size and matrix size
	for i in range(matrix_size):
		for j in range(matrix_size):
				xlow  = seg_size * j - half_size
				xhigh = seg_size * j + seg_size - half_size
				yhigh = -(seg_size * i - half_size)
				ylow  = -(seg_size * i + seg_size - half_size)
				seg[i][j] = {
						"xbounds": [xlow, xhigh],
						"ybounds": [ylow, yhigh],
						"counts":  0
				}
	# get prompt-delayed displacement coordinates for binning
	x_coords = filtered_data["trackPosX"] - truth_data["mcx"]
	y_coords = filtered_data["trackPosY"] - truth_data["mcy"]
	z_coords = filtered_data["trackPosZ"] - truth_data["mcz"]
	# bin neutron captures
	for x, y, z in zip(x_coords, y_coords, z_coords):
		# Generate a random point in the center square segment.
		xrand = random.uniform(-seg_size / 2.0, seg_size / 2.0)
		yrand = random.uniform(-seg_size / 2.0, seg_size / 2.0)
		# Distance calculation in 2d.
		r = np.sqrt(x**2 + y**2)
		# Initial angle of capture.
		theta0 = np.arctan2(y, x)
		# Convert rotation angle to radians
		phi = theta * np.pi / 180.0
		# Coordinate transformation.
		xprime = r * np.cos(theta0 - phi)
		yprime = r * np.sin(theta0 - phi)
		# Randomize the rotated event's location in the center square segment.
		xc = xprime + xrand
		yc = yprime + yrand
		# Bin the event
		for i in range(matrix_size):
			for j in range(matrix_size):
				s = seg[i][j]
				if (s["xbounds"][0] < xc) and (xc < s["xbounds"][1]) and \
				(s["ybounds"][0] < yc) and (yc < s["ybounds"][1]):
					s["counts"] += 1
	caps = np.zeros((matrix_size, matrix_size), dtype=int)
	for i in range(matrix_size):
		for j in range(matrix_size):
			caps[i][j] = seg[i][j]["counts"]
	return caps
def angleFit(nbins, seg_size, num_events, it_num, save_output=True, save_plots=True, output_path="fits/", show_plots=False, debug=False):
	# read data files
	truth_angle = 0 # truth angle of "data"
	data_range = 180 # range of data
	th_range = [truth_angle - data_range, truth_angle + data_range]
	# path to import "data" file
	pre_str = "/home/jack/RATPAC2/ratpac-setup/ratpac/output/mtv_directionality/res"
	num_ev_str = {10: "_10", 100: "_100", 1000: "_1k", 10000: "_10k"}
	dop_str = "_001wt"
	run_type_str = {"fixed": "_fixed", "ref": "_ref"}
	data_type_str = {"truth": "/truth.txt", "neutrons": "/neutrons.txt"}
	fixed_str = pre_str + num_ev_str[num_events] + dop_str + run_type_str["fixed"]
	# read truth.txt and neutrons.txt for fixed sets. Filter neutron events for captures on Li-6
	fixed_set_truth = readTruthData(fixed_str + data_type_str["truth"])
	fixed_set_neutrons, fixed_set_truth = readNeutrons(fixed_str + data_type_str["neutrons"], fixed_set_truth)
	# get neutron capture events and IBD vertices for fixed data set
	locs = fixed_set_neutrons.groupby("Row").last()
	truth_locs = fixed_set_truth.groupby("Row").first()# keep only the first entry. MC values don't change
	# bin fixed location events and normalize binning
	caps_fixed = binEvents(locs, truth_locs, truth_angle, seg_size, nbins)
	caps_fixed = caps_fixed/np.sum(caps_fixed)	
	# read reference data set and filter neutron events for captures on Li-6
	ref_str = pre_str + num_ev_str[10000] + dop_str + run_type_str["ref"]
	ref_set_truth = readTruthData(ref_str + data_type_str["truth"])
	ref_set_neutrons, ref_set_truth = readNeutrons(ref_str + data_type_str["neutrons"], ref_set_truth)
	ref_locs = ref_set_neutrons.groupby("Row").last()# keep only captures
	ref_truth_locs = ref_set_truth.groupby("Row").first()# keep only first entry of MC events
	data_str = f"Calculating FND fit #{it_num} for {fixed_str} and {ref_str} ..."
	if (debug):
		print(data_str)
	# Iterate over reference angles to minimize FND for |sin(th-th0)| fit
	angles = []
	norms = []
	for th in range(-data_range,data_range):
		angles.append(th)
		caps_ref = binEvents(ref_locs, ref_truth_locs, th, seg_size, nbins)
		caps_ref = caps_ref/np.sum(caps_ref)
		fnd_sq = np.sum(np.square(caps_ref - caps_fixed))
		fnd = np.sqrt(fnd_sq)
		norms.append(fnd)
		#print(f"Angle (deg:) {theta}\t FND value: {fnd:.2f}")
	# set up the curve fit to norms or sum_sq
	fit_range = 179
	# use argmin of norms as initial guess for fit angle
	angle_guess = angles[np.argmin(norms)]
	filename = f"segsz_{seg_size}mm_FND_fit{num_ev_str[num_events]}.txt"
	out_str1 = "Initial guess for fit #"
	out_str1 += f"{it_num} in {num_events} events dataset: {angle_guess:.0f}\nFND value at initial guess: {norms[np.argmin(norms)]:.2f}\n"
	if (save_output):
		fstring = output_path + filename
		with open(fstring, "a") as f:
			print(out_str1, file=f)
	elif (debug):
		print(filename)
		print(out_str1)
	# do |sin(th-th0)| fit---TODO: needs status bar for angles being fit
	sin_y_offset = np.min(norms)
	sin_fit_norms = np.array([(fn - sin_y_offset) for fn in norms])
	sin_fit_angles = angles[(truth_angle - th_range[0] - fit_range):-(th_range[1] - truth_angle - fit_range)]
	sin_fit_norms = sin_fit_norms[(truth_angle - th_range[0] - fit_range):-(th_range[1] - truth_angle - fit_range)]
	# fit params and covariance
	sin_popt, sin_pcov = curve_fit(abs_sin, sin_fit_angles, sin_fit_norms, p0=[sin_y_offset, np.max(sin_fit_norms), angle_guess])
	# output fit params and covariance matrix to a file
	sin_y_fit = abs_sin(sin_fit_angles, sin_popt[0], sin_popt[1], sin_popt[2])
	sin_y_errors = np.sqrt(np.diag(sin_pcov))
	renorms = norms / np.max(sin_fit_norms)
	# do plots and save if called for
	fig2, ax2 = plt.subplots(figsize=(6, 4))
	ax2.set_xlabel("$\\vartheta$ (${}^\\circ$)")
	# function, y axis labels, and plot
	out_str2 = "Fit parameters for |sin(x-x_0)| fit #"
	out_str2 += f"{it_num} in {num_events} events dataset:\n[y_offset, amplitude, angle (degrees:)] {sin_popt}\nerror: {sin_y_errors}\n"
	if (save_output):
		filename = f"FND_fit{num_ev_str[num_events]}.txt"
		fstring = output_path + filename
		with open(fstring, "a") as f:
			print(out_str2, file = f)
	elif (debug):
		print(out_str2)
	fnd_label = "Frobenius norm"
	fit_label = "$\\alpha|\\sin(\\vartheta-\\theta_0)|$\n(fit: $\\theta_0 = "
	if(num_events==10):
		fit_label += f"{sin_popt[2]:.1f}\\pm{sin_y_errors[2]:.1f}" + "{}^\\circ$\n" + f"($\\alpha = {sin_popt[1]:.3f}\\pm{sin_y_errors[1]:.3f}$))"
	else:
		fit_label += f"{sin_popt[2]:.1f}\\pm{sin_y_errors[2]:.1f}" + "{}^\\circ$\n" + f"($\\alpha = {sin_popt[1]:.3f}\\pm{sin_y_errors[1]:.4f}$))"
	ax2.set_ylabel("FND")
	ax2.plot(sin_fit_angles, [y + sin_y_offset for y in sin_y_fit], color="red", label=fit_label)
	ax2.plot(angles, norms, "b.", label=fnd_label, alpha=0.5)
	# finish the plot
	ax2.axvline(truth_angle, linestyle="--", color="gray")
	ax2.legend(loc="upper left", framealpha=1.0)
	fig2.tight_layout()
	if (show_plots):
		plt.show()
	if (save_plots):
		plt.savefig(output_path + f"/plots/angle_{truth_angle}_deg{num_ev_str[num_events]}_evts_fit_{it_num}.pdf", format="pdf")
	plt.close()
	return sin_popt[2]
# TODO: GUI and QOL improvements. Do money plot for 5, 50 and 150 mm segment sizes using 9, 5, and 3 matrix sizes, respectively
# TODO: For GUI, specify reference file (menu option) and data file (menu option), segment size (checkboxes, allowing all requiring at least one), pushbutton to calculate fits to data file, pushbutton to do resolution analysis
# TODO: QOL improvement -- first time data is loaded for reference set, save it to a JSON. GUI option/prompt to load from JSON on start, or specify new reference set
# segment size in mm
# TODO: GUI/QOL improvement -- text box to specify number of iterations to perform for resolution analysis using statistical approach, radio button options to use either statistical or covariance matrix to determine resolution from fits
segment_size = [150, 50, 5]
# matrix size for n by n binning matrix -- events are binned by counting the number of segments between prompt and delayed vertices
grid_size = [3, 5, 9]
seg_style = {150: "b--", 50: "b-.", 5: "b:"}
fit_res = {150: [], 50: [], 5: []}
write_outputs = False
keep_plots = False
debug_msg = True
out_file_path = "fits/"
iterations = 30
n_evts = [10, 100, 1000, 10000]
dir_str = {10: "moneyplot_10_evt_fits/", 100: "moneyplot_100_evt_fits/", 1000: "moneyplot_1k_evt_fits/", 10000: "moneyplot_10k_evt_fits/"}
plt.rc('font', family='serif', size = 14)
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'cm' # computer modern
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.set_xlabel("Number of events")
ax1.set_xscale("log")
ax1.set_ylabel("Angular resolution (${}^\\circ$)")
for k, sz in enumerate(segment_size):
	fit_angle = {10: [], 100: [], 1000: [], 10000: []}
	fit_means = {10: 0, 100: 0, 1000: 0, 10000: 0}
	fit_stds = {10: 0, 100: 0, 1000: 0, 10000: 0}
	for j, num in enumerate(n_evts):
		if (debug_msg):
			print(f"Calculating fits and angular resolution for {num} events in {segment_size[k]}mm segments with {iterations} iterations...\n")
		out_file_str = f"seg_size_{sz}_resolution_{num}_evts.txt"
		for i in range(iterations):
			angle = angleFit(grid_size[k], sz, num, i, write_outputs, keep_plots, out_file_path + dir_str[num], False, True)
			fit_angle[num].append(angle)
		fit_means[num] = np.mean(fit_angle[num])
		fit_stds[num] = np.std(fit_angle[num])
		fit_res[sz].append(fit_stds[num])
		out_str = f"Fit value and angular resolution for {segment_size[k]}mm after {iterations} iterations on {num} events (deg.)\nfit: {fit_means[num]} \tres: {fit_stds[num]}\n"
		if (write_outputs):
			with open(out_file_path + dir_str[num] + out_file_str, "w") as f:
				print(out_str, file=f)
		elif (debug_msg):
			print(out_str)
	if (write_outputs):
		summary_file_str = f"angular_res_summary_{segment_size[k]}mm_segments.txt"
		with open(out_file_path + summary_file_str, "w") as f:
			print(f"Fit angles (deg:)\n{fit_angle}", file=f)
			print(f"Fit values (deg:)\n{fit_means}", file=f)
			print(f"Fit resolutions (deg:)\n{fit_stds}", file=f)
	print(f"Fit angles for {segment_size[k]}mm segents (deg:)\n{fit_angle}")
	print(f"Fit values for {segment_size[k]}mm segents (deg:)\n{fit_means}")
	print(f"Fit resolutions for {segment_size[k]}mm segents (deg:)\n{fit_stds}")
	ax1.plot(n_evts, fit_res[sz], seg_style[sz], label = f"{segment_size[k]}mm")
# show me the money!!!
ax1.legend(loc="upper right", framealpha=1.0)
fig.tight_layout()
plt.show()
if (keep_plots):
	plt.savefig(output_path + "moneyplot.pdf", format="pdf")
plt.close()
# dumping ground for unused code

# conditional handling of significant figures on plots goes before else statement on line 185:

"""
	elif(num_events==100):
		fit_label += f"{sin_popt[2]:.1f}\\pm{sin_y_errors[2]:.1f}" + "{}^\\circ$\n" + f"($\\alpha = {sin_popt[1]:.4f}\\pm{sin_y_errors[1]:.4f}$))"
	elif(num_events==1000):
		fit_label += f"{sin_popt[2]:.1f}\\pm{sin_y_errors[2]:.1f}" + "{}^\\circ$\n" + f"($\\alpha = {sin_popt[1]:.4f}\\pm{sin_y_errors[1]:.4f}$))"
	elif(num_events==10000):
		fit_label += f"{sin_popt[2]:.1f}\\pm{sin_y_errors[2]:.1f}" + "{}^\\circ$\n" + f"($\\alpha = {sin_popt[1]:.4f}\\pm{sin_y_errors[1]:.4f}$))"
"""

# goes before truth_angle declaration on line 107:

"""
# fit type -- valid types are "|sin|" and "sin**2"
fit_type = "|sin|"
# plot type -- valid types are "fit" and "func"
plot_type = "fit"
# simulations for reference sets
refs = ["_000", "_001", "_002", "_003", "_004", "_005", "_006", "_007", "_008", "_009",
	"_010", "_011", "_012", "_013", "_014", "_015", "_016", "_017", "_018", "_019",
	"_020", "_021", "_022", "_023", "_024", "_025", "_026", "_027", "_028", "_029"
]
# number of reference sets for 1k event runs
n_sets = 30
"""

# related to previous block -- goes before ref_str declaration on line 127:

"""
if (num_events == "1000"):
	set_ind = int(random.uniform(0, n_sets-1))
	ref_ind_string = refs[set_ind]
	ref_str = pre_str + num_ev_str[num_events] + dop_str + run_type_str["ref"] + ref_ind_string + file_str
else:
	
"""

# old fitters for sin**2 fit and for analytical function plots taken from after plot declaration for |sin| fit in line 172:

# note: if re-inserted, line indents of previous statements must be fixed to match conditionals, and the following line must be inserted uncommented before the out_str2 declaration:
#if ( fit_type == "|sin|" ):
"""
	elif ( plot_type == "func" ):
		ax1.plot(sin_fit_angles, sin_y_vals, color = "green", label=func_label)
		ax1.plot(angles, renorms, "b.", label=fnd_label, alpha=0.5)
		ax2.plot(sin_fit_angles, sin_y_func_res, "y.", label=res_label, alpha=0.3)
	else:
		print("Error: valid types are either \"fit\" or \"func.\" Exiting...")
		sys.exit()
elif ( fit_type == "sin**2" ):
	out_str2 = "fit parameters for sin**2(x-x_0) fit:\n[y_offset, amplitude, angle (degrees:)]" + f" {sin_sq_popt}\nerror: {sin_sq_y_errors}\n"
	fnd_label = "Frobenius norm squared"
	fit_label = f"$\\alpha\\sin^2(\\vartheta-\\theta_0)$\n(fit: $\\theta_0 = {sin_sq_popt[2]:.1f}\\pm{sin_sq_y_errors[2]:.1f}" + "{}^\\circ$\n" + f"($\\alpha = {sin_sq_popt[1]:.4f}\\pm{sin_sq_y_errors[1]:.4f}$))"
	func_label = "$\\sin^2(\\vartheta-\\theta_0)$"
	res_label = "difference between\n$\\alpha\\sin^2(\\vartheta-\\theta_0)$ and FND${}^2$"
	ax1.set_ylabel("FND${}^2$")
	if ( plot_type == "fit" ):
		ax1.plot(sin_fit_angles, [y + sin_sq_y_offset for y in sin_sq_y_fit], color="red", label=fit_label)
		ax1.plot(angles, sum_sq, "b.", label=fnd_label, alpha=0.5)
		ax2.plot(sin_fit_angles, sin_sq_y_res, "y.", label=res_label, alpha=0.3)
	elif ( plot_type == "func" ):
		ax1.plot(sin_fit_angles, sin_sq_y_vals, color = "green", label=func_label)
		ax1.plot(angles, ress, "b.", label=fnd_label, alpha=0.5)
		ax2.plot(sin_fit_angles, sin_sq_y_func_res, "y.", label=res_label, alpha=0.3)
	else:
		print("Error: valid types are either \"fit\" or \"func.\" Exiting...")
		sys.exit()
else:
	print("Error: valid fit types are either \"|sin|\" or \"sin**2.\" Exiting...")
	sys.exit()
"""

# lists for sin**2 fit (declare after |sin| fit params)

"""
# fit params and covariance -- sin**2
sin_sq_popt, sin_sq_pcov = curve_fit(sin_sq, sin_fit_angles, sin_sq_fit_norms, p0=[sin_sq_y_offset, np.max(sin_sq_fit_norms), angle_guess])
sin_sq_y_fit = sin_sq(sin_fit_angles, sin_sq_popt[0], sin_sq_popt[1], sin_sq_popt[2])
sin_sq_y_errors = np.sqrt(np.diag(sin_sq_pcov))
"""

# renorm for sin**2 fit, residuals and analytical functions (declare after renorms declaration)

"""
ress = sum_sq / np.max(sin_sq_fit_norms)
sin_y_vals = np.array([abs_sin(x, 0, 1, truth_angle) for x in angles])
sin_sq_y_vals = np.array([sin_sq(x, 0, 1, truth_angle) for x in angles])
# difference between norms and value of abs(sin) function, or between sum_sq and sin**2
sin_y_res = sin_y_fit[:] - sin_fit_norms[:]
sin_y_func_res = sin_y_vals[:] - renorms[:]
sin_sq_y_res = sin_sq_y_fit[:] - sin_sq_fit_norms[:]
sin_sq_y_func_res = sin_sq_y_vals[:] - renorms[:]
sin_y_vals = sin_y_vals[(truth_angle - th_range[0] - fit_range):-(th_range[1] - truth_angle - fit_range)]
sin_y_func_res = sin_y_func_res[(truth_angle - th_range[0] - fit_range):-(th_range[1] - truth_angle - fit_range)]
sin_sq_y_vals = sin_sq_y_vals[(truth_angle - th_range[0] - fit_range):-(th_range[1] - truth_angle - fit_range)]
sin_sq_y_func_res = sin_sq_y_func_res[(truth_angle - th_range[0] - fit_range):-(th_range[1] - truth_angle - fit_range)]
"""

# unused axis object for residuals plot
#ax2=ax1.twinx()
#ax2.set_ylabel("fit resisduals")

# unused plot labels and objects (function plots, residual plots)
#func_label = "$|\\sin(\\vartheta-\\theta_0)|$"
#res_label = "difference between\n$\\alpha|\\sin(\\vartheta-\\theta_0)|$ and FND"
	#if ( plot_type == "fit" ):
#ax2.plot(sin_fit_angles, sin_y_res, "y.", label=res_label, alpha=0.3)
#ax2.legend(loc="upper right", framealpha=1.0)

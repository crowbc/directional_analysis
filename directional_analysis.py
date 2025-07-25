
#import os
#import sys
#import time
import random
#import json
import math
#from tqdm import tqdm
#import copy

import pandas as pd

#from scipy.stats import poisson
#from scipy.special import gamma
from scipy.optimize import curve_fit
#import scipy.integrate as spi

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator

# define functions for fits -- gaussian and sin_sq not used TODO: remove if they remain unused in future edits
#def gaussian(x, amplitude, mean, stddev):
	#return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)
def abs_sin(x, yoffset, amplitude, xoffset):
	return yoffset + amplitude * np.abs(np.sin( (np.pi/360.0) * (x - xoffset) ))
#def sin_sq(x, yoffset, amplitude, xoffset):
	#return yoffset + amplitude * np.square(np.sin( (np.pi/360.0) * (x - xoffset) ))
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
def readNeutrons(neutron_file, truth_data_set, filt_caps=True):
	neutron_data = pd.read_csv(neutron_file, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
	if (filt_caps):
		neutron_data, truth_data = filterCaptures(neutron_data, truth_data_set)
	else:
		truth_data = truth_data_set
		neutron_data = neutron_data[neutron_data["trackPDG"] == 2112]
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
# define fitting function
def angleFit(nbins, seg_size, num_events, it_num, save_output=True, save_plots=True, output_path="fits/", show_plots=False, debug=False):
	# read data files
	truth_angle = 0 # truth angle of "data"
	data_range = 180 # range of data
	Li6_filter = True
	th_range = [truth_angle - data_range, truth_angle + data_range]
	# path to import "data" file
	pre_str = "/home/jack/RATPAC2/ratpac-setup/ratpac/output/mtv_directionality/res"
	num_ev_str = {10: "_10", 100: "_100", 1000: "_1k", 10000: "_10k"}
	dop_str = "_001wt"
	run_type_str = {"f": "_fixed", "r": "_ref"}
	data_type_str = {"t": "/truth.txt", "n": "/neutrons.txt"}
	fixed_str = pre_str + num_ev_str[num_events] + dop_str + run_type_str["f"]
	# read truth.txt and neutrons.txt for fixed sets. Filter neutron events for captures on Li-6
	fixed_set_truth = readTruthData(fixed_str + data_type_str["t"])
	fixed_set_neutrons, fixed_set_truth = readNeutrons(fixed_str + data_type_str["n"], fixed_set_truth, Li6_filter)
	# get neutron capture events and IBD vertices for fixed data set
	locs = fixed_set_neutrons.groupby("Row").last()
	truth_locs = fixed_set_truth.groupby("Row").first()# keep only the first entry. MC values don't change
	# bin fixed location events and normalize binning
	caps_fixed = binEvents(locs, truth_locs, truth_angle, seg_size, nbins)
	caps_fixed = caps_fixed/np.sum(caps_fixed)	
	# read reference data set and filter neutron events for captures on Li-6
	ref_str = pre_str + num_ev_str[10000] + dop_str + run_type_str["r"]
	ref_set_truth = readTruthData(ref_str + data_type_str["t"])
	ref_set_neutrons, ref_set_truth = readNeutrons(ref_str + data_type_str["n"], ref_set_truth, Li6_filter)
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
	# set up the curve fit to norms or sum_sq
	fit_range = 179
	# use argmin of norms as initial guess for fit angle
	fnd_min = norms[np.argmin(norms)]
	angle_guess = angles[np.argmin(norms)]
	filename = f"segsz_{seg_size}mm_FND_fit{num_ev_str[num_events]}.txt"
	out_str1 = "Initial guess for fit #"
	out_str1 += f"{it_num} in {num_events} events dataset: {angle_guess}\nFND value at initial guess: {fnd_min}\n"
	if (save_output):
		fstring = output_path + filename
		with open(fstring, "a") as f:
			print(out_str1, file=f)
	elif (debug):
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
	# do plots and save if called for
	fig2, ax2 = plt.subplots(figsize=(6, 4))
	ax2.set_xlabel("$\\vartheta$ (${}^\\circ$)")
	# function, y axis labels, and plot
	out_str2 = "Fit parameters for |sin(x-x_0)| fit #"
	out_str2 += f"{it_num} in {num_events} events dataset:\n[y_offset, amplitude, angle (degrees:)] {sin_popt}\nerror: {sin_y_errors}\n"
	if (save_output):
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
		plt.savefig(output_path + f"plots/angle_{truth_angle}_deg_{seg_size}mm_segsz{num_ev_str[num_events]}_evts_fit_{it_num}.pdf", format="pdf")
	plt.close()
	return sin_popt[2]
# TODO: GUI and QOL improvements. Do money plot for 5, 50 and 150 mm segment sizes using 9, 5, and 3 matrix sizes, respectively
# TODO: For GUI, specify reference file (menu option) and data file (menu option), segment size (checkboxes, allowing all requiring at least one), pushbutton to calculate fits to data file, pushbutton to do resolution analysis
# TODO: QOL improvement -- first time data is loaded for reference set, save it to a JSON. GUI option/prompt to load from JSON on start, or specify new reference set
# segment size in mm
# TODO: GUI/QOL improvement -- text box to specify number of iterations to perform for resolution analysis using statistical approach, radio button options to use either statistical or covariance matrix to determine resolution from fits
# main function iterates fits through various segment sizes, number of events in each "data" file, and number of iterations for fits
seg_sz = [150, 50, 5]#[5]
# matrix size for n by n binning matrix -- events are binned by counting the number of segments between prompt and delayed vertices TODO: find proper binning size for 5 mm segments
grid_size = [3, 5, 17]#[17]
seg_style = {150: "b--", 50: "b-.", 5: "b:"}#{5: "b--"}
fit_res = {150: [], 50: [], 5: []}#{5: []}
write_outputs = True
keep_plots = True
debug_msg = False
out_file_path = "fits/"
iters = 5
n_evts = [10, 100, 1000, 10000]
dir_str = {10: "moneyplot_10_evt_fits/", 100: "moneyplot_100_evt_fits/", 1000: "moneyplot_1k_evt_fits/", 10000: "moneyplot_10k_evt_fits/"}
n_evts_str = {10: "_10", 100: "_100", 1000: "_1k", 10000: "_10k"}
plt.rc('font', family='serif', size = 14)
plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'cm' # computer modern
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.set_xlabel("Number of events")
ax1.set_xscale("log")
ax1.set_ylabel("Angular resolution (${}^\\circ$)")
ax1.set_yscale("log")
for k, sz in enumerate(seg_sz):
	fit_angle = {10: [], 100: [], 1000: [], 10000: []}
	fit_means = {10: 0, 100: 0, 1000: 0, 10000: 0}
	fit_stds = {10: 0, 100: 0, 1000: 0, 10000: 0}
	for j, num in enumerate(n_evts):
		if (debug_msg):
			print(f"Calculating fits and angular resolution for {num} events in {seg_sz[k]}mm segments with {iters} iterations...\n")
		out_file_str = f"seg_size_{sz}_resolution_{n_evts_str[num]}_evts.txt"
		for i in range(iters):
			angle = angleFit(grid_size[k], sz, num, i, write_outputs, keep_plots, out_file_path + dir_str[num], False, False)
			fit_angle[num].append(angle)
		fit_means[num] = np.mean(fit_angle[num])
		fit_stds[num] = np.std(fit_angle[num])
		fit_res[sz].append(fit_stds[num])
		out_str = f"Fit value and angular resolution for {sz}mm after {iters} iterations on {num} events (deg.)\nfit: {fit_means[num]} \tres: {fit_stds[num]}\n"
		if (write_outputs):
			with open(out_file_path + dir_str[num] + out_file_str, "w") as f:
				print(out_str, file=f)
		elif (debug_msg):
			print(out_str)
	if (write_outputs):
		summary_file_str = f"angular_res_summary_{sz}mm_segments.txt"
		with open(out_file_path + summary_file_str, "w") as f:
			print(f"Fit angles (deg:)\n{fit_angle}", file=f)
			print(f"Fit values (deg:)\n{fit_means}", file=f)
			print(f"Fit resolutions (deg:)\n{fit_stds}", file=f)
	print(f"Fit angles for {sz}mm segents (deg:)\n{fit_angle}")
	print(f"Fit values for {sz}mm segents (deg:)\n{fit_means}")
	print(f"Fit resolutions for {sz}mm segents (deg:)\n{fit_stds}")
	ax1.plot(n_evts, fit_res[sz], seg_style[sz], label = f"{sz}mm")
# show me the money!!!
ax1.legend(loc="upper right", framealpha=1.0)
fig.tight_layout()
plt.show()
if (keep_plots):
	plt.savefig(out_file_path + f"moneyplot_{iters}iters.pdf", format="pdf")
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



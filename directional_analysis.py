
import os
import sys
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

# Notes:
# TODO: GUI and QOL improvements. Do money plot for 5, 50 and 150 mm segment sizes using 9, 5, and 3 matrix sizes, respectively
# TODO: For GUI, specify reference file (menu option) and data file (menu option), segment size (checkboxes, allowing all requiring at least one), pushbutton to calculate fits to data file, pushbutton to do resolution analysis
# TODO: QOL improvement -- first time data is loaded for reference set, save it to a JSON. GUI option/prompt to load from JSON on start, or specify new reference set
# segment size in mm
# TODO: GUI/QOL improvement -- text box to specify number of iterations to perform for resolution analysis using statistical approach, radio button options to use either statistical or covariance matrix to determine resolution from fits


################################################################
################################################################
################################################################
class DirectionalAnalysis():
	"""
	DirectionalAnalysis class for use on processed RATPAC2 output files. This code reads the truth.txt and neutrons.txt files, bins the events in (normalized) directional matrices and performs Frobenius norm of the difference fits on these matrices to find the reference angle. The fits are performed multiple times for statistical uncertainty of the angle, which is then plotted as angular resolution as a function of number of events used in the fit.
	"""
	################################################################
	def __init__(self):
		# control Booleans
		self.debug = False
		self.latex = True
		self.Li6_filter = True
		
		if(self.debug):
			print("DirectionalAnalysis class method: __init__")
			
		if(self.latex):
			plt.rc('font', family='serif', size = 14)
			plt.rcParams['text.usetex'] = True
			plt.rcParams['mathtext.fontset'] = 'cm' # computer modern
			
		# data processing objects
		self.pdg_codes = {"alpha": 1.00002004e+09, "triton": 1.00001003e+09, "neutron": 2112, "positron": -11}
		self.num_ev_str = {10: "_10", 20: "_20", 50: "_50", 100: "_100", 200: "_200", 500: "_500", 1000: "_1k", 2000: "_2k", 5000: "_5k", 10000: "_10k"}
		self.truth_file_ref = "/home/jack/RATPAC2/ratpac-setup/ratpac/output/mtv_directionality/res_10k_001wt_ref/truth.txt"
		self.neutron_file_ref = "/home/jack/RATPAC2/ratpac-setup/ratpac/output/mtv_directionality/res_10k_001wt_ref/neutrons.txt"
		self.truth_data_ref = self.readTruthData(self.truth_file_ref)
		self.neutron_data_ref, self.truth_data_ref = self.readNeutrons(self.neutron_file_ref, self.truth_data_ref, self.Li6_filter)
		# list objects for segment size and binning matrix size
		self.seg_sz = [150, 50, 5]#[150]
		self.grid_size = [3, 5, 17]#[3]
		self.out_file_path = "fits/"
		self.iters = 5
		return
	################################################################
	# define functions for fits -- gaussian and sin_sq not used for now
	def gaussian(self, x, amplitude, mean, stddev):
		return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)
	def abs_sin(self, x, yoffset, amplitude, xoffset):
		return yoffset + amplitude * np.abs(np.sin( (np.pi/360.0) * (x - xoffset) ))
	def sin_sq(self, x, yoffset, amplitude, xoffset):
		return yoffset + amplitude * np.square(np.sin( (np.pi/360.0) * (x - xoffset) ))
	################################################################
	# define data filtering function for neutron captures on Li-6
	def filterCaptures(self, unfiltered_data, truth_data):
		if(self.debug):
			print("DirectionalAnalysis class method: filterCaptures")
		# identify tracks containing He-4 and H-3 nuclei (byproducts of Li-6 capture)
		valid_rows = unfiltered_data.groupby("Row")["trackPDG"].apply(
			lambda pdgs: (self.pdg_codes["alpha"] in pdgs.values) and (self.pdg_codes["triton"] in pdgs.values)
		)
		# select sets of neutron tracks containing Li-6 byproducts
		filtered_data = unfiltered_data[(unfiltered_data["Row"].isin(valid_rows[valid_rows].index))]
		truth_data = truth_data[(truth_data["Row"].isin(valid_rows[valid_rows].index))]
		# remove Li-6 byproducts, keeping only neutrons for neutron events
		filtered_data = filtered_data[filtered_data["trackPDG"] == self.pdg_codes["neutron"]]
		# return data sets of filtered (neutron and MC truth) events
		return filtered_data, truth_data
	################################################################
	# define functions to read truth.txt, neutrons.txt and positrons.txt
	def readTruthData(self, truth_file):
		truth_data = pd.read_csv(truth_file, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
		return truth_data
	def readNeutrons(self, neutron_file, truth_data_set, filt_caps=True):
		neutron_data = pd.read_csv(neutron_file, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
		if (filt_caps):
			neutron_data, truth_data = self.filterCaptures(neutron_data, truth_data_set)
		else:
			truth_data = truth_data_set
			neutron_data = neutron_data[neutron_data["trackPDG"] == self.pdg_codes["neutron"]]
		return neutron_data, truth_data
	def readPositrons(self, positron_file):
		positron_data = pd.read_csv(positron_file, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
		return positron_data
	################################################################
	# define binning function
	def binEvents(self, filtered_data, truth_data, theta, seg_size, matrix_size, plot=False):
		if(self.debug):
			print("DirectionalAnalysis class method: binEvents")
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
	def angleFit(self, nbins, seg_size, num_events, it_num, save_output=True, save_plots=True, output_path="fits/", show_plots=False, debug=False):
		if(debug):
			print("DirectionalAnalysis class method: angleFit")
		# read data files
		truth_angle = 0 # truth angle of "data"
		data_range = 180 # range of data
		#Li6_filter = True
		th_range = [truth_angle - data_range, truth_angle + data_range]
		# strings to build path to import "data" file
		pre_str = "/home/jack/RATPAC2/ratpac-setup/ratpac/output/mtv_directionality/res"
		dop_str = "_001wt"
		run_type_str = {"f": "_fixed", "r": "_ref"}
		# file name strings
		data_type_str = {"t": "/truth.txt", "n": "/neutrons.txt"}
		fixed_str = pre_str + self.num_ev_str[num_events] + dop_str + run_type_str["f"]
		ref_str = pre_str + self.num_ev_str[num_events] + dop_str + run_type_str["r"]
		# read truth.txt and neutrons.txt for fixed sets. Filter neutron events for captures on Li-6
		fixed_set_truth = self.readTruthData(fixed_str + data_type_str["t"])
		fixed_set_neutrons, fixed_set_truth = self.readNeutrons(fixed_str + data_type_str["n"], fixed_set_truth)
		# get neutron capture events and IBD vertices for fixed data set
		locs = fixed_set_neutrons.groupby("Row").last()
		truth_locs = fixed_set_truth.groupby("Row").first()# keep only the first entry. MC values don't change
		# bin fixed location events and normalize binning
		caps_fixed = self.binEvents(locs, truth_locs, truth_angle, seg_size, nbins)
		num_caps = np.sum(caps_fixed)
		if(debug):
			print(f"Number of binned captures: {num_caps}")
		caps_fixed = caps_fixed/num_caps
		ref_locs = self.neutron_data_ref.groupby("Row").last()# keep only captures
		ref_truth_locs = self.truth_data_ref.groupby("Row").first()# keep only first entry of MC events
		fnd_fit_str = f"Calculating FND fit #{it_num} for {fixed_str} and {ref_str} ..."
		if (debug):
			print(fnd_fit_str)
		# Iterate over reference angles to minimize FND for |sin(th-th0)| fit
		angles = []
		norms = []
		for th in range(-data_range,data_range):
			angles.append(th)
			caps_ref = self.binEvents(ref_locs, ref_truth_locs, th, seg_size, nbins)
			caps_ref = caps_ref/np.sum(caps_ref)
			fnd_sq = np.sum(np.square(caps_ref - caps_fixed))
			fnd = np.sqrt(fnd_sq)
			norms.append(fnd)
		# set up the curve fit to norms or sum_sq
		fit_range = 179
		# use argmin of norms as initial guess for fit angle
		fnd_min = norms[np.argmin(norms)]
		angle_guess = angles[np.argmin(norms)]
		filename = f"segsz_{seg_size}mm_FND_fit{self.num_ev_str[num_events]}.txt"
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
		sin_popt, sin_pcov = curve_fit(self.abs_sin, sin_fit_angles, sin_fit_norms, p0=[sin_y_offset, np.max(sin_fit_norms), angle_guess])
		# output fit params and covariance matrix to a file
		sin_y_fit = self.abs_sin(sin_fit_angles, sin_popt[0], sin_popt[1], sin_popt[2])
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
			plt.savefig(output_path + f"plots/angle_{truth_angle}_deg_{seg_size}mm_segsz{self.num_ev_str[num_events]}_evts_fit_{it_num}.pdf", format="pdf")
		plt.close()
		return sin_popt[2]
	################################################################
	# moneyplot function iterates fits through various segment sizes, number of events in each "data" file, and number of iterations for fits
	def resolutionPlot(self, n_iters, write_outputs=True, keep_plots=True, debug_msg=False):
		# matrix size for n by n binning matrix -- events are binned by counting the number of segments between prompt and delayed vertices
		# TODO: find proper binning size for 5 mm segments
		seg_style = {150: "b--", 50: "b-.", 5: "b:"}#{150: "b--"}
		fit_res = {150: [], 50: [], 5: []}#{150: []}
		n_evts = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
		dir_str = {
			10: "moneyplot_10_evt_fits/", 20: "moneyplot_20_evt_fits/", 50: "moneyplot_50_evt_fits/", 
			100: "moneyplot_100_evt_fits/", 200: "moneyplot_200_evt_fits/", 500: "moneyplot_500_evt_fits/", 
			1000: "moneyplot_1k_evt_fits/", 2000: "moneyplot_2k_evt_fits/", 5000: "moneyplot_5k_evt_fits/", 
			10000: "moneyplot_10k_evt_fits/"
		}
		fig, ax1 = plt.subplots(figsize=(6, 4))
		ax1.set_xlabel("Number of events")
		ax1.set_xscale("log")
		ax1.set_ylabel("Angular resolution (${}^\\circ$)")
		ax1.set_yscale("log")
		for k, sz in enumerate(self.seg_sz):
			fit_angle = {10: [], 20: [], 50: [], 100: [], 200: [], 500: [], 1000: [], 2000: [], 5000: [], 10000: []}
			#num_fits = {10: [], 20: [], 50: [], 100: [], 200: [], 500: [], 1000: [], 2000: [], 5000: [], 10000: []}
			fit_means = {10: 0, 20: 0, 50: 0, 100: 0, 200: 0, 500: 0, 1000: 0, 2000: 0, 5000: 0, 10000: 0}
			fit_stds = {10: 0, 20: 0, 50: 0, 100: 0, 200: 0, 500: 0, 1000: 0, 2000: 0, 5000: 0, 10000: 0}
			for j, num in enumerate(n_evts):
				if (debug_msg):
					print(f"Calculating fits and angular resolution for {num} events in {sz}mm segments with {iters} iterations...\n")
				out_file_str = f"seg_size_{sz}_resolution{self.num_ev_str[num]}_evts.txt"
				for i in range(n_iters):
					angle = self.angleFit(self.grid_size[k], sz, num, i, write_outputs, keep_plots, self.out_file_path + dir_str[num], False, False)
					fit_angle[num].append(angle)
				fit_means[num] = np.mean(fit_angle[num])
				fit_stds[num] = np.std(fit_angle[num])
				fit_res[sz].append(fit_stds[num])
				out_str = "Fit value and angular resolution for "
				out_str += f"{sz}mm after {self.iters} iterations on {num} events (deg.)\nfit: {fit_means[num]} \tres: {fit_stds[num]}\n"
				if (write_outputs):
					with open(self.out_file_path + dir_str[num] + out_file_str, "w") as f:
						print(out_str, file=f)
				elif (debug_msg):
					print(out_str)
			if (write_outputs):
				summary_file_str = f"angular_res_summary_{sz}mm_segments.txt"
				with open(self.out_file_path + summary_file_str, "w") as f:
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
			plt.savefig(self.out_file_path + f"moneyplot_{self.iters}iters.pdf", format="pdf")
		plt.close()
		return
################################################################
if __name__ == "__main__":
    """
    This is the main statement where the user can execute subroutines to produce figures.
    """
da = DirectionalAnalysis()
itnums = 5
da.resolutionPlot(itnums)
sys.exit()
# dumping ground for unused code



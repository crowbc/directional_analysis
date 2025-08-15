
import os
import sys
import time as timer
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
# TODO: GUI and QOL improvements. Select statistical resolution plot (iterate fits, selectable, but default to 30) or covariance derived angular resolution plot.
# TODO: For GUI, specify reference file (menu option) and data file (menu option), segment size (checkboxes, allowing all requiring at least one), pushbutton to calculate fits to data file, pushbutton to do resolution analysis
# TODO: QOL improvement -- first time data is loaded for reference set, save it to a JSON. GUI option/prompt to load from JSON on start, or specify new reference set
# segment size in mm (especially useful for shell scripting)
# TODO: GUI/QOL improvement -- text box to specify number of iterations to perform for resolution analysis using statistical approach, radio button options to use either statistical or covariance matrix to determine resolution from fits
# TODO: 100k event reference run? Discuss merits before committing, as it will most likely increase the time needed to do the fits.
# TODO: Create functions to initialize reference data and perform the binning and rotating of the reference set. Call these from __init__ and make the reference matrices callable objects for use in the angleFit() method.
# TODO: Create command line control of output, saving fitting outputs to default paths. Warn user that output files will be appended and plots will be overwritten
# TODO: Get environment variables for RATPAC path -- add these to .bashrc if needed.
# TODO: Find correct binning matrix size for 5 mm segments. 17x17 gets about 40% of captures. Trying 25x25
# TODO: Fitter function needs status bar for angles being fit

################################################################
################################################################
################################################################
class DirectionalAnalysis():
	"""
	DirectionalAnalysis class for use on processed RATPAC2 output files. This code reads the truth.txt and neutrons.txt files, bins the events in (normalized) directional matrices 
	and performs Frobenius norm of the difference fits on these matrices to find the reference angle. The fits are performed multiple times for statistical uncertainty of the angle,
	which is then plotted as angular resolution as a function of number of events used in the fit.
	"""
	################################################################
	# __init__ method to initialize class and create instance with
	# defined parameters
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
		
		self.time_start = timer.time()	
		# data processing objects
		self.pdg_codes = {
					"alpha": 1.00002004e+09, "triton": 1.00001003e+09, 
					"proton": 2212, "neutron": 2112, 
					"electron": 11, "positron": -11, 
					"nu_e": 12, "nu_ebar": -12
		}
		self.num_ev_str = {
					10: "_10", 20: "_20", 50: "_50", 100: "_100", 200: "_200", 500: "_500", 
					1000: "_1k", 2000: "_2k", 5000: "_5k", 10000: "_10k"
		}
		# initialize reference data to get vertices for binning and fits
		self.initRefData()
		
		if(self.debug):
			print("Reference data for MC truth vertices and neutron captures initialized.")
			print(f"Time to create data objects:\t{timer.time() - self.time_start}s")
			#print(f"Neutron capture vertices:\n{self.ref_locs}")
		# list objects for segment size and binning matrix size --- note: use only 150 mm 3x3 matrix for testing and debugging
		self.seg_sz = [150]#[150, 50, 5]
		self.grid_size = {150: 3, 50: 5, 5: 25}
		# parameters for creating binning matrices of reference data and doing fits
		self.data_range = 180
		self.truth_angle = 0
		self.th_range = [self.truth_angle - self.data_range, self.truth_angle + self.data_range]
		# reference binning matrices -- do for each reference angle from -180 to 180, but number from 0 to 359 -- need to do for all segment sizes and grid sizes
		#if(self.debug):
			#print("Initializing reference matrices...")
		# initialze reference matrices with 0 in all elements
		#self.initRefMatrix()
		#if(self.debug):
			#print(f"Creating binning matrices for reference data set in angle range {self.th_range}...")
		#for i, sz in enumerate(self.seg_sz):
			#if(self.debug):
				#print(f"Creating binning matrices for {sz}mm segments.")
			#for angle in range(self.th_range[0], self.th_range[1]):
				#self.binned_ref[sz][angle] = self.binEvents(self.ref_locs, self.ref_truth_locs, angle, sz, self.grid_size[sz])
				#if(self.debug):
					#print(f"{self.binned_ref[sz][angle]}")
			#if(self.debug):
				#print(f"binned events for angles {self.th_range[0]} through {self.th_range[1]} on {sz}mm segments.")
		self.out_file_path = "fits/"
		return
	################################################################
	# define functions for fits (only abs_sin is currently used)
	################################################################
	def gaussian(self, x, amplitude, mean, stddev):
		if(self.debug):
			print("DirectionalAnalysis class method: gaussian")
		return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)
	def abs_sin(self, x, yoffset, amplitude, xoffset):
		if(self.debug):
			print("DirectionalAnalysis class method: abs_sin")
		return yoffset + amplitude * np.abs(np.sin( (np.pi/360.0) * (x - xoffset) ))
	def sin_sq(self, x, yoffset, amplitude, xoffset):
		if(self.debug):
			print("DirectionalAnalysis class method: sin_sq")
		return yoffset + amplitude * np.square(np.sin( (np.pi/360.0) * (x - xoffset) ))
	################################################################
	# define data filtering function for neutron captures on Li-6
	################################################################
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
	# define functions to read truth.txt, neutrons.txt and
	# positrons.txt (not currently used)
	################################################################
	def readTruthData(self, truth_file):
		if(self.debug):
			print("DirectionalAnalysis class method: readTruthData")
		truth_data = pd.read_csv(truth_file, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
		return truth_data
	def readNeutrons(self, neutron_file, truth_data_set, filt_caps=True):
		if(self.debug):
			print("DirectionalAnalysis class method: readNeutrons")
		neutron_data = pd.read_csv(neutron_file, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
		if (filt_caps):
			neutron_data, truth_data = self.filterCaptures(neutron_data, truth_data_set)
		else:
			truth_data = truth_data_set
			neutron_data = neutron_data[neutron_data["trackPDG"] == self.pdg_codes["neutron"]]
		return neutron_data, truth_data
	def readPositrons(self, positron_file):
		if(self.debug):
			print("DirectionalAnalysis class method: readPositrons")
		positron_data = pd.read_csv(positron_file, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
		return positron_data
	################################################################
	# define function to read truth.txt & neutrons.txt for reference
	# dataset. positrons.txt is not currently used.
	################################################################
	def initRefData(self):
		if(self.debug):
			print("DirectionalAnalysis class method: initRefData")
		ref_truth_file = "/home/jack/RATPAC2/ratpac-setup/ratpac/output/mtv_directionality/res_10k_001wt_ref/truth.txt"
		ref_neutron_file = "/home/jack/RATPAC2/ratpac-setup/ratpac/output/mtv_directionality/res_10k_001wt_ref/neutrons.txt"
		try:
			self.ref_truth_data = self.readTruthData(ref_truth_file)
		except Exception as err:
			print("Error in truth data or in pandas dataframe")
			print(err)
			print("Quitting directional analysis...")
			sys.exit(1)
		try:
			self.ref_neutron_data, self.ref_truth_data = self.readNeutrons(ref_neutron_file, self.ref_truth_data, self.Li6_filter)
		except Exception as err:
			print("Error in neutron data or in pandas dataframe")
			print(err)
			print("Quitting directional analysis...")
			sys.exit(1)
		try:
			self.ref_locs = self.ref_neutron_data.groupby("Row").last()# keep only captures
			self.ref_truth_locs = self.ref_truth_data.groupby("Row").first()# keep only first entry of MC events
		except Exception as err:
			print("Error in picking reference locations from pandas dataframe.")
			print(err)
			print("Quitting directional analysis...")
			sys.exit(1)
		return
	################################################################
	# define reference matrix initializer
	################################################################
	def initRefMatrix(self):
		if(self.debug):
			print("DirectionalAnalysis class method: initRefMatrix")
		try:
			for i, sz in enumerate(self.seg_sz):
				if(self.debug):
					print(f"Creating binning matrices for {sz}mm segments.")
				for angle in range(self.th_range[0], self.th_range[1]):
					self.binned_ref[sz][angle] = self.initMatrix(sz, self.grid_size[sz])
			if(self.debug):
				print(f"Created binning matrices for {self.seg_sz} segment sizes, and {self.th_range} deg angle range.")
				print(f"Example matrix:\n{self.binned_ref[150][0]}")
		except Exception as err:
			print("Error in creating binning matrices for reference data set.")
			print(err)
			print("Quitting directional analysis...")
			sys.exit(2)
		return
	################################################################
	# define binning matrix initializer
	################################################################
	def initMatrix(self, seg_sz, mat_sz):
		#if(self.debug):# suppressed due to excessive output in log files
			#print("DirectionalAnalysis class method: initMatrix")
		# Initializes matrix for segmentation.
		bin_mat = np.zeros((mat_sz, mat_sz), dtype=dict)
		# Size of half of the total detector grid in mm.
		half_size = seg_sz * mat_sz / 2.0
		# Initialize binning matrix for specified segment size and matrix size -- might help to make functions to initialize and re-zero matrices
		for i in range(mat_sz):
			for j in range(mat_sz):
				xlow  = seg_sz * j - half_size
				xhigh = seg_sz * j + seg_sz - half_size
				yhigh = -(seg_sz * i - half_size)
				ylow  = -(seg_sz * i + seg_sz - half_size)
				bin_mat[i][j] = {
						"xbounds": [xlow, xhigh],
						"ybounds": [ylow, yhigh],
						"counts":  0
				}
		return bin_mat
	################################################################
	# define binning function
	################################################################
	def binEvents(self, filtered_data, truth_data, theta, seg_size, matrix_size):#, plot=False):
		if(self.debug):# suppressed due to excessive output in log files
			print("DirectionalAnalysis class method: binEvents")
		time_bin_start = timer.time()
		# Initializes matrix for segmentation.
		seg = self.initMatrix(seg_size, matrix_size)
		# get prompt-delayed displacement coordinates for binning -- use pandas for this
		x_coords = filtered_data["trackPosX"] - truth_data["mcx"]
		y_coords = filtered_data["trackPosY"] - truth_data["mcy"]
		z_coords = filtered_data["trackPosZ"] - truth_data["mcz"]
		# bin neutron captures
		for x, y, z in zip(x_coords, y_coords, z_coords):
			# Generate a random point in the center square segment. -- use pandas for this
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
		time_ck = timer.time()
		if (self.debug):
			debug_str = f"Time for binning of {theta}deg angle: {time_ck - time_bin_start}s"
			print(debug_str)
		return caps
	################################################################
	# define fitting function
	################################################################
	def angleFit(
			self, nbins, seg_size, num_events, it_num, 
			save_output=True, save_plots=True, output_path="fits/", 
			show_plots=False, debug=False
	):
		if (self.debug):
			print("DirectionalAnalysis class method: angleFit")
		# strings to build path to import "data" file
		pre_str = "/home/jack/RATPAC2/ratpac-setup/ratpac/output/mtv_directionality/res"
		dop_str = "_001wt"
		run_type_str = {"f": "_fixed", "r": "_ref"}
		data_type_str = {"t": "/truth.txt", "n": "/neutrons.txt"}
		# file name strings
		fixed_str = pre_str + self.num_ev_str[num_events] + dop_str + run_type_str["f"]
		ref_str = pre_str + self.num_ev_str[10000] + dop_str + run_type_str["r"]
		filename = f"FND_fit_{seg_size}mm_segsz{self.num_ev_str[num_events]}_evts.txt"
		out_str = ""
		if (debug):
			print(f"Reading data for {num_events} IBD events.")
		# debug timer
		time_fit_start = timer.time()
		# read truth.txt and neutrons.txt for fixed sets. Filter neutron events for captures on Li-6
		try:
			fixed_set_truth = self.readTruthData(fixed_str + data_type_str["t"])
			truth_set_read_time = timer.time() - time_fit_start
			out_str = f"Time to read MC truth data: {truth_set_read_time}s"
			if (debug):
				print(out_str)
		except Exception as err:
			print(f"Error reading MC truth IBD vertices for {num_events} or pandas dataframe.")
			print(err)
			print("Quitting directional analysis...")
			sys.exit(3)
		try:
			fixed_set_neutrons, fixed_set_truth = self.readNeutrons(
										fixed_str + data_type_str["n"], 
										fixed_set_truth, self.Li6_filter
			)
			neutron_set_read_time = timer.time() - truth_set_read_time
			out_str = f"Time to read neutron capture data: {neutron_set_read_time}s"
			if (debug):
				print(out_str)
		except Exception as err:
			print(f"Error reading neutron capture data for {num_events} IBD events or pandas dataframe.")
			print(err)
			print("Quitting directional analysis...")
			sys.exit(3)
		# get neutron capture events and IBD vertices for fixed data set
		locs = fixed_set_neutrons.groupby("Row").last()
		truth_locs = fixed_set_truth.groupby("Row").first()
		if (self.Li6_filter):
			num_Li6_caps = len(truth_locs.groupby("Row"))
			out_str = f"Number of captures on Li-6: {num_Li6_caps}"
			if(debug):
				print(out_str)
		if (debug):
			out_str = f"Begin binning of {num_events} data set."
			print(out_str)
		# bin fixed location events and normalize binning
		caps_fixed = self.binEvents(
						locs, truth_locs, 
						self.truth_angle, seg_size, 
						nbins
		)
		if (debug):
			out_str = f"Binning of {num_events} data set is complete."
			print(out_str)
		num_caps = np.sum(caps_fixed)
		out_str = f"Number of binned captures: {num_caps}"
		if (debug):
			print(out_str)
		caps_fixed = caps_fixed/num_caps
		# Iterate over reference angles to minimize FND for |sin(th-th0)| fit
		angles = []
		norms = []
		time_FND_start = timer.time()
		out_str = f"Calculating FND fit #{it_num} for {fixed_str} and {ref_str} ..."
		if (debug):
			print(out_str)
		# do binning of reference matrices and calculate FND values
		for th in range(-self.data_range,self.data_range):
			angles.append(th)
			#caps_ref = self.binned_ref[seg_size][th]
			caps_ref = self.binEvents(
							self.ref_locs, 
							self.ref_truth_locs, 
							th, seg_size, nbins
			)
			num_ref = np.sum(caps_ref)
			caps_ref = caps_ref/num_ref
			fnd_sq = np.sum(np.square(caps_ref - caps_fixed))
			fnd = np.sqrt(fnd_sq)
			norms.append(fnd)
		time_FND_calc = timer.time() - time_FND_start
		if (debug):
			out_str = f"Time to bin neutron captures of reference set and compute FND with fixed set: {time_FND_calc}"
			print(out_str)
		# set up the curve fit to norms or sum_sq
		fit_range = 179
		# use argmin of norms as index for initial guess for fit angle and minimum
		idx_min = np.argmin(norms)
		fnd_min = norms[idx_min]
		angle_guess = angles[idx_min]
		out_str = f"Initial guess for fit #{it_num}: {angle_guess}\nFND value: {fnd_min}\n"
		if (save_output):
			fstring = output_path + filename
			with open(fstring, "a") as f:
				print(out_str, file=f)
		elif (debug):
			print(out_str)
		# do |sin(th-th0)| fit
		sin_y_offset = np.min(norms)
		sin_fit_norms = np.array([(fn - sin_y_offset) for fn in norms])
		sin_fit_angles = angles[(self.truth_angle - self.th_range[0] - fit_range):-(self.th_range[1] - self.truth_angle - fit_range)]
		sin_fit_norms = sin_fit_norms[(self.truth_angle - self.th_range[0] - fit_range):-(self.th_range[1] - self.truth_angle - fit_range)]
		time_fit_start = timer.time()
		# fit params and covariance
		sin_popt, sin_pcov = curve_fit(
						self.abs_sin, sin_fit_angles, sin_fit_norms, 
						p0=[sin_y_offset, np.max(sin_fit_norms), angle_guess]
		)
		time_to_fit = timer.time() - time_fit_start
		if (debug):
			out_str = f"Time to fit |sin(th-th0)| to {num_caps} events FNDs: {time_to_fit}s"
			print(out_str)
		# output fit params and covariance matrix to a file
		sin_y_fit = self.abs_sin(sin_fit_angles, sin_popt[0], sin_popt[1], sin_popt[2])
		sin_y_errors = np.sqrt(np.diag(sin_pcov))
		# do plots and save if called for
		fig2, ax2 = plt.subplots(figsize=(6, 4))
		ax2.set_xlabel("$\\vartheta$ (${}^\\circ$)")
		# function, y axis labels, and plot
		out_str = f"Fit parameters: \n [y_offset, amplitude, angle (degrees:)] {sin_popt}\nerrors: {sin_y_errors}\n"
		if (save_output):
			fstring = output_path + filename
			with open(fstring, "a") as f:
				print(out_str, file = f)
		elif (debug):
			print(out_str)
		fnd_label = "Frobenius norm"
		fit_label = f"$\\alpha|\\sin(\\vartheta-\\theta_0)|$\n(fit: $\\theta_0 = {sin_popt[2]:.1f}\\pm{sin_y_errors[2]:.1f}" + "{}^\\circ$\n"
		if (num_events<100):
			fit_label += f"($\\alpha = {sin_popt[1]:.3f}\\pm{sin_y_errors[1]:.3f}$))"
		else:
			fit_label += f"($\\alpha = {sin_popt[1]:.4f}\\pm{sin_y_errors[1]:.4f}$))"
		if(debug):
			print(fit_label)
		ax2.set_ylabel("FND")
		ax2.plot(sin_fit_angles, [y + sin_y_offset for y in sin_y_fit], color="red", label=fit_label)
		ax2.plot(angles, norms, "b.", label=fnd_label, alpha=0.5)
		# finish the plot
		ax2.axvline(self.truth_angle, linestyle="--", color="gray")
		ax2.legend(loc="upper left", framealpha=1.0)
		fig2.tight_layout()
		if (show_plots):
			plt.show()
		if (save_plots):
			save_str = output_path
			save_str += f"plots/angle_{self.truth_angle}_deg_{seg_size}"
			save_str += f"mm_segsz{self.num_ev_str[num_events]}_evts_fit_{it_num}.pdf"
			plt.savefig(save_str, format="pdf")
		plt.close()
		return sin_y_errors[2], sin_popt[2], num_caps
	"""
	################################################################
	# this function creates the plot for the FND fit if called for
	# and shows and saves the plots depending on keep_plots and
	# show_plots being turned on or off
	################################################################
	def plotFNDFit(self, p_name, fit_vals, fit_errs, keep_plots=True, show_plots=False, debug=False):
		fig2, ax2 = plt.subplots(figsize=(6, 4))
		ax2.set_xlabel("$\\vartheta$ (${}^\\circ$)")
		fnd_label = "Frobenius norm"
		fit_label = f"$\\alpha|\\sin(\\vartheta-\\theta_0)|$\n(fit: $\\theta_0 = {fit_vals[2]:.1f}\\pm{fit_errors[2]:.1f}" + "{}^\\circ$\n"
		if (debug):
			print(fit_label)
		ax2.set_ylabel("FND")
		ax2.plot(sin_fit_angles, [y + sin_y_offset for y in sin_y_fit], color="red", label=fit_label)
		# finish the plot
		ax2.axvline(self.truth_angle, linestyle="--", color="gray")
		ax2.legend(loc="upper left", framealpha=1.0)
		fig2.tight_layout()
		if (show_plots):
			plt.show()
		if (save_plots):
			plt.savefig(p_name, format="pdf")
		plt.close()
		return
	"""
	################################################################
	# this function iterates fits through various segment sizes, 
	# number of events in each "data" file, and number of iterations
	# for fits and generates the resolution plot
	################################################################
	def doIterativeFits(self, n_iters, p_name, f_name, write_outputs=True, keep_plots=True, show_plots=False, debug_msg=False):
		# matrix size for n by n binning matrix -- events are binned by counting the number of segments between prompt and delayed vertices
		if(self.debug):
			print("DirectionalAnalysis class method: doIterativeFits")
		show_fit_plots = False
		seg_style = {150: "b--", 50: "b-.", 5: "b:"}
		fit_res = {150: [], 50: [], 5: []}
		n_evts = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
		dir_str = {
				10: "res_plot_10_evt_fits/", 20: "res_plot_20_evt_fits/", 
				50: "res_plot_50_evt_fits/", 100: "res_plot_100_evt_fits/", 
				200: "res_plot_200_evt_fits/", 500: "res_plot_500_evt_fits/", 
				1000: "res_plot_1k_evt_fits/", 2000: "res_plot_2k_evt_fits/", 
				5000: "res_plot_5k_evt_fits/", 10000: "res_plot_10k_evt_fits/"
		}
		write_it = self.out_file_path + f_name
		plot_it = self.out_file_path + p_name
		summarize_it = self.out_file_path + "ang_res_summary_"
		fig, ax1 = plt.subplots(figsize=(6, 4))
		ax1.set_xlabel("Number of events")
		ax1.set_xscale("log")
		ax1.set_ylabel("Angular resolution (${}^\\circ$)")
		ax1.set_yscale("log")
		out_str = f"Iterative fitting for {self.seg_sz}mm segments and {n_evts} events using {n_iters} iterations:\n"
		if (write_outputs):
			with open(write_it, "w") as f:
				print(out_str, file=f)
		elif (debug_msg):
			print(out_str)
		for k, sz in enumerate(self.seg_sz):
			fit_angle = {
					10: [], 20: [], 50: [], 
					100: [], 200: [], 500: [], 
					1000: [], 2000: [], 5000: [], 
					10000: []
			}
			num_fits = {
					10: [], 20: [], 50: [], 
					100: [], 200: [], 500: [], 
					1000: [], 2000: [], 5000: [], 
					10000: []
			}
			fit_means = {
					10: 0, 20: 0, 50: 0, 
					100: 0, 200: 0, 500: 0, 
					1000: 0, 2000: 0, 5000: 0, 
					10000: 0
			}
			fit_stds = {
					10: 0, 20: 0, 50: 0, 
					100: 0, 200: 0, 500: 0, 
					1000: 0, 2000: 0, 5000: 0, 
					10000: 0
			}
			out_str = f"____________________\nSegment size: {sz}mm\n____________________\n"
			if (write_outputs):
				with open(write_it, "a") as f:
					print(out_str, file=f)
			elif (debug_msg):
				print(out_str)
			time_seg_start = timer.time()
			for j, num in enumerate(n_evts):
				out_str = f"Calculating fits and angular resolution for {num} events "
				out_str += f"in {sz}mm segments with {n_iters} iterations...\n"
				if (debug_msg):
					print(out_str)
				out_path_param = self.out_file_path + dir_str[num]
				time_num_start = timer.time()
				for i in range(n_iters):
					_, angle, n_caps = self.angleFit(
										self.grid_size[sz], sz, 
										num, i+1, write_outputs, 
										keep_plots, out_path_param, 
										show_fit_plots, debug_msg
					)
					fit_angle[num].append(angle)
					num_fits[num].append(n_caps)
				n_recons = np.mean(num_fits[num])
				fit_means[num] = np.mean(fit_angle[num])
				fit_stds[num] = np.std(fit_angle[num])
				fit_res[sz].append(fit_stds[num])
				time_ck_num = timer.time()
				if (debug_msg):
					time_running_num = time_ck_num - self.time_start
					time_num = time_ck_num - time_num_start
					out_str = f"Time since analysis started: {time_running_num}s"
					print(out_str)
					out_str = f"Time to fit {num} events in {sz}mm segments: {time_num}s"
					print(out_str)
				out_str = f"{num} events:\nfit: {fit_means[num]} deg\t"
				out_str += f"res (from statistics:) {fit_stds[num]} deg\n"
				out_str += f"<# of reconstructions> on {num} events: {n_recons} \n"
				if (write_outputs):
					with open(write_it, "a") as f:
						print(out_str, file=f)
				elif (debug_msg):
					print(out_str)
			time_ck_seg = timer.time()
			if (debug_msg):
				time_running = time_ck_seg - self.time_start
				out_str = f"Time since analysis started: {time_running}s"
				print(out_str)
				time_seg = time_ck_seg - time_seg_start
				out_str = f"Time to bin all event count files for {sz}mm segments: {time_seg}s"
				print(out_str)
			fit_complete_str = f"Summary of fit for {sz}mm segments:\n"
			fit_complete_str += f"Fit angles (deg:)\n{fit_angle}"
			fit_complete_str += f"\n<Fit angles> (deg:)\n{fit_means}"
			fit_complete_str += f"\nResolutions from statistics (deg:)\n{fit_res[sz]}\n"
			if (write_outputs):
				summary_file_str = summarize_it + f"{sz}mm_seg.txt"
				with open(summary_file_str, "w") as f:
					print(fit_complete_str, file=f)
			elif (debug_msg):
				print(fit_complete_str)
			ax1.plot(n_evts, fit_res[sz], seg_style[sz], label = f"{sz}mm")
		fit_summary_str = f"Fit resolutions for {self.seg_sz} segments:\n{fit_res}"
		summarize_it += "all.txt"
		if (write_outputs):
			with open(summarize_it, "w") as f:
				print(fit_summary_str, file=f)
		print(fit_summary_str)
		if(debug_msg):
			time_running = timer.time() - self.time_start
			out_str = f"Total time to run all fits: {time_running}s"
			print(out_str)
		# show me the money!!!
		ax1.legend(loc="upper right", framealpha=1.0)
		fig.tight_layout()
		if (keep_plots):
			plt.savefig(plot_it, format="pdf")
		if (show_plots):
			plt.show()
		plt.close()
		return
	################################################################
	# function makes single fit for various segment sizes, and
	# number of events in each "data" file, and saves output for use
	# in parallel shell script.
	# Alternatively, a single fit resolution plot can be made using
	# the covariance matrix values from the fits as the angular
	# resolutions.
	################################################################
	def doSingleFit(self, it_num, f_name, write_outputs=True, debug_msg=False):# unused parameters: keep_plots=True (after write_outputs)
		# matrix size for n by n binning matrix -- events are binned by counting the number of segments between prompt and delayed vertices
		# TODO: find proper binning size for 5 mm segments 17x17 bins approximately 40% of events, or half of captures
		if(self.debug):
			print("DirectionalAnalysis class method: doSingleFit")
		if (f_name == ""):
			f_name = f"fit_params_itnum{it_num}.txt"
		write_it = self.out_file_path + f_name
		summarize_it = self.out_file_path + "ang_res_summary_"
		out_str = f"Fits for iteration #{it_num}:\n"
		save_fit_plots = True
		if (write_outputs):
			with open(write_it, "w") as f:
				print(out_str, file=f)
		elif (debug_msg):
			print(out_str)
		fit_res = {150: [], 50: [], 5: []}
		n_evts = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
		dir_str = {
				10: "res_plot_10_evt_fits/", 20: "res_plot_20_evt_fits/", 
				50: "res_plot_50_evt_fits/", 100: "res_plot_100_evt_fits/", 
				200: "res_plot_200_evt_fits/", 500: "res_plot_500_evt_fits/", 
				1000: "res_plot_1k_evt_fits/", 2000: "res_plot_2k_evt_fits/", 
				5000: "res_plot_5k_evt_fits/", 10000: "res_plot_10k_evt_fits/"
		}
		for k, sz in enumerate(self.seg_sz):
			fit_angle = {
					10: [], 20: [], 50: [], 
					100: [], 200: [], 500: [], 
					1000: [], 2000: [], 5000: [], 
					10000: []
			}
			num_fits = {
					10: [], 20: [], 50: [], 
					100: [], 200: [], 500: [], 
					1000: [], 2000: [], 5000: [], 
					10000: []
			}
			out_str = f"____________________\nSegment size: {sz}mm\n____________________\n"
			time_seg_start = timer.time()
			if (write_outputs):
				with open(write_it, "a") as f:
					print(out_str, file=f)
			elif (debug_msg):
				print(out_str)
			for j, num in enumerate(n_evts):
				if (debug_msg):
					print(f"Calculating fits and angular resolution for {num} events in {sz}mm segments ...\n")
				out_path_param = self.out_file_path + dir_str[num]
				time_num_start = timer.time()
				# Get fit values
				res, angle, n_caps = self.angleFit(
									self.grid_size[sz], sz, num, 
									it_num, write_outputs, save_fit_plots, 
									out_path_param, False, debug_msg
				)
				fit_angle[num].append(angle)
				num_fits[num].append(n_caps)
				n_recons = np.mean(num_fits[num])
				fit_res[sz].append(res)
				time_ck_num = timer.time()
				if (debug_msg):
					time_num = time_ck_num - time_num_start
					time_running_num = time_ck_num - self.time_start
					out_str = f"Time since analysis started: {time_running_num}"
					print(out_str)
					out_str = f"Time to bin and fit {num} events in {sz}mm segments: {time_num}s"
					print(out_str)
				out_str = f"{num} events:\nfit: {angle} deg\t"
				out_str += f"res (from covariance:) {res} deg\n"
				out_str += f"# of reconstructions on {num} events: {n_caps}\n"
				if (write_outputs):
					with open(write_it, "a") as f:
						print(out_str, file=f)
				elif (debug_msg):
					print(out_str)
			time_ck_seg = timer.time()
			fit_complete_str = f"Summary of fit for {sz}mm segments:\n"
			print(fit_complete_str)
			fit_complete_str = f"Fit angles (deg:)\n{fit_angle}\n"
			print(fit_complete_str)
			fit_complete_str = f"Resolutions from covariance (deg:)\n{fit_res[sz]}\n"
			print(fit_complete_str)
		fit_summary_str = f"Fit resolutions for {self.seg_sz} segments:\n{fit_res}"
		print(fit_summary_str)
		return
################################################################
# __main__ method for execution of other methods
################################################################
if __name__ == "__main__":
	"""
	This is the main statement where the user can execute subroutines to produce figures and/or output files.
	The default pattern will be to set a number of iterations to perform within the script to produce a 
	resolution plot as a function of number of IBD events detected. This can be performed without passing 
	parameters from the command line and setting the number of iterations to perform here in the __main__ method.
	Alternatively, the user can call the output file and iteration number from the command line and execute from a 
	shell script, saving the output as a .txt file for use in a separate script for producing figures. In this case,
	only one fit will be performed per data set.
	"""
	da = DirectionalAnalysis()
	# TODO: name parameters for selecting fit method (iterative in script or call single fit.) Use these to select fit method and apply
	# Default method should be iterative fitting
	if len(sys.argv) > 1:# TODO: add exception handling for too few arguments, or unparsable arguments
		output_fname = sys.argv[1]
		iter_num = sys.argv[2]
		da.doSingleFit(iter_num, output_fname)#, True, True)
	else:# set options for doIterativeFits and call this method for else
		itnums = 5
		plot_name = f"ang_res_plot_{itnums}iters.pdf"
		output_fname = f"fit_params_{itnums}iters.txt"
		da.doIterativeFits(itnums, plot_name, output_fname, True, False, False, True)
	# any other commands go here
	sys.exit()
# dumping ground for unused code



# directional_analysis
directional_analysis.py
# Description
A Python script that uses Pandas to read data files from a RATPAC2 simulation and bin positions of neutron captures on lithium 6 relative to the position of the IBD vertex in a 2-D segmented detector. The binning of a simulated data set is compared to a reference data file containing 10,000 IBD events which are filtered to analyze and compare captures on Li-6 to determine incoming antineutrino direction using a Frobenius norm difference (FND) analysis. The FND minimum is then passed to a fitter to find the angle of the simulated set by fitting to a |sin(x-x0)| function, where x is the reference angle and x0 is the angle found by the fitter which can then be compared to MC truth from the simulated data set to assess the accuracy of the fit. 
This script will also perform these fits a specified number of times on an array of different data sets containing 10, 100, 1k and 10k IBD events in order to find the angular resolution at 1 sigma confidence as a function of number of IBD events analyzed. All output logs and fits are saved to text (.txt) and plots (.pdf) respectively, and the main plot and output file are saved in the ./fits/ directory.
# Version History
v 0.1.0 Initial Commit 18 JUL 2025

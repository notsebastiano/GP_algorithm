
'''
COPYRIGHT sebastiano.bontorin@unitn.it
'''



import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def td_embedding(data,emb,tau):
	'''
	Time delay embedding of timeseries of scalars
	Args:
		timeseries: array of scalars
		emb: (int) embedding dimension
		tau = (int) time delay between values in phase space reconstruction
	Returns:
		array of embedded vectors:
		[x[i],x[i+tau],x[i+2*tau],...,x[i + (m-1)*tau]]
	'''
	indexes = np.arange(0,emb,1)*tau
	return np.array([data[indexes +i] for i in range(len(data)-(emb-1)*tau)])


def logarithmic_r(min_n, max_n, factor):
	'''
	Creates array of values distributed such as log(values) is an array of
	evenly spaced (space between values = log(factor)) values between log(min_n) and log(max_n)
	Args:
		arg1: min_n: minimum value
		arg2: max_n: maximum value ( > arg1 )
		factor: log(factor) is the space between values
	Returns:
		min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
	'''

	if max_n <= min_n:
		raise ValueError("arg1 has to be < arg2")
	if factor <= 1:
		raise ValueError("factor(arg3) has to be > 1")
	max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
	return np.array([min_n * (factor ** i) for i in range(max_i + 1)])


def grassberg_procaccia(data,emb_dim,time_delay,plot = None):
	'''
	Implementation of the Gassberger-Procaccia algorithm to estimate the
	correlation dimension of a set of points in an m-dimensional space.

	This code takes in input a timeseries of scalar values and the embedding dimension + time delay
	necessary to perform a time-delay embedding in phase space to reconstruct the attractor

	Args:
		data: array of scalars - a timeseries
		emb_dim: (int) embedding dimension
		time_delay = (int) time delay between values in phase space reconstruction
	Kwargs:
		plot: if set to True: plots the logarithm of the correlation
		sums against the logarithm of the set of values of r considered in the algorithm

	r is the scaling factor, it tells the threshold distance between points. if we have a plateau
	of local slopes means that we are in a scaling range.

	Returns:
		Correlation dimension (scalar)

	'''

	# Phase space points reconstructed via time-delay embedding
	orbit = td_embedding(data, emb_dim, time_delay)
	n_points = len(orbit)

	# Timeseries standard deviation
	data_std = np.std(data)

	# Generate a series of r distances evenly spaced in log scale, these are 
	# generated starting from the timeseries of scalars standard deviation.
	# The r distance is a scalar used to find the fraction of points in phase space for which
	# the euclidean distance between them is smaller than r
	r_vals = logarithmic_r(0.1 * data_std, 0.7 * data_std, 1.03)

	distances = np.zeros(shape=(n_points,n_points))
	r_matrix_base = np.zeros(shape=(n_points,n_points))

	# Euclidean distance of points in phase space
	for i in range(n_points):
		for j in range(i,n_points):
			distances[i][j] = np.linalg.norm(orbit[i]-orbit[j])
			r_matrix_base[i][j] = 1

	# Correlation sum
	C_r = []
	for r in r_vals:
		r_matrix = r_matrix_base*r
		heavi_matrix = np.heaviside( r_matrix - distances, 0)
		corr_sum = (2/float(n_points*(n_points-1)))*np.sum(heavi_matrix)
		C_r.append(corr_sum)

	#strong assumption: the log-log plot is assumed to be a smooth, monotonic function,
	#hence the slope in the scaling region should be the maximum gradient ( in this case
	#is taken as the mean of the last five maximum gradients as they are calculated for every point )
	gradients = np.gradient(np.log2(C_r),np.log2(r_vals))
	gradients.sort()
	D = np.mean(gradients[-5:])

	if plot:
		# plot the trend of Cr)
		plt.plot(np.log2(r_vals),np.log2(C_r))
		plt.xlabel("Distance r")
		plt.ylabel("C(r)")
		plt.title("Correlation sum in log2-log2 plot. Dimension D is "+str(round(D,2)))
		plt.show()
	
	return D


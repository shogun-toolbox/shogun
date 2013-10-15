#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from pylab import show, imshow

def simulate_data (n,p):
	from modshogun import SparseInverseCovariance
	import numpy as np

	#create a random pxp covariance matrix
	cov = np.random.normal(size=(p,p))

	#generate data set with multivariate Gaussian distribution
	mean = [0] * p
	data = np.random.multivariate_normal(mean, cov, n)

	return data

def inverse_covariance (data,lc):
	from modshogun import SparseInverseCovariance
	from numpy import dot

	sic = SparseInverseCovariance()

	#by default cov() expects each row to represent a variable, with observations in the columns
	cov = np.cov(data.T)

	max_cov = cov.max()
	min_cov = cov.min()

	#compute inverse conariance matrix
	Si = sic.estimate(cov,lc)

	return Si


def draw_graph(sic, subplot):
	import numpy as np
	import networkx as nx

	#create list of edges
	#an egde means there is a dependency between variables
	#0 value in sic matrix mean independent variables given all the other variables
	p = sic.shape[0]
	X, Y = np.meshgrid(range(p), range(p))
	graph = np.array((X[sic != 0], Y[sic != 0])).T

	# extract nodes from graph
	nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

    # create networkx graph
	G=nx.Graph()

    # add nodes
	for node in nodes:
		G.add_node(node)

    # add edges
	for edge in graph:
		G.add_edge(edge[0], edge[1])

    # draw graph
	nx.draw(G, ax=subplot)

    # show graph
	return graph


if __name__=='__main__':


	#edit here for your own simulation
	num_observations = 100
	num_variables = 11
	penalties = [0.00001, 0.05, 0.1, 0.5, 1, 2]

	columns = len(penalties)

	#plot the heat map and the graphs of dependency between variables
	#for different penaltiy values
	f, axarr = plt.subplots(2, columns)
	f.suptitle('Inverse Covariance Estimation\nfor ' +str(num_variables)+' variables and '+str(num_observations)+' observations', fontsize=20)

	data = simulate_data (num_observations, num_variables)
	print data.shape

	column = -1;
	for p in penalties:
		column = column + 1

		sic = inverse_covariance (data,p)

		i = axarr[0, column].imshow(sic, cmap="hot", interpolation='nearest')
		axarr[0, column].set_title('penalty='+str(p), fontsize=10)
		graph = draw_graph(sic, plt.subplot(2, columns, column + columns + 1))
		axarr[1, column].set_title(str((len(graph) - num_variables)/2) + ' depedences', fontsize=10)


	f.subplots_adjust(right=0.8)
	cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
	f.colorbar(i, cax=cbar_ax)

	show();



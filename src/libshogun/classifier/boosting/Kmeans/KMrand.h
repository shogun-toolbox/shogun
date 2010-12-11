//----------------------------------------------------------------------
//	File:		KMrand.h
//	Programmer:	Sunil Arya and David Mount
//	Last modified:	03/27/02
//	Description:	Basic include file for random point generators
//----------------------------------------------------------------------
// Copyright (C) 2004-2005 David M. Mount and University of Maryland
// All Rights Reserved.
// 
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or (at
// your option) any later version.  See the file Copyright.txt in the
// main directory.
// 
// The University of Maryland and the authors make no representations
// about the suitability or fitness of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//----------------------------------------------------------------------

#ifndef KM_RAND_H
#define KM_RAND_H

//----------------------------------------------------------------------
//  Basic includes
//----------------------------------------------------------------------
#include <cstdlib>			// standard C++ includes
#include <math.h>			// math routines
#include "KMeans.h"			// KMeans includes

//----------------------------------------------------------------------
//  Globals
//----------------------------------------------------------------------
extern	int	kmIdum;			// used for random number generation

//----------------------------------------------------------------------
//  External entry points
//----------------------------------------------------------------------

int kmRanInt(			// random integer
	int		n);		// in the range [0,n-1]

double kmRanUnif(		// random uniform in [lo,hi]
	double		lo = 0.0,
	double		hi = 1.0);

void kmUniformPts(		// uniform distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim);		// dimension

void kmGaussPts(			// Gaussian distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	double		std_dev);	// standard deviation

void kmCoGaussPts(		// correlated-Gaussian distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	double		correlation);	// correlation

void kmLaplacePts(		// Laplacian distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim);		// dimension

void kmCoLaplacePts(		// correlated-Laplacian distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	double		correlation);	// correlation

void kmClusGaussPts(		// clustered-Gaussian distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	int		n_col,		// number of colors (clusters)
	bool		new_clust = true,   // generate new cluster centers
	double		std_dev = 0.1,	    // std deviation within clusters
	double*		clus_sep = NULL);   // cluster separation (returned)

KMpointArray kmGetCGclusters(); // get clustered-gauss cluster centers

void kmClusOrthFlats(           // clustered along orthogonal flats
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	int		n_col,		// number of colors
	bool		new_clust,	// generate new clusters.
	double		std_dev,	// standard deviation within clusters
	int		max_dim);	// maximum dimension of the flats

void kmClusEllipsoids(		// clustered around ellipsoids
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	int		n_col,		// number of colors
	bool		new_clust,	// generate new clusters.
	double		std_dev_small,	// small standard deviation
	double		std_dev_lo,	// low standard deviation for ellipses
	double		std_dev_hi,	// high standard deviation for ellipses
	int		max_dim);	// maximum dimension of the flats

void kmMultiClus(		// multi-sized clusters
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	int		&k,		// number of clusters (returned)
	double		base_dev);	// base standard deviation

#endif

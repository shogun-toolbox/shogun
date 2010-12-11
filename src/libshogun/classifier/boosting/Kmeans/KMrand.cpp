//----------------------------------------------------------------------
//	File:		KMrand.cpp
//	Programmer:	Sunil Arya and David Mount
//	Last modified:	05/14/04
//	Description:	Routines for random point generation
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

#include "KMrand.h"			// random generator declarations

#ifdef WIN32				// Visual C++ (no srandom/random)
void srandom(unsigned int seed) { srand(seed); }
long random(void) { return long(rand()); }
#endif

//----------------------------------------------------------------------
//  Globals
//----------------------------------------------------------------------
int	kmIdum = 0;			// used for random number generation

//------------------------------------------------------------------------
//	kmRan0 - (safer) uniform random number generator
//
//	The code given here is taken from "Numerical Recipes in C" by
//	William Press, Brian Flannery, Saul Teukolsky, and William
//	Vetterling. The task of the code is to do an additional randomizing
//	shuffle on the system-supplied random number generator to make it
//	safer to use. 
//
//	Returns a uniform deviate between 0.0 and 1.0 using the
//	system-supplied routine "random()". Set kmIdum to any negative value
//	to initialise or reinitialise the sequence.
//------------------------------------------------------------------------

static double kmRan0()
{
    int j;

    static double y, maxran, v[98];	// The exact number 98 is unimportant
    static int iff = 0;

    // As a precaution against misuse, we will always initialize on the first
    // call, even if "kmIdum" is not set negative. Determine "maxran", the
    // next integer after the largest representable value of type int. We
    // assume this is a factor of 2 smaller than the corresponding value of
    // type unsigned int. 

    if (kmIdum < 0 || iff == 0) {	// initialize
		/* compute maximum random number */
#ifdef WIN32				// Microsoft Visual C++
	maxran = RAND_MAX;
#else
	unsigned i, k;
	i = 2;
	do {
	    k = i;
	    i <<= 1;
	} while (i);
	maxran = (double) k;
#endif
 	iff = 1;
  
	srandom(kmIdum);
	kmIdum = 1;

	for (j = 1; j <= 97; j++)	// exercise the system routine
	    random();			// (value intentionally ignored)

	for (j = 1; j <= 97; j++)	// Then save 97 values and a 98th
	    v[j] = random();
	y = random();
     }

    // This is where we start if not initializing. Use the previously saved
    // random number y to get an index j between 1 and 97. Then use the
    // corresponding v[j] for both the next j and as the output number. */

    j = 1 + (int) (97.0 * (y / maxran));
    y = v[j];
    v[j] = random();			// Finally, refill the table entry
					// with the next random number from
					// "random()" 
    return(y / maxran);
}

//------------------------------------------------------------------------
//  kmRanInt - generate a random integer from {0,1,...,n-1}
//
//	If n == 0, then -1 is returned.
//------------------------------------------------------------------------

int kmRanInt(
    int                 n)
{
    int r = (int) (kmRan0()*n);
    if (r == n) r--;			// (in case kmRan0() == 1 or n == 0)
    return r;
}

//------------------------------------------------------------------------
//  kmRanUnif - generate a random uniform in [lo,hi]
//------------------------------------------------------------------------

double kmRanUnif(
    double		lo,
    double		hi)
{
    return kmRan0()*(hi-lo) + lo;
}

//------------------------------------------------------------------------
//  kmRanGauss - Gaussian random number generator
//	Returns a normally distributed deviate with zero mean and unit
//	variance, using kmRan0() as the source of uniform deviates.
//------------------------------------------------------------------------

static double kmRanGauss()
{
    static int iset=0;
    static double gset;

    if (iset == 0) {			// we don't have a deviate handy
	double v1, v2;
	double r = 2.0;
	while (r >= 1.0) {
	    //------------------------------------------------------------
	    // Pick two uniform numbers in the square extending from -1 to
	    // +1 in each direction, see if they are in the circle of radius
	    // 1.  If not, try again 
	    //------------------------------------------------------------
	    v1 = kmRanUnif(-1, 1);
	    v2 = kmRanUnif(-1, 1);
	    r = v1 * v1 + v2 * v2;
	}
        double fac = sqrt(-2.0 * log(r) / r);
	//-----------------------------------------------------------------
	// Now make the Box-Muller transformation to get two normal
	// deviates.  Return one and save the other for next time.
	//-----------------------------------------------------------------
	gset = v1 * fac;
	iset = 1;		    	// set flag
	return v2 * fac;
    }
    else {				// we have an extra deviate handy
	iset = 0;			// so unset the flag
	return gset;			// and return it
    }
}

//------------------------------------------------------------------------
//  kmRanLaplace - Laplacian random number generator
//	Returns a Laplacian distributed deviate with zero mean and
//	unit variance, using kmRan0() as the source of uniform deviates. 
//
//		prob(x) = b/2 * exp(-b * |x|).
//
//	b is chosen to be sqrt(2.0) so that the variance of the Laplacian
//	distribution [2/(b^2)] becomes 1. 
//------------------------------------------------------------------------

static double kmRanLaplace() 
{
    const double b = 1.4142136;

    double laprand = -log(kmRan0()) / b;
    double sign = kmRan0();
    if (sign < 0.5) laprand = -laprand;
    return(laprand);
}

//----------------------------------------------------------------------
//  kmUniformPts - Generate uniformly distributed points
//	A uniform distribution over [-1,1].
//----------------------------------------------------------------------

void kmUniformPts(		// uniform distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim)		// dimension
{
    for (int i = 0; i < n; i++) {
	for (int d = 0; d < dim; d++) {
	    pa[i][d] = (KMcoord) (kmRanUnif(-1,1));
	}
    }
}

//----------------------------------------------------------------------
//  kmGaussPts - Generate Gaussian distributed points
//	A Gaussian distribution with zero mean and the given standard
//	deviation.
//----------------------------------------------------------------------

void kmGaussPts(			// Gaussian distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	double		std_dev)	// standard deviation
{
    for (int i = 0; i < n; i++) {
	for (int d = 0; d < dim; d++) {
	    pa[i][d] = (KMcoord) (kmRanGauss() * std_dev);
	}
    }
}

//----------------------------------------------------------------------
//  kmLaplacePts - Generate Laplacian distributed points
//	Generates a Laplacian distribution (zero mean and unit variance).
//----------------------------------------------------------------------

void kmLaplacePts(		// Laplacian distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim)		// dimension
{
    for (int i = 0; i < n; i++) {
	for (int d = 0; d < dim; d++) {
            pa[i][d] = (KMcoord) kmRanLaplace();
	}
    }
}

//----------------------------------------------------------------------
//  kmCoGaussPts - Generate correlated Gaussian distributed points
//	Generates a Gauss-Markov distribution of zero mean and unit
//	variance.
//----------------------------------------------------------------------

void kmCoGaussPts(		// correlated-Gaussian distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	double		correlation)	// correlation
{
    double std_dev_w = sqrt(1.0 - correlation * correlation);
    for (int i = 0; i < n; i++) {
	double previous = kmRanGauss();
	pa[i][0] = (KMcoord) previous;
	for (int d = 1; d < dim; d++) {
	    previous = correlation*previous + std_dev_w*kmRanGauss();
	    pa[i][d] = (KMcoord) previous;
	} 
    }
}

//----------------------------------------------------------------------
//  kmCoLaplacePts - Generate correlated Laplacian distributed points
//	Generates a Laplacian-Markov distribution of zero mean and unit
//	variance.
//----------------------------------------------------------------------

void kmCoLaplacePts(		// correlated-Laplacian distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	double		correlation)	// correlation
{
    double wn;
    double corr_sq = correlation * correlation;

    for (int i = 0; i < n; i++) {
	double previous = kmRanLaplace();
	pa[i][0] = (KMcoord) previous;
	for (int d = 1; d < dim; d++) {
	    double temp = kmRan0();
	    if (temp < corr_sq)
		wn = 0.0;
	    else
		wn = kmRanLaplace();
	    previous = correlation * previous + wn;
	    pa[i][d] = (KMcoord) previous;
        } 
    }
}

//----------------------------------------------------------------------
//  kmClusGaussPts - Generate clusters of Gaussian distributed points
//	Cluster centers are uniformly distributed over [-1,1], and the
//	standard deviation within each cluster is fixed.
//
//	Note: Once cluster centers have been set, they are not changed,
//	unless new_clust = true.  This is so that subsequent calls generate
//	points from the same distribution.  It follows, of course, that any
//	attempt to change the dimension or number of clusters without
//	generating new clusters is asking for trouble.
//
//	Note: Cluster centers are not generated by a call to uniformPts().
//	Although this could be done, it has been omitted for
//	compatibility with kmClusGaussPts() in the colored version,
//	rand_c.cc.
//
//	As a side effect we compute the cluster separation. This is
//	defined to be min_{i != j} dist(cc[i],cc[j])/std, where cc[i] is
//	the i-th cluster center and std is the global deviation.  It is
//	defined to be the square root of the trace of the covariance
//	matrix, which is equivalent to the std_dev for each coordinate
//	times sqrt(d).
//----------------------------------------------------------------------
static KMpointArray cgClusters = NULL;	// cluster storage

void kmClusGaussPts(		// clustered-Gaussian distribution
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	int		n_col,		// number of colors
	bool		new_clust,	// generate new clusters.
	double		std_dev,	// standard deviation within clusters
	double*		clus_sep)	// cluster separation (returned)
{
    if (cgClusters == NULL || new_clust) {// need new cluster centers
	if (cgClusters != NULL)		// clusters already exist
	    kmDeallocPts(cgClusters);	// get rid of them
	cgClusters = kmAllocPts(n_col, dim);
					// generate cluster center coords
	for (int i = 0; i < n_col; i++) {
	    for (int d = 0; d < dim; d++) {
		cgClusters[i][d] = (KMcoord) kmRanUnif(-1,1);
	    }
	}
    }

    double minDist = double(dim);	// minimum inter-center sq'd distance
    for (int i = 0; i < n_col; i++) {	// compute minimum separation
	for (int j = i+1; j < n_col; j++) {
	    double dist = kmDist(dim, cgClusters[i], cgClusters[j]);
	    if (dist < minDist) minDist = dist;
	}
    }
					// cluster separation
    if (clus_sep != NULL)
	*clus_sep = sqrt(minDist)/(sqrt(double(dim))*std_dev);

    for (int i = 0; i < n; i++) {
	int c = kmRanInt(n_col);	// generate cluster index
	for (int d = 0; d < dim; d++) {
          pa[i][d] = (KMcoord) (std_dev*kmRanGauss() + cgClusters[c][d]);
	}
    }
}

KMpointArray kmGetCGclusters()	// get clustered gauss cluster centers
{
    return cgClusters;
}

//----------------------------------------------------------------------
//  kmClusOrthFlats - points clustered along orthogonal flats
//
//	This distribution consists of a collection points clustered
//	among a collection of low dimensional flats (hyperplanes) in
//	the hypercube [-1,1]^d.  A set of n_col orthogonal flats are
//	generated, each whose dimension is a random number between 1
//	and max_dim.  The points are evenly distributed among the clusters.
//	For each cluster, we generate points uniformly distributed along
//	the flat within the hypercube.
//
//	This is done as follows.  Each cluster is defined by a d-element
//	control vector whose components are either:
//	
//		CO_FLAG	indicating that this component is to be generated
//			uniformly in [-1,1],
//		x 	a value other than CO_FLAG in the range [-1,1],
//			which indicating that this coordinate is to be
//			generated as x plus a Gaussian random deviation
//			with the given standard deviation.
//			
//	The number of zero components is the dimension of the flat, which
//	is a random integer in the range from 1 to max_dim.  The points
//	are disributed between clusters in nearly equal sized groups.
//
//	Note: Once cluster centers have been set, they are not changed,
//	unless new_clust = true.  This is so that subsequent calls generate
//	points from the same distribution.  It follows, of course, that any
//	attempt to change the dimension or number of clusters without
//	generating new clusters is asking for trouble.
//
//	To make this a bad scenario at query time, query points should be
//	selected from a different distribution, e.g. uniform or Gaussian.
//
//	We use a little programming trick to generate groups of roughly
//	equal size.  If n is the total number of points, and n_col is
//	the number of clusters, then the c-th cluster (0 <= c < n_col)
//	is given floor((n+c)/n_col) points.  It can be shown that this
//	will exactly consume all n points.
//
//----------------------------------------------------------------------

void kmClusOrthFlats(		// clustered along orthogonal flats
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	int		n_col,		// number of colors
	bool		new_clust,	// generate new clusters.
	double		std_dev,	// standard deviation within clusters
	int		max_dim)	// maximum dimension of the flats
{
    const double CO_FLAG = 999;			// special flag value
    static KMpointArray control = NULL;	// control vectors

    if (control == NULL || new_clust) {		// need new cluster centers
	if (control != NULL) {			// clusters already exist
	    kmDeallocPts(control);		// get rid of them
	}
	control = kmAllocPts(n_col, dim);

	for (int c = 0; c < n_col; c++) {	// generate clusters
	    int n_dim = 1 + kmRanInt(max_dim);	// number of dimensions in flat
	    for (int d = 0; d < dim; d++) {	// generate side locations
						// prob. of picking next dim
	    	double Prob = ((double) n_dim)/((double) (dim-d));
		if (kmRan0() < Prob) {		// add this one to flat
		    control[c][d] = CO_FLAG;	// flag this entry
		    n_dim--;			// one fewer dim to fill
		}
		else {				// don't take this one
		    control[c][d] = kmRanUnif(-1,1);// random value in [-1,1]
		}
	    }
	}
    }

    int next = 0;				// next slot to fill
    for (int c = 0; c < n_col; c++) {		// generate clusters
	int pick = (n+c)/n_col;			// number of points to pick
	for (int i = 0; i < pick; i++) {
	    for (int d = 0; d < dim; d++) {
		if (control[c][d] == CO_FLAG)	// dimension on flat
        	    pa[next][d] = (KMcoord) kmRanUnif(-1,1);
		else				// dimension off flat
        	    pa[next][d] =
			(KMcoord) (std_dev*kmRanGauss() + control[c][d]);
	    }
	    next++;
	}
    }
}

//----------------------------------------------------------------------
//  kmClusEllipsoids - points clustered around ellipsoids
//
//	This distribution consists of a collection points clustered
//	among a collection of low dimensional ellipsoids in the
//	hypercube [-1,1]^d.  The objective is to model distributions
//	in which the points are distributed in lower dimensional
//	subspaces, and within this lower dimensional space the points
//	are distributed with a Gaussian distribution (with no
//	correlation between the dimensions).
//
//	The distribution is given the number of clusters or "colors"
//	(n_col), maximum number of dimensions (max_dim) of the lower
//	dimensional subspace, a "small" standard deviation (std_dev_small),
//	and a "large" standard deviation range (std_dev_lo, std_dev_hi).
//
//	The algorithm generates n_col cluster centers uniformly from the
//	hypercube [-1,1]^d.  For each cluster, it selects the dimension
//	of the subspace as a random number r between 1 and max_dim.
//	These are the dimensions of the ellipsoid.  Then it generates
//	a d-element std dev vector whose entries are the standard
//	deviation for the coordinates of each cluster in the distribution.
//	Among the d-element control vector, r randomly chosen values are
//	chosen uniformly from the range [std_dev_lo, std_dev_hi].  The
//	remaining values are set to std_dev_small.
//
//	Note that kmClusGaussPts is a special case of this in which
//	max_dim = 0, and std_dev = std_dev_small.
//
//	If the flag new_clust is set, then new cluster centers are
//	generated.
//
//----------------------------------------------------------------------

void kmClusEllipsoids(		// clustered around ellipsoids
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	int		n_col,		// number of colors
	bool		new_clust,	// generate new clusters.
	double		std_dev_small,	// small standard deviation
	double		std_dev_lo,	// low standard deviation for ellipses
	double		std_dev_hi,	// high standard deviation for ellipses
	int		max_dim)	// maximum dimension of the flats
{
    static KMpointArray clusters = NULL;	// cluster centers
    static KMpointArray stdDev = NULL;		// standard deviations

    if (clusters == NULL || new_clust) {	// need new cluster centers
	if (clusters != NULL)			// clusters already exist
	    kmDeallocPts(clusters);		// get rid of them
	if (stdDev != NULL)			// std deviations already exist
	    kmDeallocPts(stdDev);		// get rid of them

	clusters = kmAllocPts(n_col, dim);	// alloc new clusters and devs
	stdDev   = kmAllocPts(n_col, dim);

	for (int i = 0; i < n_col; i++) {	// gen cluster center coords
	    for (int d = 0; d < dim; d++) {
		clusters[i][d] = (KMcoord) kmRanUnif(-1,1);
	    }
	}
	for (int c = 0; c < n_col; c++) {	// generate cluster std dev
	    int n_dim = 1 + kmRanInt(max_dim);	// number of dimensions in flat
	    for (int d = 0; d < dim; d++) {	// generate std dev's
						// prob. of picking next dim
	    	double Prob = ((double) n_dim)/((double) (dim-d));
		if (kmRan0() < Prob) {		// add this one to ellipse
						// generate random std dev
		    stdDev[c][d] = kmRanUnif(std_dev_lo, std_dev_hi);
		    n_dim--;			// one fewer dim to fill
		}
		else {				// don't take this one
		    stdDev[c][d] = std_dev_small;// use small std dev
		}
	    }
	}
    }

    int next = 0;				// next slot to fill
    for (int c = 0; c < n_col; c++) {		// generate clusters
	int pick = (n+c)/n_col;			// number of points to pick
	for (int i = 0; i < pick; i++) {
	    for (int d = 0; d < dim; d++) {
        	pa[next][d] = (KMcoord)
			(stdDev[c][d]*kmRanGauss() + clusters[c][d]);
	    }
	    next++;
	}
    }
}

//----------------------------------------------------------------------
//  kmMultiClus - multi-sized clusters
//	This distribution is designed to be a challenge for clustering
//	algorithm.  It consists of a clusters of varying sizes, located
//	within the cube [-1,1]^d.  The cluster centers are uniformly
//	distributed.  We generate clusters one by one.  For each
//	cluster, the size of the cluster m is chosen so that the
//	probability of generating a cluster of size 2^i is 1/2^i.
//
//	We want each cluster to have a similar total squared distortion
//	(variance).  Thus a group of size m is generated from a gaussian
//	distribution with a standard deviation of B*sqrt(1/m).  We call
//	B the base standard deviation, which is given as an argument.
//
//	The expected number of clusters is (lg n)-2, but there is a very
//	high variance here.  The total distortion in each dimension then
//	is B^2*((lg n)-2) (but don't trust me here, since this is just
//	based on back-of- the-envelope computations).  Hence the expected
//	average distortion, for n points in dim d would be:
//
//		d*B^2*((log n)-2)/n
//
//	The reason that this is challenging is that each cluster has an
//	equal claim to a center (since distortions are similar) but many
//	sampling based methods will favor placing clusters in the
//	relatively few clusters with a large number of points.
//----------------------------------------------------------------------

void kmMultiClus(		// multi-sized clusters
	KMpointArray	pa,		// point array (modified)
	int		n,		// number of points
	int		dim,		// dimension
	int		&k,		// number of clusters (returned)
	double		base_dev)	// base standard deviation
{
    int next = 0;			// next point in array
    int nSamp = 0;			// number of points sampled
    k = 0;				// number of clusters generated
    KMpoint clusCenter = kmAllocPt(dim); // allocate center storage
    while (nSamp < n) {			// until we have sampled enough
	int remain = n - nSamp;		// number remaining to sample
	int clusSize = 2;
					// repeatedly double cluster size
					// with prob 1/2
	while ((clusSize < remain) && (kmRan0() < 0.5))
	    clusSize *= 2;
					// don't exceed upper limit
	if (clusSize > remain) clusSize = remain;

					// generate center uniformly
	for (int d = 0; d < dim; d++) {
	    clusCenter[d] = (KMcoord) kmRanUnif(-1,1);
	}
    					// desired std dev for cluster
	double stdDev = base_dev*sqrt(1.0/clusSize);
					// generate cluster points
	for (int i = 0; i < clusSize; i++) {
	    for (int d = 0; d < dim; d++) {
		pa[next][d] = (KMcoord) (stdDev*kmRanGauss()+clusCenter[d]);
	    }
	    next++;
	}
	nSamp += clusSize;		// update number sampled
	k++;				// one more cluster
    }
    kmDeallocPt(clusCenter);
}

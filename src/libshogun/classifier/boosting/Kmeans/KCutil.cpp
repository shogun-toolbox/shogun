//----------------------------------------------------------------------
//	File:		KCutil.h
//	Programmer:	David Mount
//	Last modified:	03/27/02
//	Description:	Utilities for kc-tree splitting rules
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

#include "KCutil.h"			// kc-utility declarations

//----------------------------------------------------------------------
//	NOTE: Virtually all point indexing is done through an
//	index (i.e. permutation) array pidx.  Consequently,
//	a reference to the d-th coordinate of the i-th point
//	is pa[pidx[i]][d].  The macro PA(i,d) is a shorthand
//	for this.
//----------------------------------------------------------------------
					// standard 2-d indirect indexing
#define PA(i,d) 	(pa[pidx[(i)]][(d)])
					// accessing a single point
#define PP(i)	 	(pa[pidx[(i)]])
					// swap two points
#define PASWAP(a,b) { int tmp = pidx[a];\
                    pidx[a] = pidx[b];\
                    pidx[b] = tmp; }

//----------------------------------------------------------------------
//  Constants
//----------------------------------------------------------------------

const double ERR = 0.001;		// a small value

//----------------------------------------------------------------------
//  kmEnclRect, kmEnclCube
//	These utilities compute the smallest rectangle and cube enclosing
//	a set of points, respectively.
//----------------------------------------------------------------------

void kmEnclRect(
    KMpointArray	pa,		// point array
    KMidxArray		pidx,		// point indices
    int			n,		// number of points
    int			dim,		// dimension
    KMorthRect		&bnds)		// bounding cube (returned)
{
    for (int d = 0; d < dim; d++) {	// find smallest enclosing rectangle
	KMcoord lo_bnd = PA(0,d);	// lower bound on dimension d
	KMcoord hi_bnd = PA(0,d);	// upper bound on dimension d
        for (int i = 0; i < n; i++) {
	    if (PA(i,d) < lo_bnd) lo_bnd = PA(i,d);
	    else if (PA(i,d) > hi_bnd) hi_bnd = PA(i,d);
	}
	bnds.lo[d] = lo_bnd;
	bnds.hi[d] = hi_bnd;
    }
}

//----------------------------------------------------------------------
//  kmSpread - find spread along given dimension
//  kmMinMax - find min and max coordinates along given dimension
//  kmMaxSpread - find dimension of max spread
//----------------------------------------------------------------------

KMcoord kmSpread(		// compute point spread along dimension
    KMpointArray	pa,		// point array
    KMidxArray		pidx,		// point indices
    int			n,		// number of points
    int			d)		// dimension to check
{
    KMcoord min = PA(0,d);		// compute max and min coords
    KMcoord max = PA(0,d);
    for (int i = 1; i < n; i++) {
	KMcoord c = PA(i,d);
	if (c < min) min = c;
	else if (c > max) max = c;
    }
    return (max - min);			// total spread is difference
}

void kmMinMax(			// compute min and max coordinates along dim
    KMpointArray	pa,		// point array
    KMidxArray		pidx,		// point indices
    int			n,		// number of points
    int			d,		// dimension to check
    KMcoord		&min,		// minimum value (returned)
    KMcoord		&max)		// maximum value (returned)
{
    min = PA(0,d);			// compute max and min coords
    max = PA(0,d);
    for (int i = 1; i < n; i++) {
	KMcoord c = PA(i,d);
	if (c < min) min = c;
	else if (c > max) max = c;
    }
}

//----------------------------------------------------------------------
//  kmPlaneSplit - split point array about a cutting plane
//	Split the points in an array about a given plane along a
//	given cutting dimension.  On exit, br1 and br2 are set so
//	that:
//	
//		pa[ 0 ..br1-1] <  cv
//		pa[br1..br2-1] == cv
//		pa[br2.. n -1] >  cv
//
//	All indexing is done indirectly through the index array pidx.
//----------------------------------------------------------------------

void kmPlaneSplit(		// split points by a plane
    KMpointArray	pa,		// points to split
    KMidxArray		pidx,		// point indices
    int			n,		// number of points
    int			d,		// dimension along which to split
    KMcoord		cv,		// cutting value
    int			&br1,		// first break (values < cv)
    int			&br2)		// second break (values == cv)
{
    int l = 0;
    int r = n-1;
    for(;;) {				// partition pa[0..n-1] about cv
	while (l < n && PA(l,d) < cv) l++;
	while (r >= 0 && PA(r,d) >= cv) r--;
	if (l > r) break;
	PASWAP(l,r);
	l++; r--;
    }
    br1 = l;			// now: pa[0..br1-1] < cv <= pa[br1..n-1]
    r = n-1;
    for(;;) {				// partition pa[br1..n-1] about cv
	while (l < n && PA(l,d) <= cv) l++;
	while (r >= br1 && PA(r,d) > cv) r--;
	if (l > r) break;
	PASWAP(l,r);
	l++; r--;
    }
    br2 = l;			// now: pa[br1..br2-1] == cv < pa[br2..n-1]
}

//----------------------------------------------------------------------
//  sl_midpt_split - sliding midpoint splitting rule
//
//	This is a modification of midpt_split, which has the nonsensical
//	name "sliding midpoint".  The idea is that we try to use the
//	midpoint rule, by bisecting the longest side.  If there are
//	ties, the dimension with the maximum spread is selected.  If,
//	however, the midpoint split produces a trivial split (no points
//	on one side of the splitting plane) then we slide the splitting
//	(maintaining its orientation) until it produces a nontrivial
//	split.  For example, if the splitting plane is along the x-axis,
//	and all the data points have x-coordinate less than the x-bisector,
//	then the split is taken along the maximum x-coordinate of the
//	data points.
//
//	Intuitively, this rule cannot generate trivial splits, and
//	hence avoids midpt_split's tendency to produce trees with
//	a very large number of nodes.
//
//----------------------------------------------------------------------

void sl_midpt_split(
    KMpointArray	pa,		// point array
    KMidxArray		pidx,		// point indices (permuted on return)
    const KMorthRect	&bnds,		// bounding rectangle for cell
    int			n,		// number of points
    int			dim,		// dimension of space
    int			&cut_dim,	// cutting dimension (returned)
    KMcoord		&cut_val,	// cutting value (returned)
    int			&n_lo)		// num of points on low side (returned)
{
    int d;

    KMcoord max_length = bnds.hi[0] - bnds.lo[0];
    for (d = 1; d < dim; d++) {		// find length of longest box side
	KMcoord length = bnds.hi[d] - bnds.lo[d];
	if (length  > max_length) {
	    max_length = length;
	}
    }
    KMcoord max_spread = -1;		// find long side with most spread
    for (d = 0; d < dim; d++) {
					// is it among longest?
	if ((bnds.hi[d] - bnds.lo[d]) >= (1-ERR)*max_length) {
					// compute its spread
	    KMcoord spr = kmSpread(pa, pidx, n, d);
	    if (spr > max_spread) {	// is it max so far?
		max_spread = spr;
		cut_dim = d;
	    }
	}
    }
					// ideal split at midpoint
    KMcoord ideal_cut_val = (bnds.lo[cut_dim] + bnds.hi[cut_dim])/2;

    KMcoord min, max;
    kmMinMax(pa, pidx, n, cut_dim, min, max);	// find min/max coordinates

    if (ideal_cut_val < min)		// slide to min or max as needed
	cut_val = min;
    else if (ideal_cut_val > max)
	cut_val = max;
    else
	cut_val = ideal_cut_val;

					// permute points accordingly
    int br1, br2;
    kmPlaneSplit(pa, pidx, n, cut_dim, cut_val, br1, br2);
    //------------------------------------------------------------------
    //	On return:	pa[pidx[[0..br1-1]]   < cut_val
    //			pa[pidx[[br1..br2-1]] = cut_val
    //			pa[pidx[[br2..n-1]]   > cut_val
    //
    //	We can set n_lo to any value in the range [br1..br2] to satisfy
    //	the exit conditions of the procedure.
    //
    //	if ideal_cut_val < min (implying br2 >= 1),
    //		then we select n_lo = 1 (so there is one point on left) and
    //  if ideal_cut_val > max (implying br1 <= n-1),
    //		then we select n_lo = n-1 (so there is one point on right).
    //	Otherwise, we select n_lo as close to n/2 as possible within
    //		[br1..br2].
    //------------------------------------------------------------------
    if (ideal_cut_val < min) n_lo = 1;
    else if (ideal_cut_val > max) n_lo = n-1;
    else if (br1 > n/2) n_lo = br1;
    else if (br2 < n/2) n_lo = br2;
    else n_lo = n/2;
}

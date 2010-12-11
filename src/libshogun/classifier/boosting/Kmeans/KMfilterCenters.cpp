//----------------------------------------------------------------------
//	File:           KMfilterCenters.cc
//	Programmer:     David Mount
//	Last modified:  08/10/2005
//	Description:    Member functions for KMfilterCenters
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

#include "KMfilterCenters.h"
#include "KMrand.h"

					// standard constructor
KMfilterCenters::KMfilterCenters(int k, KMdata& p, double df)
    : KMcenters(k, p) {
    if (p.getKcTree() == NULL) {	// kc-tree not yet built?
      kmError("Building kc-tree", KMwarn);
      p.buildKcTree();			// build it now
    }
    sums	= kmAllocPts(kCtrs, getDim());
    sumSqs	= new double[kCtrs];
    weights	= new int[kCtrs];
    dists	= new double[kCtrs];
    currDist	= KM_HUGE;
    dampFactor	= df;
    invalidate();			// distortions are initially invalid
}
					// copy constructor
KMfilterCenters::KMfilterCenters(const KMfilterCenters& s)
	: KMcenters(s) {
    sums	= kmAllocCopyPts(kCtrs, getDim(), s.sums);
    sumSqs	= kmAllocCopy(kCtrs, s.sumSqs);
    weights	= kmAllocCopy(kCtrs, s.weights);
    dists	= kmAllocCopy(kCtrs, s.dists);
    currDist	= s.currDist;
    dampFactor	= s.dampFactor;
    valid	= s.valid;
}
					// assignment operator
KMfilterCenters& KMfilterCenters::operator=(const KMfilterCenters& s) {
    if (this != &s) {			// avoid self copy (x=x)
					// different sizes?
	if (kCtrs != s.kCtrs || getDim() != s.getDim()) {
	    kmDeallocPts(sums);		// deallocate old storage
	    delete [] sumSqs;
	    delete [] weights;
	    delete [] dists;
	    				// allocate new storage
	    sums    = kmAllocPts(s.kCtrs, s.getDim());
	    sumSqs  = new double[s.kCtrs];
	    weights = new int[s.kCtrs];
	    dists   = new double[s.kCtrs];
	}
	KMcenters& base = *this;	
	base.operator=(s);		// copy base class
					// copy array contents
	kmCopyPts(kCtrs, getDim(), s.sums, sums);
	kmCopy(kCtrs, s.sumSqs, sumSqs);
	kmCopy(kCtrs, s.weights, weights);
	kmCopy(kCtrs, s.dists, dists);
	valid   = s.valid;
    }
    currDist = s.currDist;
    dampFactor = s.dampFactor;
    return *this;
}
    					// virtual destructor
KMfilterCenters::~KMfilterCenters() {
    kmDeallocPts(sums);
    delete [] sumSqs;
    delete [] weights;
	delete [] dists;
}

//----------------------------------------------------------------------
//  computeDistortion
//	This procedure computes the total and individual distortions for
//	a set of center points.  It invokes getNeighbors() on the
//	kc-tree for the point set,which computes the values of weights,
//	sums, and sumSqs, from which the distortion is computed as
//	follows.
//
//	Distortion Computation:
//	-----------------------
//	Assume that some center has been fixed (indexed by j in the code
//	below).  Let SUM_i denote a summation over all (wgt[j])
//	neighbors of the given center.  The data points (p[i]) and
//	center points (c[j]) are vectors, and the product of two vectors
//	means the dot product (u*v = (u.v), u^2 = (u.u)).  The
//	distortion for a single center j, denoted dists[j], is defined
//	to be the sum of squared distances from each point to its
//	closest center,  That is:
//
//	    dists[j] = SUM_i (p[i] - c[j])^2
//		= SUM_i (p[i]^2 - 2*c[j]*p[i] + c[j]^2)
//		= SUM_i p[i]^2 - 2*c[j]*SUM_i p[i] + wgt[j]*c[j]^2
//		= sumSqs[j] - 2*(c[j].sums[j]) + wgt[j]*(c[j]^2)
//
//	Thus the individual distortion can be computed from these
//	quantities.  The total distortion is the sum of the individual
//	distortions.
//----------------------------------------------------------------------

void KMfilterCenters::computeDistortion() // compute distortions
{
    // *kmOut << "------------------------------Computing Distortions" << endl;
    KCtree* t = getData().getKcTree();
    assert(t != NULL);				// tree better exist
    t->getNeighbors(*this);			// get neighbors
    double totDist = 0;
    for (int j = 0; j < kCtrs; j++) {
	double cDotC = 0;			// init: (c[j] . c[j])
	double cDotS = 0;			// init: (c[j] . sum[j])
	for (int d = 0; d < getDim(); d++) {	// compute dot products
	    cDotC += ctrs[j][d] * ctrs[j][d];
	    cDotS += ctrs[j][d] * sums[j][d];
	}
						// final distortion
	dists[j] = sumSqs[j] - 2*cDotS + weights[j]*cDotC;
	totDist += dists[j];
    }
    currDist = totDist;				// save total distortion

    validate();					// distortions are now valid
}

//----------------------------------------------------------------------
//  getAssignments
//	This procedure computes the assignment of points to centers.
//	It simply passes the request along to the associated kc-tree.
//
//	Even though this makes a full traversal of the kc-tree, it does
//	not update the sum or sum of squares, etc., but it does not
//	modify them either.  Thus, we do not change the validation
//	status.
//----------------------------------------------------------------------
//
void KMfilterCenters::getAssignments(	// get point assignments
    KMctrIdxArray	closeCtr,		// closest center per point
    double*		sqDist)			// sq'd dist to center
{
    KCtree* t = getData().getKcTree();
    assert(t != NULL);				// tree better exist
    t->getAssignments(*this, closeCtr, sqDist);	// ask KC tree to do it
}

//----------------------------------------------------------------------
//  moveToCentroid
//	This procedure moves each center point to the centroid of its
//	associated cluster.  We call computeDistortion() if necessary to
//	compute the weights and sums.  The centroid is the weighted
//	average of the sum of neighbors.  Thus the 
//
//	    ctrs[j] = sums[j] / weights[j].
//
//	We generally allow a dampening factor on the motion, which is a
//	floating quantity between 0 (full dampening) and 1 (no
//	dampening).  Given the dampening factor df, the above formula
//	is:
//	    ctrs[j] = (1-df) * ctrs[j] + df * sums[j]/ weights[j]
//
//----------------------------------------------------------------------

void KMfilterCenters::moveToCentroid()	// move center to cluster centroid
{
    if (!valid) computeDistortion();		// compute sums if needed
    for (int j = 0; j < kCtrs; j++) {
	int wgt = weights[j];			// weight of this center
	if (wgt > 0) {				// update only if weight > 0
	    for (int d = 0; d < getDim(); d++) {
    		ctrs[j][d] = (1 - dampFactor) * ctrs[j][d] +
				dampFactor * sums[j][d]/wgt;
	    }
	}
    }
    invalidate();				// distortions now invalid
}

//----------------------------------------------------------------------
//  swapOneCenter
//	Swaps one center point with a sample point.  Optionally we make
//	sure that the new point is not a duplicate of any of the centers
//	(including the point being replaced).
//----------------------------------------------------------------------
void KMfilterCenters::swapOneCenter(		// swap one center
    bool allowDuplicate)			// allow duplicate centers
{
    int rj = kmRanInt(kCtrs);			// index of center to replace
    int dim = getDim();
    KMpoint p = kmAllocPt(dim);			// alloc replacement point
    pts->sampleCtr(p);				// sample a replacement
    if (!allowDuplicate) {			// duplicates not allowed?
        bool dupFound;				// was a duplicate found?
        do {					// repeat until successful
	    dupFound = false;
	    for (int j = 0; j < kCtrs; j++) { 	// search for duplicates
		if (kmEqualPts(dim, p, ctrs[j])) {
		    dupFound = true;
		    pts->sampleCtr(p);		// try again
		    break;
		}
	    }
	} while (dupFound);
    }
    kmCopyPt(dim, p, ctrs[rj]);			// copy sampled point
    if (kmStatLev >= STEP) {			// output swap info
        *kmOut << "\tswapping: ";
        kmPrintPt(p, getDim(), true);
        *kmOut << "<-->Center[" << rj << "]\n";
    }
    kmDeallocPt(p);				// deallocate point storage
    invalidate();				// distortions now invalid
}

//----------------------------------------------------------------------
//  print centers and distortions
//----------------------------------------------------------------------

void KMfilterCenters::print(bool fancy)		// print centers and distortion
{
    for (int j = 0; j < kCtrs; j++) {
	*kmOut << "    " << setw(4) << j << "\t";
	kmPrintPt(ctrs[j], getDim(), true);
	*kmOut << " dist = " << setw(8) << dists[j] << endl;
    }
}

//----------------------------------------------------------------------
//	File:           KMdata.cc
//	Programmer:     David Mount
//	Last modified:  03/27/02
//	Description:    Functions for KMdata
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

#include "KMdata.h"
#include "KMrand.h"			// provides kmRanInt()

					// standard constructor
KMdata::KMdata(int d, int n) : dim(d), maxPts(n), nPts(n) {
    pts = kmAllocPts(n, d);
    kcTree = NULL;
}

KMdata::~KMdata() {			// destructor
    kmDeallocPts(pts);				// deallocate point array
    delete kcTree;				// deallocate kc-tree
}

void KMdata::buildKcTree() {		// build kc-tree for points
    if (kcTree != NULL) delete kcTree;		// destroy existing tree
    kcTree = new KCtree(pts, nPts, dim);	// construct the tree
}

void KMdata::resize(int d, int n) {	// resize point array
    if (d != dim || n != nPts) {		// size change?
	dim = d;
	nPts = n;
	kmDeallocPts(pts);			// deallocate old points
	pts = kmAllocPts(nPts, dim);
    }
    if (kcTree != NULL) {			// kc-tree exists?
	delete kcTree;				// deallocate kc-tree
	kcTree = NULL;
    }
}

//------------------------------------------------------------------------
//  sampleCtr - Sample a center point at random.
//	Generates a randomly sampled center point.
//------------------------------------------------------------------------

void KMdata::sampleCtr(			// sample a center point
    KMcenter	sample)				// where to store sample
{
    int ri = kmRanInt(nPts);			// generate random index
    kmCopyPt(dim, pts[ri], sample);		// copy to destination
}

//------------------------------------------------------------------------
//  sampleCtrs - Sample center points at random.
//	Generates a set of center points by sampling (allowing or
//	disallowing duplicates) from this point set.  It is assumed that
//	the point storage has already been allocated.
//------------------------------------------------------------------------

void KMdata::sampleCtrs(			// sample points randomly
    KMcenterArray	sample,			// where to store sample
    int			k,			// number of points to sample
    bool		allowDuplicate)		// sample with replacement?
{
    if (!allowDuplicate)			// duplicates not allowed
	assert(k <= nPts);			// can't do more than nPts

    int* sampIdx = new int[k];			// allocate index array

    for (int i = 0; i < k; i++) {		// sample each point of sample
	int ri = kmRanInt(nPts);		// random index in pts
	if (!allowDuplicate) {			// duplicates not allowed?
	    bool dupFound;			// duplicate found flag
    	    do {				// repeat until successful
		dupFound = false;
		for (int j = 0; j < i; j++) { 	// search for duplicates
		    if (sampIdx[j] == ri) {	// duplicate found
			dupFound = true;
			ri = kmRanInt(nPts);	// try again
			break;
		    }
	    	}
	    } while (dupFound);
	}
	kmCopyPt(dim, pts[ri], sample[i]);	// copy sample point
	sampIdx[i] = ri;			// save index
    }
    delete [] sampIdx;
}

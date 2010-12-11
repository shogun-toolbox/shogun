//----------------------------------------------------------------------
//	File:           KMcenters.cc
//	Programmer:     David Mount
//	Last modified:  03/27/02
//	Description:    Functions for KMcenters
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

#include "KMcenters.h"
    					// standard constructor
KMcenters::KMcenters(int k, KMdata& p)
    : kCtrs(k), pts(&p) {
    ctrs = kmAllocPts(kCtrs, p.getDim());
}
    					// copy constructor
KMcenters::KMcenters(const KMcenters& s)
    : kCtrs(s.kCtrs), pts(s.pts) {
    ctrs = kmAllocCopyPts(kCtrs, s.getDim(), s.ctrs);
}
    					// assignment operator
KMcenters& KMcenters::operator=(const KMcenters& s) {
    if (this != &s) {			// avoid self assignment (x=x)
					// size change?
	if (kCtrs != s.kCtrs || getDim() != s.getDim()) {
	    kmDeallocPts(ctrs);		// reallocate points
	    ctrs = kmAllocPts(s.kCtrs, s.getDim());
	}
	kCtrs = s.kCtrs;
	pts = s.pts;
	kmCopyPts(kCtrs, s.getDim(), s.ctrs, ctrs);
    }
    return *this;
}

KMcenters::~KMcenters() {		// destructor
    kmDeallocPts(ctrs);
}

void KMcenters::resize(int k) {		// resize array (if needed)
    if (k == kCtrs) return;
    kCtrs = k;
    kmDeallocPts(ctrs);
    ctrs = kmAllocPts(kCtrs, pts->getDim());
}

void KMcenters::print(			// print centers
    bool fancy) {
    kmPrintPts("Center_Points", ctrs, getK(), fancy);
}

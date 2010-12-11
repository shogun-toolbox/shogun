//----------------------------------------------------------------------
//	File:           KMCenters.h
//	Programmer:     David Mount
//	Last modified:  03/27/02
//	Description:    Include file for KMcenters
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

#ifndef KM_CENTERS_H
#define KM_CENTERS_H

#include "KMeans.h"			// kmeans includes
#include "KMdata.h"			// provides KMdata

//----------------------------------------------------------------------
//  KMcenters - set of centers
//	This object encodes the information needed for describing a set
//	of centers.  It also stores a pointer to the data set.
//
//	When copying this object, we allocate new storage for the center
//	points, but we just copy the pointer to the data set.
//----------------------------------------------------------------------

class KMcenters {
protected:
    int			kCtrs;		// number of centers
    KMdata*		pts;		// the data points
    KMcenterArray	ctrs;		// the centers
public:					// constructors, etc.
    KMcenters(int k, KMdata& p);	// standard constructor
    KMcenters(const KMcenters& s);	// copy constructor
					// assignment operator
    KMcenters& operator=(const KMcenters& s);
    virtual ~KMcenters();		// virtual destructor
public:					// accessors
    int getDim() const {		// get dimension
	return pts->getDim();
    }
    int getNPts() const {		// get number of points
	return pts->getNPts();
    }
    int getK() const {			// get number of centers
	return kCtrs;
    }
    KMdata& getData() {			// get the data point structure
	return *pts;
    }
    KMpointArray getDataPts() const {	// get the data point array
	return pts->getPts();
    }
    KMcenterArray getCtrPts() const {	// get the center points
	return ctrs;
    }
    KMcenter& operator[](int i) {	// index centers
	return ctrs[i];
    }
    const KMcenter& operator[](int i) const {
	return ctrs[i];
    }
    void resize(int k);			// resize array

    virtual void print(			// print centers
        bool fancy = true);
};
#endif

//----------------------------------------------------------------------
//	File:           KMData.h
//	Programmer:     David Mount
//	Last modified:  03/27/02
//	Description:    Include file for KMdata
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

#ifndef KM_DATA_H
#define KM_DATA_H

#include "KMeans.h"			// kmeans includes
#include "KCtree.h"			// kc-tree includes

//----------------------------------------------------------------------
//  KMdata - data point set
//  	This object represents a set of data points in d-space.  The
//  	array can be resized.  Doing so destroys the existing contents.
// 
// 	In addition to the points, we also (optionally) provide a
// 	kc-tree data structure for the points as well.  This is
// 	constructed by first initializing the points and then calling
// 	buildKcTree().
//
// 	We support a virtual function samplePt and samplePts, which
// 	sample one or a set of random center points.  In this version,
// 	the sample is just a random sample of the point set.  However,
// 	it is possible to derive classes from this in which sampling is
// 	done by some more sophisticated method.
//
// 	Note that this structure does not support copying or
// 	assignments.  If you want to resuse the structure, the only way
// 	to do so is to first apply resize(), which destroys the kc-tree
// 	(if it exists), and then assign to it a new set of points.
//----------------------------------------------------------------------

class KMdata {
private:
    int			dim;		// dimension
    int			maxPts;		// max number of points
    int			nPts;		// number of data points
    KMdataArray		pts;		// the data points
    KCtree*		kcTree;		// kc-tree for the points
private:				// copy functions (not implemented)
    KMdata(const KMdata& p)		// copy constructor
      { assert(false); }
    KMdata& operator=(const KMdata& p)	// assignment operator
      { assert(false);  return *this; }
public:
    KMdata(int d, int n);		// standard constructor

    int getDim() const {		// get dimension
	return dim;
    }
    int getNPts() const {		// get number of points
	return nPts;
    }
    KMdataArray getPts() const {	// get the points
	return pts;
    }
    KCtree* getKcTree() const {		// get kc-tree
	return kcTree;
    }
    KMdataPoint& operator[](int i) {	// index
	return pts[i];
    }
    const KMdataPoint& operator[](int i) const {
	return pts[i];
    }
    void setNPts(int n) {		// set number of points
	assert(n <= maxPts);		// can't be more than array size
	nPts = n;
    }
    void buildKcTree();			// build the kc-tree for points

    virtual void sampleCtr(		// sample a center point
	KMpoint		sample);		// where to store sample

    virtual void sampleCtrs(		// sample center points
	KMpointArray	sample,			// where to store sample
	int		k,			// number of points to sample
	bool		allowDuplicate);	// allowing duplicates?

    void resize(int d, int n);		// resize array

    void print(				// print data points
    	bool		fancy = true) {		// nicely formatted?
	kmPrintPts("Data_Points", pts, nPts, dim, fancy);
    }

    virtual ~KMdata();			// destructor
};

typedef KMdata* KMdataPtr;		// pointer to KMdata
#endif

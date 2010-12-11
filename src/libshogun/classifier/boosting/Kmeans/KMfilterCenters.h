//----------------------------------------------------------------------
//	File:           KMfilterCenters.h
//	Programmer:     David Mount
//	Last modified:  08/10/2005
//	Description:    Include file for KMfilterCenters
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

#ifndef KM_FILTER_CENTERS_H
#define KM_FILTER_CENTERS_H

#include "KMcenters.h"			// provides KMcenters

//----------------------------------------------------------------------
//  KMfilterCenters
//	This extends the KMcenters class, by providing a more efficient
//	algorithm for computing distortions, by the filtering algorithm.
//	This algorithm makes use of a data structure called a kc-tree,
//	which is given in the file KCtree.h.  In addition to the
//	KMcenter functions we provide the following additional
//	functions:
//
//	getDist()
//	getDists()
//		Computes the total and individual distortions,
//		respectively, for the centers points (see definitions
//		below).
//	moveToCentroid()
//		Moves the center points to the centroids of their
//		associated neighborhoods.
//	getAssignments()
//		Computes the assignment of points to the closest center.
//
//	These functions are not computed independently.  In particular,
//	for a given set of centers, they can each be computed very
//	efficiently (in O(k*d) time) provided that some intermediate
//	values has already been computed.  We maintain a status variable
//	"valid," which indicates whether these intermediate values have
//	been computed and are current.  The intermediate values are
//	computed by computeDistortion().
//
//	>> If you modify this class note that any function that	<<
//	>> modifies the center or data points must set run	<<
//	>> invalidate() or equivalently, set valid=false.	<<
//
//	Immediate access:
//	-----------------
//	To disable the automatic recomputation of distortions on
//	getDist() and getDists(), call them with a "false" argument.
//
//	Distortion Overview:
//	--------------------
//	Let C[j] denote the j-th center.  For each j in [0..k-1], define
//	the j-th neighborhood V(j), to be the set of data points that
//	are closer to j than to any other center.  The "j-th distortion"
//	is defined to be the sum of squared distances of every point in
//	V(j) to the j-th center.  The "total distortion" is the sum of
//	the distortions over all the centers.
//
//	Intermediate Values:
//	--------------------
//	Instead of computing distortions from scratch by brute force
//	(which would take O(n*k*d) time), we use an algorithm called the
//	filtering algorithm.  This algorithm does not compute the
//	distortion directly, but instead computes the following
//	intermediate values, from which the distortion can be computed
//	efficiently.  Let j be an index in [0..k-1].  The notation (u.v)
//	denotes the dot product of vectors u and v.
//
//	KMpoint sums[j]		Vector sum of points in V(j)
//	double sumsSqs[j]	Sum of (u.u) for all u in V(j)
//	double weights[j]	Number of data points such that
//				  this C[j] is closest
//
//	See the function computeDistortion() and moveToCentroid() for
//	explanations of how these quantities are combined to compute
//	the total distortion and move centers to their centroids.
//
//	Final Values:
//	-------------
//	Given the above intermediate values, we then compute the
//	following final distortion values.
//
//	double dists[j]		Total distortion for points of V(j)
//	double currDist		Current total distortion
//
// 	Although they are not used by this program, the center
// 	distortions are useful, because they may be used in a more
// 	general clustering algorithm to determine whether clusters
// 	should be split or merged.
//----------------------------------------------------------------------

class KMfilterCenters : public KMcenters{
protected:			// intermediates
    KMpointArray	sums;		// vector sum of points
    double*		sumSqs;		// sum of squares
    int*		weights;	// the weight of each center
protected:			// distortion data
    double*		dists;		// individual distortions
    double		currDist;	// current total distortion
    bool		valid;		// are sums/distortions valid?
    double		dampFactor;	// dampening factor [0,1]
protected:			// local utilities
    void computeDistortion();		// compute distortions
    void moveToCentroid();		// move centers to cluster centroids
    					// swap one center
    void swapOneCenter(bool allowDuplicate = true);
    void validate()			// make valid
      { valid = true; }
    void invalidate() {			// make invalid
      if (kmStatLev >= CENTERS) print();// print centers
      valid = false;
    }
public:
    					// standard constructor
    KMfilterCenters(int k, KMdata& p, double df = 1);
					// copy constructor
    KMfilterCenters(const KMfilterCenters& s);
					// assignment operator
    KMfilterCenters& operator=(const KMfilterCenters& s);

    virtual ~KMfilterCenters();		// virtual destructor

public:					// public accessors
    					// returns sums
    KMpointArray getSums(bool autoUpdate = true) {
	if (autoUpdate && !valid) computeDistortion();
	return sums;
    }
    					// returns sums of squares
    double* getSumSqs(bool autoUpdate = true) {
	if (autoUpdate && !valid) computeDistortion();
	return sumSqs;
    }
    					// returns weights
    int* getWeights(bool autoUpdate = true) {
	if (autoUpdate && !valid) computeDistortion();
	return weights;
    }
					// returns total distortion
    double getDist(bool autoUpdate = true)	{
	if (autoUpdate && !valid) computeDistortion();
	return currDist;
    }
					// returns average distortion
    double getAvgDist(bool autoUpdate = true)	{
	if (autoUpdate && !valid) computeDistortion();
	return currDist/double(getNPts());
    }
					// returns individual distortions
    double* getDists(bool autoUpdate = true) {
	if (autoUpdate && !valid) computeDistortion();
	return dists;
    }

    void getAssignments(		// get point assignments
	KMctrIdxArray	closeCtr,		// closest center per point
	double*		sqDist);		// sq'd dist to center

    void genRandom() {			// generate random centers
	pts->sampleCtrs(ctrs, kCtrs, false);
	invalidate();
    }
    void lloyd1Stage() {		// one stage of LLoyd's algorithm
	moveToCentroid();
    }
    void swap1Stage() {			// one stage of swap heuristic
	swapOneCenter();
    }
    virtual void print(			// print centers
        bool fancy = true);
};
#endif

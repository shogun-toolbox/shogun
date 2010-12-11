//----------------------------------------------------------------------
//	File:           KMterm.cc
//	Programmer:     David Mount
//	Last modified:  03/27/02
//	Description:    Functions for KMterm.h
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

#include <cmath>			// math includes
#include "KMterm.h"

//----------------------------------------------------------------------
//  Default constructor
//  These are not reasonable values.  Use the standard constructor if
//  you want meaningful results.
//----------------------------------------------------------------------
KMterm::KMterm() {			// default constructor
    for (int i = 0; i < KM_TERM_VEC_LEN; i++) {
	maxTotStageVec[i] = 0;
    }
    minConsecRDL	= 0;
    minAccumRDL		= 0;
    maxRunStage		= 0;
    initProbAccept	= 0;
    tempRunLength	= 0;
    tempReducFact	= 0;
}

//----------------------------------------------------------------------
//  Standard constructor
//----------------------------------------------------------------------
KMterm::KMterm(				// standard constructor
	double a, double b, double c, double d,	// maxTotStage
	double mcr, double mar, int mrs,
	double ipa, int trl, double trf)
{
    maxTotStageVec[0] = a;	maxTotStageVec[1] = b;
    maxTotStageVec[2] = c;	maxTotStageVec[3] = d;
    minConsecRDL	= mcr;
    minAccumRDL		= mar;
    maxRunStage		= mrs;
    initProbAccept	= ipa;
    tempRunLength	= trl;
    tempReducFact	= trf;
}

int KMterm::maxStage(const double param[KM_TERM_VEC_LEN],
				int k, int n) const
{
    double count = param[KM_TERM_CONST];
    if (param[KM_TERM_POW] != 0) {
    	double sum = param[KM_TERM_LIN_K] * k + param[KM_TERM_LIN_N] * n;
	count += pow(sum, param[KM_TERM_POW]);
    }
    assert(count >= 0 && count <= INT_MAX);	// should be positive integer
    if (count <= 0) count = INT_MAX;		// 0 means infinity
    return int(count);
}

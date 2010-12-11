//----------------------------------------------------------------------
//	File:           KMterm.h
//	Programmer:     David Mount
//	Last modified:  03/27/02
//	Description:    Include file for kmeans algorithms.
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

#ifndef KM_TERM_H
#define KM_TERM_H

#include "KMeans.h"

//------------------------------------------------------------------------
//  KMterm - termination condition
//  	This structure is used for storing information controlling the
//  	termination and phase changes of the various algorithms.
//
//	Maximum Total Stages:
//	---------------------
//	The algorithm is terminated after some maximum number of stages
//	have been performed.  Rather than computing this as a fixed
//	constant, it is given as a function of n (number of data points)
//	and k (number of centers).  We use the following formula, where
//	the coefficients a, ..., d are supplied by the user (see
//	kmltest.cpp).
//
//		MAX_STAGE = a + (b*k + c*n)^d
//
//	Parameters Determining Phase/Run Transitions:
//	---------------------------------------------
//	The local improvement algorithms consist of a series of stages,
//	which are grouped into "runs" and series of runs are grouped in
//	"phases".  The meaning of these groupings depends on the
//	particular algorithm. (See KMlocal.h for more information.)  The
//	transition between runs and phases are either based on the
//	number of stages performed or on the change in distortion over
//	the course of the run.
//
//	maxRunStage
//		This is used to limit the maximum number of stages in
//		any run.
//
//	Some transitions are defined in terms of a quantity called the
//	"relative distortion loss" (RDL), which is defined to be the
//	relative decrease in the distortion.  (See KMlocal.h for
//	definition.) The relative distortion loss between the current
//	and previous stages is called the "consecutive RDL" and the
//	relative distortion loss since the start of a run is called the
//	"accumulated RDL".
//
//	minConsecRDL
//		This is used in the hybrid's algorithm.  If the RDL of
//		two consecutive runs is less than this value, Lloyd's
//		algorithm is deemed to have converged.
//	minAccumRDL
//		This is used in run-based algorithms.  It is the RDL of
//		the current distortion relative to the distortion at
//		some prior time (e.g. the start of a run).
//
//	Parameters used in Simulated Annealing
//	--------------------------------------
//	initProbAccept
//		Initial probability of accepting an solution that does
//		not alter the distortion.
//	tempRunLength
//		The number of stages before chaning the temperature.
//	tempReducFactor
//		The factor by which temperature is reduced at the end of
//		a temperature run.
//------------------------------------------------------------------------

enum {				// entry names
    KM_TERM_CONST,		// constant term
    KM_TERM_LIN_K,		// linear k multiplier
    KM_TERM_LIN_N,		// linear n multiplier
    KM_TERM_POW,		// power exponent
    KM_TERM_VEC_LEN};		// length of termination param vector

class KMterm {
private:
    double   maxTotStageVec[KM_TERM_VEC_LEN];	// max total stages
    double   minConsecRDL;			// min consecutive RDL
    double   minAccumRDL;			// min accumulated RDL
    int	     maxRunStage;			// max stages/run for Lloyd's
    double   initProbAccept;			// initial prob. of acceptance
    int      tempRunLength;			// length of temp run
    double   tempReducFact;			// temperature reduction factor

protected:					// stage count
    int maxStage(const double param[KM_TERM_VEC_LEN], int k, int n) const;
public:
    KMterm(); 					// default constructor
    KMterm(					// standard constructor
	double a, double b, double c, double d,	// maxTotStage
	double mcr, double mar, int mrs,
	double ipa, int trl, double trf);
    
    void setMaxTotStage(int i, double val) {	// set max stage parameters
    	assert(i >= 0 && i < KM_TERM_VEC_LEN);
	maxTotStageVec[i] = val;
    }
    void setAbsMaxTotStage(int s) {		// set max number of stages
    	maxTotStageVec[KM_TERM_CONST] = s;
	maxTotStageVec[KM_TERM_POW] = 0;
    }
    int getMaxTotStage(int k, int n) const	// max total stages
      {  return maxStage(maxTotStageVec, k, n); }

    double getMinConsecRDL() const		// return min consec RDL
      { return minConsecRDL; }

    double getMinAccumRDL() const		// return min accum RDL
      { return minAccumRDL; }

    int getMaxRunStage() const			// return max runs per stage
      { return maxRunStage; }

    void setMinConsecRDL(double rdl)		// set min consec RDL
      {  minConsecRDL = rdl; }

    void setMinAccumRDL(double rdl)		// set min accum RDL
      {  minAccumRDL = rdl; }

    void setMaxRunStage(int ms)			// set max runs per stage
      {  maxRunStage = ms; }

    double getInitProbAccept() const		// return init. prob. accept
      {  return initProbAccept; }

    void setInitProbAccept(double ipa)		// set init. prob. accept
      {  initProbAccept = ipa; }

    int getTempRunLength() const		// return temperature run len.
      { return tempRunLength; }

    void setTempRunLength(int trl)		// set temperature run length
      { tempRunLength = trl; }

    double getTempReducFact() const		// return temp. reduction fact.
      { return tempReducFact; }

    void setTempReducFact(double trf)		// set temp. reduction fact.
      { tempReducFact = trf; }
};
#endif

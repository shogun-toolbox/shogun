//----------------------------------------------------------------------
//	File:           KMlocal.cc
//	Programmer:     David Mount
//	Last modified:  03/27/02
//	Description:    k-means clustering by local search
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

#include "KMlocal.h"				// KMlocal includes

//----------------------------------------------------------------------
//  execute - execute the clustering algorithm
//  	This function executes the clustering algorithm.  See the file
//  	KMlocal.h for a description of this algorithm.
//----------------------------------------------------------------------

KMfilterCenters KMlocal::execute()		// execute the algorithm
{
   reset();					// resets everything
   while (!isDone()) {				// while not done
	beginRun();				// start a new run
	do {					// do while run is not done
	    beginStage();			// start of stage processing
	    KMalg method = selectMethod();	// select a method
	    switch(method) {			// apply one stage
	    case LLOYD:				// Lloyd's algorithm
		curr.lloyd1Stage();
		break;
	    case SWAP:				// swap heuristic
		curr.swap1Stage();
		break;
	    case RANDOM:			// get random centers
		curr.genRandom();
		break;
	    default:				// shouldn't come here
		assert(false);
		break;
	    }
	    endStage();				// end of stage processing
	} while (!isRunDone());			// while run is not done
	endRun();				// end of run processing
	tryAcceptance();			// accept if appropriate
    }
    return best;				// return best solution
}

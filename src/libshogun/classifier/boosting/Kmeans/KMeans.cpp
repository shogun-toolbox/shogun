//----------------------------------------------------------------------
//	File:           KMeans.cc
//	Programmer:     David Mount
//	Last modified:  05/14/04
//	Description:    Shared utilities for k-means.
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

#include <iostream>			// C++ I/O

#include "KMeans.h"			// kmeans includes
#include "KCtree.h"			// kc tree 
#include "KMrand.h"			// random number generators

//------------------------------------------------------------------------
//  Global data (shared by all files)
//	The following variables are used by all the procedures and are
//	initialized in kmInitGlobals().  kmInitTime is the CPU time
//	needed to initialize things before the first stage.
//------------------------------------------------------------------------

StatLev		kmStatLev	= SILENT;	// global stats output level
ostream*	kmOut		= &std::cout;	// standard output stream
ostream*	kmErr		= &std::cerr;	// output error stream
istream*	kmIn		= &std::cin;	// input stream

//----------------------------------------------------------------------
//  Output utilities
//----------------------------------------------------------------------

void kmPrintPt(				// print a point
    KMpoint		p,			// the point
    int			dim,			// the dimension
    bool		fancy)			// print plain or fancy?
{
    if (fancy) *kmOut << "[ ";
    for (int i = 0; i < dim; i++) {
	*kmOut << setw(8) << p[i];
	if (i < dim-1) *kmOut << " ";
    }
    if (fancy) *kmOut << " ]";
}

void kmPrintPts(			// print points
    string		title,			// name of point set
    KMpointArray	pa,			// the point array
    int			n,			// number of points
    int			dim,			// the dimension
    bool		fancy)		        // print plain or fancy?
{
    *kmOut << "  (" << title << ":\n";
    for (int i = 0; i < n; i++) {
	*kmOut << "    " << i << "\t";
	kmPrintPt(pa[i], dim, fancy);
	*kmOut << "\n";
    }
    *kmOut << "  )" << endl;
}

//------------------------------------------------------------------------
//  kmError - print error message
//  	If KMerr is KMabort we also abort the program.
//------------------------------------------------------------------------

void kmError(				// error routine
    const string	&msg,		// error message
    KMerr		level)		// abort afterwards
{
    if (level == KMabort) {
	*kmErr << "kmlocal: ERROR------->" << msg << "<-------------ERROR"
	       << endl;
	*kmOut << "kmlocal: ERROR------->" << msg << "<-------------ERROR"
	       << endl;
	kmExit(1);
    }
    else {
	*kmErr << "kmlocal: WARNING----->" << msg << "<-------------WARNING"
	       << endl;
	*kmOut << "kmlocal: WARNING----->" << msg << "<-------------WARNING"
	       << endl;
    }
}

//------------------------------------------------------------------------
//  kmExit - exit from program
//  	This is used because some Windows implementations create a
//	tempoarary window, which is removed immediately on exit.
//	This keeps until the user verifies termination.
//------------------------------------------------------------------------


void kmExit(int status)			// exit program
{
    #ifdef WAIT_FOR_CONFIRM
	char ch;
	if (kmIn == &cin) {			// input from std in
	    cerr << "Hit return to continue..." << endl;
	    kmIn->get(ch);
	}
    #endif
    exit(status);
}

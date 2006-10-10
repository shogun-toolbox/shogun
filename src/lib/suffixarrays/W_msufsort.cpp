/* ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is the Suffix Array based String Kernel.
 *
 * The Initial Developer of the Original Code is
 * Statistical Machine Learning Program (SML), National ICT Australia (NICTA).
 * Portions created by the Initial Developer are Copyright (C) 2006
 * the Initial Developer. All Rights Reserved.
 *
 * Contributor(s):
 *
 *   Choon Hui Teo <ChoonHui.Teo@rsise.anu.edu.au>
 *   S V N Vishwanathan <SVN.Vishwanathan@nicta.com.au>
 *
 * ***** END LICENSE BLOCK ***** */


// File    : sask/Code/W_msufsort.cpp
//
// Authors : Choon Hui Teo      (ChoonHui.Teo@rsise.anu.edu.au)
//           S V N Vishwanathan (SVN.Vishwanathan@nicta.com.au)
//
// Created : 09 Feb 2006
//
// Updated : 24 Apr 2006


//' Wrapper for Michael Maniscalco's MSufSort version 2.2 algorithm
#ifndef W_MSUFSORT_CPP
#define W_MSUFSORT_CPP

#include "W_msufsort.h"
#include <iostream>
#include <cstring>
#include <cassert>


W_msufsort::W_msufsort()
{
	msuffixsorter = new MSufSort();
}

W_msufsort::~W_msufsort()
{
	delete msuffixsorter;
}


/**
 *  Construct Suffix Array using Michael Maniscalco's algorithm
 *
 *  \param _text - (IN) The text which resultant SA corresponds to.
 *  \param _len  - (IN) The length of the text.
 *  \param _sa   - (OUT) Suffix array instance.
 */
ErrorCode
W_msufsort::ConstructSA(SYMBOL *text, const UInt32 &len, UInt32 *&array){

	//' A temporary copy of text
	SYMBOL *text_copy = new SYMBOL[len];

	//' chteo: BUGBUG
	//' redundant?
	assert(text_copy != NULL);

	memcpy(text_copy, text, sizeof(SYMBOL)*len);
	msuffixsorter->Sort(text_copy, len);
	
	//' Code adapted from MSufSort::verifySort()	
	for (UInt32 i = 0; i < len; i++) {
		UInt32 tmp = msuffixsorter->ISA(i)-1;
		array[tmp] = i;
	}

	//' Deallocate the memory allocated for #text_copy#
	delete [] text_copy;
	
	return NOERROR;
}
#endif




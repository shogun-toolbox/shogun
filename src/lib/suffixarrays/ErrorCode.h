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


// File    :  sask/Code/ErrorCode.cpp
//
// Authors :  Choon Hui Teo      (ChoonHui.Teo@rsise.anu.edu.au)
//            S V N Vishwanathan (SVN.Vishwanathan@nicta.com.au)
//
// Created :  09 Feb 2006
//
// Updated :  24 Apr 2006


#ifndef _ERRORCODE_H_
#define _ERRORCODE_H_

#include "DataType.h"
//#define DEBUG

// Levels of output info dumping
enum {QUIET, INFO, DEBUG};


#define ErrorCode           UInt32

/**
 *  for general use
 */
#define NOERROR             0
#define GENERAL_ERROR       1
#define MEM_ALLOC_FAILED    2
#define INVALID_PARAM       3
#define ARRAY_EMPTY         4
#define OPERATION_FAILED    5

/**
 * SuffixArray
 */
#define MATCH_NOT_FOUND     101
#define PARTIAL_MATCH       102

/**
 * LCP
 */
#define LCP_COMPACT_FAILED  201


/**
 * W_msufsort
 */



/**
 * W_kasai_lcp
 */

#define CHECKERROR(i)       {									                        \
	if((i) != NOERROR) {										                            \
		std::cout << "[CHECKERROR()]  Error! Code: " << (i) << std::endl; \
		exit(EXIT_FAILURE);										                            \
	}															                                      \
}


#define MESSAGE(msg) { std::cout<<(msg)<<std::endl; }

#endif

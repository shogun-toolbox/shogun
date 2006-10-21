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


// File    : sask/Code/W_kasai_lcp.h
//
// Authors : Choon Hui Teo      (ChoonHui.Teo@rsise.anu.edu.au)
//           S V N Vishwanathan (SVN.Vishwanathan@nicta.com.au)
//
// Created : 09 Feb 2006
//
// Updated : 24 Apr 2006


#ifndef W_KASAI_LCP_H
#define W_KASAI_LCP_H

#include "lib/suffixarray/DataType.h"
#include "lib/suffixarray/ErrorCode.h"
#include "lib/suffixarray/I_LCPFactory.h"
#include "lib/suffixarray/LCP.h"

/**
 * Kasai et al's LCP array computation algorithm is
 * is slightly faster than Manzini's algorithm. However,
 * it needs inverse suffix array which costs extra memory.
 */
class W_kasai_lcp : public I_LCPFactory
{

 public:

	/// Constructor
	W_kasai_lcp(){}

	/// Desctructor
	virtual ~W_kasai_lcp(){}

	/// Compute LCP array.
	ErrorCode	ComputeLCP(const SYMBOL *text, const UInt32 &len, 
											 const UInt32 *sa, LCP& lcp);

};
#endif
